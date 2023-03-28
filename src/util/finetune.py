# partially reworked from https://github.com/adobe-research/custom-diffusion

import os
import PIL
import random 
import numpy as np
from pathlib import Path
from packaging import version
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from diffusers.models.attention import CrossAttention
from diffusers.models.cross_attention import LoRACrossAttnProcessor

PIL_INTERPOLATION = PIL.Image.Resampling.BICUBIC if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0") else PIL.Image.BICUBIC

templates = [
    "photo of {}",
    "image of {}",
    "picture of {}",
    "rendering of {}",
    "cool photo of {}",
    "good photo of {}",
    "lovely photo of {}",
    "nice picture of {}",
]
templates_style = [
    "image in the style of {}",
    "picture in the style of {}",
    "painting in the style of {}",
    "rendering in the style of {}",
    "rendition in the style of {}",
    "cool painting in the style of {}",
    "nice painting in the style of {}",
]

class Capturer():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.max_len = 32

        from transformers import AutoProcessor, BlipForConditionalGeneration # , Blip2ForConditionalGeneration
        CAPTION_MODELS = {
            'blip-base': 'Salesforce/blip-image-captioning-base',   # 990MB
            'blip-large': 'Salesforce/blip-image-captioning-large', # 1.9GB
            # 'blip2-2.7b': 'Salesforce/blip2-opt-2.7b',              # 15.5GB
            # 'blip2-flan-t5-xl': 'Salesforce/blip2-flan-t5-xl',      # 15.77GB
        }
        model_path = CAPTION_MODELS['blip-base'] # 'blip-large' is too imaginative
        model = BlipForConditionalGeneration.from_pretrained(model_path, torch_dtype=self.dtype) # Blip2ForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = model.eval().to(self.device)

    def __call__(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        inputs = inputs.to(self.dtype)
        tokens = self.model.generate(**inputs, max_new_tokens=self.max_len)
        return self.processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()

class FinetuneDataset(Dataset):
    def __init__(self, inputs, tokenizer, size=512, style=False, augment=True, add_caption=False, flip=True):
        self.size = size
        self.tokenizer = tokenizer
        self.style = style
        self.augment = augment
        self.add_caption = add_caption
        self.with_prior = 'term_data' in inputs[0]

        # pair = img + str
        self.instance_pairs = []
        self.class_pairs = []
        for batch in inputs: # may be miltiple 
            instance_pairs = [(x, batch["caption"]) for x in Path(batch["data"]).iterdir() if x.is_file()]
            self.instance_pairs.extend(instance_pairs)
            random.shuffle(self.instance_pairs)
            if self.with_prior:
                class_pairs = [(x, batch["term"]) for x in Path(batch["term_data"]).iterdir() if x.is_file()]
                self.class_pairs.extend(class_pairs)
                random.shuffle(self.class_pairs)
        
        if add_caption or any([p[1] is None for p in instance_pairs]):
            self.captur = Capturer()
        
        self.inst_len = len(self.instance_pairs)
        self._length = self.inst_len
        if self.with_prior:
            self.class_len = len(self.class_pairs)
            self._length = max(self.class_len, self._length)
        
        self.flip = transforms.RandomHorizontalFlip(0.5 * flip)

    def __len__(self):
        return self._length

    def pil_scale_np(self, image, size):
        image = image.resize((size, size), PIL_INTERPOLATION)
        return (np.array(image) / 127.5 - 1.0).astype(np.float32)
    
    def process_data(self, img_path, caption, aug_text=True):
        image = PIL.Image.open(img_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = self.flip(image)
        if self.augment:
            if np.random.randint(0, 3) < 2:
                random_scale = np.random.randint(self.size // 3, self.size + 1)
            else:
                random_scale = np.random.randint(self.size, int(1.5*self.size)) # orig 1.2~1.4
            if random_scale % 2 == 1:
                random_scale += 1
            if random_scale < 0.6 * self.size:
                cx, cy = np.random.randint(random_scale//2, self.size - random_scale//2 + 1, size=2)
                image_ = self.pil_scale_np(image, random_scale)
                image =  np.zeros((self.size, self.size, 3), dtype=np.float32)
                image[cx - random_scale//2 : cx + random_scale//2, cy - random_scale//2 : cy + random_scale//2, :] = image_
                epithet = np.random.choice(['far away ', 'distant ', 'small '])
            elif random_scale > 1.2 * self.size:
                cx, cy = np.random.randint(self.size//2, random_scale - self.size//2 + 1, size=2)
                image = self.pil_scale_np(image, random_scale)
                image = image[cx - self.size//2 : cx + self.size//2, cy - self.size//2 : cy + self.size//2, :]
                epithet = np.random.choice(['zoomed in ', 'close up ', 'large '])
            else:
                image = self.pil_scale_np(image, self.size)
                epithet = ''
            if aug_text and caption is not None:
                if self.style:
                    caption = np.random.choice(['', 'a ', 'the ']) + epithet + random.choice(templates_style).format(caption)
                else:
                    caption = np.random.choice(['', 'a ', 'the ']) + random.choice(templates).format(np.random.choice(['', 'a ', 'the ']) + epithet + caption)
        else:
            image = self.pil_scale_np(image, self.size)
        image_t = torch.from_numpy(image).permute(2,0,1)
        return image_t, caption
    
    def __getitem__(self, index):
        batch = {}
        instance_img_tensor, caption = self.process_data(*self.instance_pairs[index % self.inst_len])
        if caption is None:
            caption = self.captur(instance_img_tensor)
        elif self.add_caption:
            caption += ', ' + self.captur(instance_img_tensor)
        batch["instance_image"] = instance_img_tensor
        batch["instance_token_id"] = self.tokenizer(caption, padding="do_not_pad", truncation=True, max_length=self.tokenizer.model_max_length).input_ids
        if self.with_prior:
            class_img_tensor, term = self.process_data(*self.class_pairs[index % self.class_len], aug_text=False)
            batch["class_image"] = class_img_tensor
            batch["class_token_id"] = self.tokenizer(term, padding="do_not_pad", truncation=True, max_length=self.tokenizer.model_max_length).input_ids
        return batch


def custom_diff(unet, freeze_model):
    for name, params in unet.named_parameters():
        if freeze_model == 'crossattn':
            selected = 'attn2' in name
        else: # crossattn_kv
            selected = 'attn2.to_k' in name or 'attn2.to_v' in name
        params.requires_grad = True if selected else False

    def new_forward(self, hidden_states, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        crossattn = False
        context = None
        if 'context' in kwargs:
            context = kwargs['context']
        elif 'encoder_hidden_states' in kwargs:
            context = kwargs['encoder_hidden_states']
        if context is not None:
            crossattn = True

        q = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        k = self.to_k(context)
        v = self.to_v(context)

        if crossattn:
            modifier = torch.ones_like(k)
            modifier[:, :1, :] = modifier[:, :1, :]*0.
            k = modifier*k + (1-modifier)*k.detach()
            v = modifier*v + (1-modifier)*v.detach()

        dim = q.shape[-1]

        """ original custom diffusion, for OLD diffusers?
        q = self.reshape_heads_to_batch_dim(q)
        k = self.reshape_heads_to_batch_dim(k)
        v = self.reshape_heads_to_batch_dim(v)
        if self._use_memory_efficient_attention_xformers: 
            hidden_states = self._memory_efficient_attention_xformers(q, k, v)
            hidden_states = hidden_states.to(q.dtype) # Some versions of xformers return fp32, cast it back to the input dtype
        else:
            if self._slice_size is None or q.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(q, k, v)
            else:
                hidden_states = self._sliced_attention(q, k, v, sequence_length, dim)
        """

        # fix for NEW diffusers = old self._attention from 0.11.1
        q = self.head_to_batch_dim(q)
        k = self.head_to_batch_dim(k)
        v = self.head_to_batch_dim(v)
        if self.upcast_attention:
            q, k = q.float(), k.float()
        attn_scores = torch.baddbmm(torch.empty(q.shape[0], q.shape[1], k.shape[1], dtype=q.dtype, device=q.device), q, k.transpose(-1,-2), 
                                    beta=0, alpha=self.scale)
        if self.upcast_softmax:
            attn_scores = attn_scores.float()
        attention_probs = attn_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(v.dtype)
        hidden_states = torch.bmm(attention_probs, v) # compute attention output
        hidden_states = self.batch_to_head_dim(hidden_states)

        hidden_states = self.to_out[0](hidden_states) # linear proj
        hidden_states = self.to_out[1](hidden_states) # dropout
        return hidden_states

    def change_forward(unet):
        for layer in unet.children():
            if type(layer) == CrossAttention:
                bound_method = new_forward.__get__(layer, layer.__class__)
                setattr(layer, 'forward', bound_method)
            else:
                change_forward(layer)

    change_forward(unet)
    return unet


def save_embeds(save_path, text_encoder, tokens, tokens_id, accelerator=None):
    if accelerator is not None:
        text_encoder = accelerator.unwrap_model(text_encoder)
    assert len(tokens) == len(tokens_id)
    learned_embeds_dict = {}
    for token, token_id in zip(tokens, tokens_id):
        learned_embeds = text_encoder.get_input_embeddings().weight[token_id]
        learned_embeds_dict[token] = learned_embeds.detach().cpu()
    torch.save(learned_embeds_dict, save_path)

def load_embeds(st_dict, text_encoder, tokenizer, token_str=None):
    mod_tokens = list(st_dict.keys()) # saved tokens
    if token_str is not None and len(token_str) > 0:
        new_tokens = ['<%s>' % t for t in token_str.split('+')]
        assert len(new_tokens) == len(mod_tokens), "!! cannot replace saved tokens with new: %d vs %d" % (len(new_tokens), len(mod_tokens))
    else:
        new_tokens = mod_tokens
    mod_tokens_id = []
    for mod_token in new_tokens:
        _ = tokenizer.add_tokens(mod_token)
        mod_tokens_id.append(tokenizer.convert_tokens_to_ids(mod_token))
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    for i, id_new in enumerate(mod_tokens_id):
        token_embeds[id_new] = st_dict[mod_tokens[i]]
    return new_tokens

def save_delta(save_path, unet, text_encoder, mod_tokens=None, mod_tokens_id=None, freeze_model='crossattn_kv', unet0=None, save_txt_enc=False, accelerator=None):
    if accelerator is not None:
        text_encoder = accelerator.unwrap_model(text_encoder)
        unet = accelerator.unwrap_model(unet)
    delta_dict = {'unet': {}, 'modifier_token': {}}
    if mod_tokens is not None:
        for i in range(len(mod_tokens_id)):
            learned_embeds = text_encoder.get_input_embeddings().weight[mod_tokens_id[i]]
            delta_dict['modifier_token'][mod_tokens[i]] = learned_embeds.detach().cpu()
    elif save_txt_enc:
        delta_dict['text_encoder'] = text_encoder.state_dict()
    for name, params in unet.named_parameters():
        if freeze_model == 'crossattn':
            if 'attn2' in name:
                delta_dict['unet'][name] = params.cpu().clone()
        else:
            if 'attn2.to_k' in name or 'attn2.to_v' in name:
                delta_dict['unet'][name] = params.cpu().clone()
    if unet0 is not None: # we need original pretrained model to compress delta
        delta_dict = compress_delta(delta_dict, unet0, compression_ratio = 0.6) # compressed delta is on cuda, uncompressed on cpu
    torch.save(delta_dict, save_path)

def load_delta(st, unet, text_encoder, tokenizer, token_str=None, compress=True, freeze_model='crossattn_kv'):
    if 'text_encoder' in st:
        text_encoder.load_state_dict(st['text_encoder'])
    if 'modifier_token' in st:
        new_tokens = load_embeds(st['modifier_token'], text_encoder, tokenizer, token_str)
    for name, params in unet.named_parameters():
        if freeze_model == 'crossattn':
            if 'attn2' in name:
                if compress and ('to_k' in name or 'to_v' in name):
                    params.data += st['unet'][name]['u']@st['unet'][name]['v']
                else:
                    params.data.copy_(st['unet'][f'{name}'])
        else:
            if 'attn2.to_k' in name or 'attn2.to_v' in name:
                if compress:
                    params.data += st['unet'][name]['u']@st['unet'][name]['v']
                else:
                    params.data.copy_(st['unet'][f'{name}'])
    return new_tokens if 'modifier_token' in st else None

def compress_delta(delta_dict, unet, compression_ratio = 0.6):
    compressed_st = {'unet': {}}
    pretrained_st = unet.state_dict()
    st = delta_dict
    if 'modifier_token' in st:
        compressed_st['modifier_token'] = st['modifier_token']
    if 'text_encoder' in st:
        compressed_st['text_encoder'] = st['text_encoder']
    st = st['unet']
    layers = list(st.keys())
    for name in layers:
        if 'to_k' in name or 'to_v' in name:
            W = st[name].detach().to(unet.device)
            Wpretrain = pretrained_st[name]
            deltaW = W - Wpretrain
            u, s, vt = torch.linalg.svd(deltaW.cpu()) # OOM if on cuda during training
            explain = 0
            all_ = (s).sum()
            for i, t in enumerate(s):
                explain += t/(all_)
                if explain > compression_ratio:
                    break
            compressed_st['unet'][f'{name}'] = {}
            compressed_st['unet'][f'{name}']['u'] = (u[:, :i]@torch.diag(s)[:i, :i]).clone().cuda()
            compressed_st['unet'][f'{name}']['v'] = vt[:i].clone().cuda()
        else:
            compressed_st['unet'][f'{name}'] = st[name]
    return compressed_st

def save_lora(save_path, lora_dict, text_encoder, mod_tokens=None, mod_tokens_id=None, save_txt_enc=False, accelerator=None):
    if accelerator is not None:
        text_encoder = accelerator.unwrap_model(text_encoder)
        unet = accelerator.unwrap_model(unet)
    out_dict = {}
    if mod_tokens is not None:
        out_dict['modifier_token'] = {}
        for i in range(len(mod_tokens_id)):
            learned_embeds = text_encoder.get_input_embeddings().weight[mod_tokens_id[i]]
            out_dict['modifier_token'][mod_tokens[i]] = learned_embeds.detach().cpu()
    elif save_txt_enc:
        out_dict['text_encoder'] = text_encoder.state_dict()
    if len(out_dict) > 0: # with embeddings/encoder
        out_dict['unet'] = lora_dict
    else: # only lora, as in the original
        out_dict = lora_dict
    torch.save(out_dict, save_path)

def load_loras(st, unet, text_encoder=None, tokenizer=None, token_str=None):
    new_tokens = None
    if 'text_encoder' in st and text_encoder is not None:
        text_encoder.load_state_dict(st['text_encoder'])
    if 'modifier_token' in st and tokenizer is not None:
        new_tokens = load_embeds(st['modifier_token'], text_encoder, tokenizer, token_str)
    if 'unet' in st:
        st = st['unet']
    assert all("lora" in k for k in st.keys()), " !! loaded file does not look like LoRA !!"
    attn_processors = {}
    lora_grouped_dict = defaultdict(dict)
    for key, value in st.items():
        attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
        lora_grouped_dict[attn_processor_key][sub_key] = value
    for key, value_dict in lora_grouped_dict.items():
        rank = value_dict["to_k_lora.down.weight"].shape[0]
        cross_attention_dim = value_dict["to_k_lora.down.weight"].shape[1]
        hidden_size = value_dict["to_k_lora.up.weight"].shape[0]
        attn_processors[key] = LoRACrossAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank)
        attn_processors[key].load_state_dict(value_dict)
    attn_processors = {k: v.to(device=unet.device, dtype=unet.dtype) for k, v in attn_processors.items()}
    unet.set_attn_processor(attn_processors)
    return new_tokens

