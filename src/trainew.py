
import os, sys
import time
import random
import argparse
import numpy as np
from PIL import Image
from copy import deepcopy, copy

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import logging
logging.getLogger('xformers').setLevel(logging.ERROR) # shutup triton, before torch!
logging.getLogger('diffusers.models.modeling_utils').setLevel(logging.CRITICAL) # no bullshit about safetensors
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
warnings.filterwarnings('ignore', category=UserWarning, message='Using the model-agnostic')

import torch
import torch.nn as nn

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler

from core.finetune import Capturer, ConceptDataset, object_templates_edits
from core.utils import progbar, save_cfg, basename

parser = argparse.ArgumentParser()
# inputs
parser.add_argument('-t', '--token',    default=None, help="special word to invoke the embedding")
parser.add_argument('--term',           default=None, help="generic word, associating with that object or style")
parser.add_argument('-st', '--style',   action='store_true', help="True = style, False = object")
parser.add_argument('-m',  '--model',   default='15drm')
parser.add_argument('-md', '--maindir', default='./models', help='Main SD models directory')
parser.add_argument('-o', "--out_dir",  default="train", help="Output directory")
# train
parser.add_argument('-lr', "--lr",      default=1e-4, type=float, help="Initial learning rate (after the potential warmup period)")
parser.add_argument('-ts', '--train_steps', default=2500, type=int, help="Number of training steps")
parser.add_argument('-ups', '--upstep', default=250, type=int, help="how often to update terms and save files")
parser.add_argument('-b','--batch_size', default=1, type=int, help="batch size for training dataloader")
a = parser.parse_args()

device = torch.device('cuda')

class SDpipe(StableDiffusionPipeline):
    def __init__(self, vae, text_encoder, tokenizer, unet, scheduler, safety_checker=None, feature_extractor=None, requires_safety_checker=False):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, None, None, requires_safety_checker=False)
        self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler)

class Coach:
    def __init__(self, a):
        self.a = a

        self.model = self.load_model(a)
        self.token_id = self.update_tokenizer()
        self.set_model_gradient_flow()
        self.optimizer = self.get_optimizer()
        self.dataloader = self.get_dataloader()
        self.orig_params = self.model.text_encoder.get_input_embeddings().weight.data.clone()

        if a.upd_blip:
            self.captur = Capturer('t5xl')

    def load_model(self, a):
        if os.path.exists(a.model):
            SDload = StableDiffusionPipeline.from_single_file if os.path.isfile(a.model) else StableDiffusionPipeline.from_pretrained
            pipe = SDload(a.model, torch_dtype=torch.float16, variant='fp16', safety_checker=None, requires_safety_checker=False)
            pipe.to(device)
            text_encoder = pipe.text_encoder
            tokenizer    = pipe.tokenizer
            unet         = pipe.unet
            vae          = pipe.vae
            scheduler    = pipe.scheduler
        else:
            # paths
            subdir = 'v2' if a.model[0]=='2' else 'v1'
            txtenc_path = os.path.join(a.maindir, subdir, 'text-' + a.model[2:] if a.model[2:] in ['drm'] else 'text')
            sched_path = os.path.join(a.maindir, subdir, 'scheduler_config.json')
            unet_path = os.path.join(a.maindir, subdir, 'unet' + a.model)
            vae_path = os.path.join(a.maindir, subdir, 'vae')
            # load models
            text_encoder = CLIPTextModel.from_pretrained(txtenc_path, torch_dtype=torch.float32).to(device)
            tokenizer    = CLIPTokenizer.from_pretrained(txtenc_path, torch_dtype=torch.float16)
            unet  = UNet2DConditionModel.from_pretrained(unet_path,   torch_dtype=torch.float16).to(device)
            vae          = AutoencoderKL.from_pretrained(vae_path,    torch_dtype=torch.float16).to(device)
            scheduler    = DDIMScheduler.from_pretrained(sched_path)
            pipe = SDpipe(vae, text_encoder, tokenizer, unet, scheduler)
        return pipe

    def update_tokenizer(self):
        num_added_tokens = self.model.tokenizer.add_tokens(self.a.mod_token)
        mod_token_id = self.model.tokenizer.convert_tokens_to_ids(self.a.mod_token) # new token
        init_token_id = self.model.tokenizer.encode(self.a.term, add_special_tokens=False)[0] # existing token
        print(' tokens :: mod =', self.a.mod_token, mod_token_id, '.. init =', self.a.term, init_token_id)
        self.model.text_encoder.resize_token_embeddings(len(self.model.tokenizer)) # new token in tokenizer => resize token embeddings
        token_embeds = self.model.text_encoder.get_input_embeddings().weight.data
        token_embeds[mod_token_id] = deepcopy(token_embeds[init_token_id])
        return mod_token_id

    def set_model_gradient_flow(self):
        self.model.unet.to(dtype=torch.float16).requires_grad_(False)
        self.model.vae.to(dtype=torch.float16).requires_grad_(False)
        self.model.text_encoder.text_model.encoder.requires_grad_(False)
        self.model.text_encoder.text_model.final_layer_norm.requires_grad_(False)
        self.model.text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
        self.model.text_encoder.get_input_embeddings().weight.requires_grad_(True)
        self.model.text_encoder.to(dtype=torch.float32) # !!! must be trained as float32
        self.model.text_encoder.train()

    def get_optimizer(self):
        params_to_opt = self.model.text_encoder.get_input_embeddings().parameters()
        optimizer = torch.optim.AdamW(params=params_to_opt, lr=self.a.lr * self.a.batch_size, betas=(.9, .999), weight_decay=0.01, eps=1e-08)
        return optimizer

    def get_dataloader(self) -> torch.utils.data.DataLoader:
        dataset = ConceptDataset(token=self.a.mod_token, type = self.a.type)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.a.batch_size, shuffle=True, num_workers=self.a.num_workers)
        return dataloader

    @torch.no_grad()
    def caption(self, sampled_image) -> str:
        if self.a.type == 'style':
            question = f"What art style was used in this photo of a horse and a barn?" # must match 1st prompt at update, to avoid leaking to the answer
        else:
            question = f"What kind of {self.a.pos_terms[0]} is in this photo?"

        inputs = self.captur.processor(sampled_image, question, return_tensors="pt").to("cuda", torch.float16)
        neg_out = self.captur.processor.decode(self.captur.model.generate(**inputs)[0], skip_special_tokens=True)
        if neg_out.startswith("a "): neg_out = neg_out[2:]
        return neg_out

    def update(self, save_dir, save_prefix):
        if self.a.style:
            prompts = [f"a painting of a horse and a barn in a valley in the style of {self.a.mod_token}", # must match caption, to avoid leaking to neg_terms
                       f"a painting of a dog in the style of {self.a.mod_token}",
                       f"a painting of fruit bowl in the style of {self.a.mod_token}"]
        else:
            prompts = [f"a photo of a {self.a.mod_token}", # first prompt used by BLIP
                       f"Professional high-quality photo of a {self.a.mod_token}",
                       random.choice(object_templates_edits).format(a='a', token=self.a.mod_token)]

        clip_dtype = self.model.text_encoder.dtype
        pipetest = StableDiffusionPipeline(self.model.vae, self.model.text_encoder, self.model.tokenizer, self.model.unet, self.model.scheduler, \
                                           None, None, None, False).to(device, dtype=torch.float16)
        pipetest.set_progress_bar_config(disable=True)
        images = []
        img = None
        for prompt in prompts:
            imgs = [pipetest(prompt, num_inference_steps=23, guidance_scale=7.5, seed=self.a.seeds[i]).images[0] for i in range(len(self.a.seeds))]
            images += [np.vstack([np.array(img) for img in imgs])]
            if img is None: img = imgs[0]
        gen_images = np.hstack(images)
        Image.fromarray(gen_images).save(os.path.join(save_dir, f'{save_prefix}.jpg'))
        self.model.text_encoder.to(dtype = clip_dtype)
        return img

    def save_embeds(self, save_path):
        embeds = self.model.text_encoder.get_input_embeddings().weight[self.token_id]
        emb_dict = {self.a.mod_token: embeds.cpu().detach()}
        torch.save(emb_dict, save_path)

    @staticmethod
    def plot_distances(distances_log, output_path):
        classes = []
        for distances in distances_log:
            for curr_class in distances.keys():
                if curr_class not in classes:
                    classes.append(curr_class)
        # Generate a color for each class
        cmap = matplotlib.colormaps.get_cmap('tab20')
        colors = {class_name: cmap(i) for i, class_name in enumerate(classes)}
        plots = {}
        # Create a line plot for each class
        plt.figure(figsize=(10, 10))
        for class_name in classes:
            distances = [distances_log[i].get(class_name, None) for i in range(len(distances_log))]
            # Find first non-None index
            none_start_idx = next((i for i, d in enumerate(distances) if d is not None), None)
            distances = distances[none_start_idx:]
            # Smooth the data on a sliding window
            running_mean = np.convolve(distances, np.ones(10), 'valid') / 10
            running_mean = np.concatenate((distances[:9], running_mean))
            plt.plot(range(none_start_idx, len(distances) + none_start_idx), running_mean, color=colors[class_name], label=class_name, alpha=0.6)
            plots[class_name] = {'x': list(range(none_start_idx, len(distances) + none_start_idx)), 'y': running_mean.tolist()}
        # Set plot labels and legend
        plt.xlabel('Time')
        plt.ylabel('Similarity')
        plt.title('Similarity over Time')
        plt.legend()
        plt.savefig(output_path)

    def clip_emb_tok(self, prompt):
        tokens = self.model.tokenizer(prompt, padding="max_length", truncation=True, max_length=77, return_attention_mask=True, return_tensors="pt").to(device)
        emb = self.model.text_encoder(tokens.input_ids, attention_mask=tokens.attention_mask)[0] # [1,77,768]
        emb = emb[torch.arange(emb.shape[0]), tokens.input_ids.eq(self.token_id).nonzero()[:1, -1]] # [1,768]
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return emb # [1,768]

    @torch.no_grad()
    def clip_emb_pool(self, prompt):
        tokens = self.model.tokenizer(prompt, padding="max_length", truncation=True, max_length=77, return_attention_mask=True, return_tensors="pt").to(device)
        emb = self.model.text_encoder(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask, output_hidden_states=True).pooler_output
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return emb # [1,768]

    def train(self):
        sampled_image = self.update(save_dir=self.a.images_root, save_prefix=f'init')
        if self.a.upd_blip and len(self.a.neg_terms) == 0:
            self.a.neg_terms.append(self.caption(sampled_image))
        elif self.a.upd_fixed:
            random.shuffle(self.a.neg_terms)
            self.a.negative_pool = copy(self.a.neg_terms)
            self.a.neg_terms = [self.a.negative_pool.pop(0)]

        distances_log = []
        step = 0
        pbar = progbar(self.a.train_steps)
        while step < self.a.train_steps:
            for sample_idx, batch in enumerate(self.dataloader):

                txt_emb = self.clip_emb_tok(prompt=batch["text"][0])

                # Calculate distances
                distances_per_cls = {}
                if self.a.style:
                    assert len(self.a.pos_terms) == 1, "Style mode uses one placeholder positive class"
                    # For style mode we just use the prompt without specifying the style, so positive class is ignored
                    pos_prompts = [batch["template"][0].format(token='').replace('in the style of ', '')]
                else:
                    pos_prompts = [batch["template"][0].format(token=pos_word) for pos_word in self.a.pos_terms]

                pos_emb = self.clip_emb_pool(pos_prompts)
                pos_cossim = pos_emb.detach() @ txt_emb.T

                # Add distances to log
                for id_pos, pos_class in enumerate(self.a.pos_terms):
                    distances_per_cls[pos_class] = pos_cossim[id_pos].mean().item()

                # Restrict max cosine sim to optimize
                pos_cossim = torch.min(pos_cossim, torch.ones_like(pos_cossim) * self.a.max_cos)

                # Calc positive loss
                pos_loss: torch.Tensor = 0
                
                for id_pos, curr_pos_cosine_sim in enumerate(pos_cossim):
                    pos_loss += self.a.cws[id_pos] * (1 - curr_pos_cosine_sim)
                max_pos_cosine, pos_max_ind = pos_cossim.mean(dim=1).max(dim=0)

                neg_prompts = [batch["template"][0].format(token=neg_term) for neg_term in self.a.neg_terms]
                if len(neg_prompts) > 0:
                    # Calc distances to negative classes
                    neg_emb = self.clip_emb_pool(neg_prompts)
                    neg_cossim = neg_emb.detach() @ txt_emb.T

                    # Add distances to log
                    for id_neg, neg_class in enumerate(self.a.neg_terms):
                        distances_per_cls[neg_class] = neg_cossim[id_neg].mean().item()

                    # Restrict min cosine sim to optimize
                    neg_cossim = torch.max(neg_cossim, torch.ones_like(neg_cossim) * self.a.min_cos)

                    mean_neg_cosine = neg_cossim.mean()
                    max_neg_cosine, neg_max_ind = neg_cossim.mean(dim=1).max(dim=0)
                else:
                    mean_neg_cosine = 0
                    max_neg_cosine = 0

                distances_log.append(distances_per_cls)
                loss = pos_loss + self.a.force * 0.5 * (mean_neg_cosine + max_neg_cosine)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # We don't need update all embeddings weights. Only new embeddings.
                with torch.no_grad():
                    index_no_updates = torch.arange(len(self.model.tokenizer)) != self.token_id
                    self.model.text_encoder.get_input_embeddings().weight[index_no_updates] = self.orig_params[index_no_updates]

                step += 1 # distances are not ready at step 0

                if step % self.a.upstep == 0:
                    embed_save_path = os.path.join(self.a.out_dir, '%s-%04d.pt' % (self.a.token, step))
                    self.save_embeds(embed_save_path)
                    print(self.a.neg_terms)

                    stepnums = [i*self.a.upstep for i in range(len(self.a.neg_terms))]
                    lines = ['%04d\t%s\n' % (stepnums[i], self.a.neg_terms[i]) for i in range(len(stepnums))]
                    with open(os.path.join(self.a.out_dir, 'nega-%04d.txt' % step), 'w', encoding="utf-8") as f:
                        f.writelines(lines)

                    sampled_image = self.update(save_dir=self.a.images_root, save_prefix='test-%04d' % step)
                    figure_save_path = os.path.join(self.a.images_root, 'distance-%04d.jpg' % step)
                    self.plot_distances(distances_log=distances_log, output_path=figure_save_path)

                    if self.a.upd_blip:
                        self.a.neg_terms.append(self.caption(sampled_image))
                    elif self.a.upd_fixed:
                        if len(self.a.negative_pool) > 0:
                            self.a.neg_terms.append(self.a.negative_pool.pop(0))

                pbar.upd()

            embed_save_path = os.path.join(self.a.out_dir, '%s.pt' % self.a.token)
            self.save_embeds(embed_save_path)


def main():
    a.out_dir = os.path.join(a.out_dir, '%s-new-m%s' % (a.token, basename(a.model)))
    a.images_root = os.path.join(a.out_dir, "pix")
    os.makedirs(a.images_root, exist_ok=True)

    a.mod_token = '<%s>' % a.token
    a.pos_terms = [a.term] # Positive class, if not set use the same as init token. Not used for style mode!?
    a.neg_terms = [] # Negative prompts
    a.cws = None # Positive weights, if not set use 1/num_classes
    a.type = 'style' if a.style else 'object'
    a.force = 1. # Factor of loss between negative and positive constraints (how strong to push from the negatives)
    a.max_cos = 0.28 # Max cosine similarity to optimize for
    a.min_cos = 0.15 # Min cosine similarity to optimize for
    a.upd_blip = True
    a.upd_fixed = False # Whether to use gradual negatives
    a.seeds = [696,42] # Comma-separated seeds for reproducible inference.
    a.num_workers = 0 # Dataloader num workers.

    if len(a.pos_terms) == 0:
        print('Set positive class to init token')
        a.pos_terms = [a.term]

    if a.cws is None:
        a.cws = [1 / len(a.pos_terms) for _ in a.pos_terms]
    if len(a.pos_terms) != len(a.cws):
        raise ValueError('num cws != num pos_terms')

    save_cfg(a, a.out_dir)

    coach = Coach(a)
    coach.train()


if __name__ == '__main__':
    main()
