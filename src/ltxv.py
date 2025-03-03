# Copyright 2024 Lightricks and The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
import os
import time
import argparse
import inspect
import random
import numpy as np
from PIL import Image

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('xformers').setLevel(logging.ERROR) # shutup triton, before torch!

import torch
import torch.nn.functional as F

from transformers import T5EncoderModel, T5TokenizerFast

from diffusers.loaders import FromSingleFileMixin, LTXVideoLoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKLLTXVideo
from diffusers.models.transformers import LTXVideoTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video, load_image
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
# STG
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_ltx import LTXVideoAttentionProcessor2_0, apply_rotary_emb

logging.getLogger('diffusers').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)

from core.args import main_args
from core.text import read_txt, txt_clean
from core.utils import calc_size, isset, clean_vram, img_list, basename, progbar, save_cfg

def get_args(parser):
    parser.add_argument('-vf', "--frames", default=121, type=int, help="Number of frames to generate in the output video")
    parser.add_argument("--fps", type=int, default=25, help="Frame rate for the output video")
    # STG, doesn't work here
    parser.add_argument('-ST', "--stg_scale", default=0, type=float, help="Spatiotemporal guidance scale for the pipeline")
    parser.add_argument('-STR', "--stg_rescale", default=0.7, type=float, help="Spatiotemporal guidance rescaling scale for the pipeline")
    parser.add_argument('-STL', "--stg_layers", default="11-19", help="Attention layers to skip for spatiotemporal guidance, separated by dash")
    parser.add_argument("--vae_t", default=0.05, type=float, help="Timestep for decoding noise")
    parser.add_argument("--vae_nscale", default=0.025, type=float, help="Noise level for decoding noise")
    # override
    parser.add_argument('-C', "--cfg_scale", default=6, type=float, help="Guidance scale for the pipeline")
    parser.add_argument('-sz', '--size', default=None, help="image size, multiple of 8")
    parser.add_argument('-s', "--steps", type=int, default=40, help="Number of inference steps")
    parser.add_argument('-un', "--unprompt", default="worst quality, blurry, jittery, distorted, inconsistent motion, cartoonish")
    return parser.parse_args()

is_mac = torch.backends.mps.is_available() and torch.backends.mps.is_built() # M1/M2 chip?
is_cuda = torch.cuda.is_available()
device = 'mps' if is_mac else 'cuda' if is_cuda else 'cpu'
dtype = torch.bfloat16 if is_cuda or is_mac else torch.float32

class Captur():
    def __init__(self, model="thwri/CogFlorence-2.2-Large"):
        from transformers import AutoModelForCausalLM, AutoProcessor # AutoConfig
        if not os.path.isdir(model): model = "thwri/CogFlorence-2.2-Large"
        self.model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    def __call__(self, image):
        prompt = "<MORE_DETAILED_CAPTION>"
        if image.mode != "RGB": image = image.convert("RGB")
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(device)
        generated_ids = self.model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3, do_sample=True)
        out_generated = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        out_parsed = self.processor.post_process_generation(out_generated, task=prompt, image_size=(image.width, image.height))
        return out_parsed[prompt].strip()

# STG
class LTXVideoSTGAttentionProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the LTX model. It applies a normalization layer and rotary embedding on the query and key vector.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("LTXVideoAttentionProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states_orig = hidden_states[:2]
        encoder_hidden_states_orig = encoder_hidden_states[:2]
        hidden_states_ptb = hidden_states[-1:]
        encoder_hidden_states_ptb = encoder_hidden_states[-1:]

        batch_size, sequence_length, _ = hidden_states_orig.shape if encoder_hidden_states is None else encoder_hidden_states_orig.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states_orig)
        key = attn.to_k(encoder_hidden_states_orig)
        value = attn.to_v(encoder_hidden_states_orig)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None: # [3, 8064, 2048]
            # from diffusers.models.embeddings import apply_rotary_emb
            image_rotary_emb_orig = tuple([x[:2] for x in image_rotary_emb])
            query = apply_rotary_emb(query, image_rotary_emb_orig)
            key = apply_rotary_emb(key, image_rotary_emb_orig)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        hidden_states_orig = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states_orig = hidden_states_orig.transpose(1, 2).flatten(2, 3)
        hidden_states_orig = hidden_states_orig.to(query.dtype)

        hidden_states_orig = attn.to_out[0](hidden_states_orig)
        hidden_states_orig = attn.to_out[1](hidden_states_orig)

        # perturbed part

        query = attn.to_q(hidden_states_ptb)
        key = attn.to_k(encoder_hidden_states_ptb)
        value = attn.to_v(encoder_hidden_states_ptb)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            image_rotary_emb_ptb = tuple([x[-1:] for x in image_rotary_emb])
            query = apply_rotary_emb(query, image_rotary_emb_ptb)
            key = apply_rotary_emb(key, image_rotary_emb_ptb)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        full_seq_length = query.size(2)
        identity_block_size = full_seq_length - text_seq_length
        full_mask = torch.zeros((full_seq_length, full_seq_length), device=query.device, dtype=query.dtype)
        full_mask[:identity_block_size, :identity_block_size] = float("-inf")
        full_mask[:identity_block_size, :identity_block_size].fill_diagonal_(0)
        full_mask = full_mask.unsqueeze(0).unsqueeze(0)
            
        hidden_states_ptb = F.scaled_dot_product_attention(query, key, value, attn_mask=full_mask, dropout_p=0.0, is_causal=False)
        hidden_states_ptb = hidden_states_ptb.transpose(1, 2).flatten(2, 3)
        hidden_states_ptb = hidden_states_ptb.to(query.dtype)

        hidden_states_ptb = attn.to_out[0](hidden_states_ptb)
        hidden_states_ptb = attn.to_out[1](hidden_states_ptb)

        hidden_states = torch.cat([hidden_states_orig, hidden_states_ptb], dim=0)
        return hidden_states

# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.16):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(scheduler, steps=None, device=None, timesteps=None, sigmas=None, **kwargs):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        steps = len(timesteps)
    else:
        scheduler.set_timesteps(steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, steps

def retrieve_latents(encoder_output, generator=None, sample_mode="sample"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class LTXPipeline(DiffusionPipeline, FromSingleFileMixin, LTXVideoLoraLoaderMixin):
    r""" Pipeline for text-to-video generation.
    Reference: https://github.com/Lightricks/LTX-Video
    Args:
        transformer ([`LTXVideoTransformer3DModel`]):
            Conditional Transformer architecture to denoise the encoded video lats.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image lats.
        vae ([`AutoencoderKLLTXVideo`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["lats", "cs", "uc"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLLTXVideo,
        text_encoder: T5EncoderModel,
        tokenizer: T5TokenizerFast,
        transformer: LTXVideoTransformer3DModel,
    ):
        super().__init__()

        self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=transformer, scheduler=scheduler)

        self.vae_spatial_compression_ratio = self.vae.spatial_compression_ratio if hasattr(self, "vae") else 32
        self.vae_temporal_compression_ratio = self.vae.temporal_compression_ratio if hasattr(self, "vae") else 8
        self.transformer_spatial_patch_size = self.transformer.config.patch_size if hasattr(self, "transformer") else 1
        self.transformer_temporal_patch_size = self.transformer.config.patch_size_t if hasattr(self, "transformer") else 1

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)
        self.tokenizer_max_length = self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 128

    def _get_t5_prompt_embeds(self, prompt=None, num=1, max_seq_len=128, device=None, dtype=None):
        device = device or self._execution_device
        dtype = self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        bs = len(prompt)

        text_inputs = self.tokenizer(prompt, padding="max_length", max_length=max_seq_len, truncation=True, add_special_tokens=True, return_tensors="pt")
        text_input_ids = text_inputs.input_ids
        p_attn_mask = text_inputs.attention_mask
        p_attn_mask = p_attn_mask.bool().to(device)

        cs = self.text_encoder(text_input_ids.to(device))[0]
        cs = cs.to(device, dtype=dtype)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = cs.shape
        cs = cs.repeat(1, num, 1).view(bs * num, seq_len, -1)
        p_attn_mask = p_attn_mask.view(bs, -1).repeat(num, 1)
        return cs, p_attn_mask

    # Copied from diffusers.pipelines.mochi.pipeline_mochi.MochiPipeline.encode_prompt with 256->128
    def encode_prompt(self, prompt, unprompt=None, cs=None, uc=None, p_attn_mask=None, unp_attn_mask=None, do_cfg=True, num=1, max_seq_len=128, device=None):
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        bs = len(prompt) if prompt is not None else cs.shape[0]
        if cs is None:
            cs, p_attn_mask = self._get_t5_prompt_embeds(prompt, num, max_seq_len, device)
        if do_cfg and uc is None:
            unprompt = unprompt or ""
            if isinstance(unprompt, str): unprompt = bs * [unprompt]
            uc, unp_attn_mask = self._get_t5_prompt_embeds(unprompt, num, max_seq_len, device)
        return cs, p_attn_mask, uc, unp_attn_mask

    @staticmethod
    def _pack_latents(lats, patch_size=1, patch_size_t=1):
        # Unpacked lats of shape are [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p, p].
        # The patch dimensions are then permuted and collapsed into the channel dimension of shape:
        # [B, F // p_t * H // p * W // p, C * p_t * p * p] (an ndim=3 tensor).
        # dim=0 is the batch size, dim=1 is the effective video sequence length, dim=2 is the effective number of input features
        bs, num_channels, frames, height, width = lats.shape
        post_patch_num_frames = frames // patch_size_t
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size
        lats = lats.reshape(bs, -1, post_patch_num_frames, patch_size_t, post_patch_height, patch_size, post_patch_width, patch_size)
        lats = lats.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        return lats

    @staticmethod
    def _unpack_latents(lats, frames, height, width, patch_size=1, patch_size_t=1):
        # Packed lats of shape [B, S, D] (S is the effective video sequence length, D is the effective feature dimensions)
        # are unpacked and reshaped into a video tensor of shape [B, C, F, H, W]. This is the inverse operation of
        # what happens in the `_pack_latents` method.
        bs = lats.size(0)
        lats = lats.reshape(bs, frames, height, width, -1, patch_size_t, patch_size, patch_size)
        lats = lats.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return lats

    @staticmethod
    def _normalize_latents(lats, latents_mean, latents_std, scaling_factor=1.): # Normalize lats across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(lats.device, lats.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(lats.device, lats.dtype)
        lats = (lats - latents_mean) * scaling_factor / latents_std
        return lats

    @staticmethod
    def _denormalize_latents(lats, latents_mean, latents_std, scaling_factor=1.): # Denormalize lats across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(lats.device, lats.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(lats.device, lats.dtype)
        lats = lats * latents_std / scaling_factor + latents_mean
        return lats

    def prepare_latents(self, image=None, bs=1, height=512, width=704, frames=161, dtype=None, device=None, generator=None, lats=None):
        if lats is not None:
            return lats.to(device, dtype=dtype)

        height = height // self.vae_spatial_compression_ratio
        width = width // self.vae_spatial_compression_ratio
        frames = (frames - 1) // self.vae_temporal_compression_ratio + 1
        shape = (bs, self.transformer.config.in_channels, frames, height, width)

        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        if image is None:
            lats = self._pack_latents(noise, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size)
            return lats, None

        else:
            init_lats = [retrieve_latents(self.vae.encode(img.unsqueeze(0).unsqueeze(2)), generator) for img in image]
            init_lats = torch.cat(init_lats, dim=0).to(dtype)
            init_lats = self._normalize_latents(init_lats, self.vae.latents_mean, self.vae.latents_std)
            init_lats = init_lats.repeat(1, 1, frames, 1, 1)
            cond_mask = torch.zeros((bs, 1, frames, height, width), device=device, dtype=dtype)
            cond_mask[:, :, 0] = 1.0

            lats = init_lats * cond_mask + noise * (1 - cond_mask)

            cond_mask = self._pack_latents(cond_mask, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size).squeeze(-1)
            lats = self._pack_latents(lats, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size)
            return lats, cond_mask

    @torch.no_grad()
    def __call__(self, prompt=None, unprompt=None, width=704, height=512, frames=161, steps=50, cfg_scale=3, stg_scale=0., stg_rescale=0.7, stg_layers=[19], 
        image=None, fps=25, num=1, generator=None, lats=None, cs=None, p_attn_mask=None, uc=None, unp_attn_mask=None, timesteps=None,
        vae_t=0.05, vae_nscale=0.025, output_type="pil", attention_kwargs=None, max_seq_len=128,
        # skip_layer_strategy=None, skip_block_list=None, image_cond_noise_scale=0., 
        do_rescaling=True):
        r"""
            p_attn_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for text embeddings.
            unp_attn_mask (`torch.FloatTensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            vae_t (`float`, defaults to `0.0`):
                The timestep at which generated video is decoded.
            vae_nscale (`float`, defaults to `None`):
                The interpolation factor between random noise and denoised lats at the decode timestep.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        """

        if prompt is not None and isinstance(prompt, str):
            bs = 1
        elif prompt is not None and isinstance(prompt, list):
            bs = len(prompt)
        else:
            bs = cs.shape[0]

        device = self._execution_device

        num_conds = 1
        do_cfg = cfg_scale > 1.
        if do_cfg: num_conds += 1

        do_stg = stg_scale > 0.
        if do_stg: 
            num_conds += 1
            layers = [(n, m) for n, m in self.transformer.named_modules() if "attn1" in n and "to" not in n and "add" not in n and "norm" not in n]
            # print([l[0] for l in layers])
            for i in range(len(layers)):
                layers[i][1].processor = LTXVideoAttentionProcessor2_0()
            replace_processor = LTXVideoSTGAttentionProcessor2_0()
            for i in stg_layers:
                layers[i][1].processor = replace_processor
            print(f" Replaced {stg_layers} layers with STG attn proc")

        cs, p_attn_mask, uc, unp_attn_mask = self.encode_prompt(prompt, unprompt, cs, uc, p_attn_mask, unp_attn_mask, do_cfg, num, max_seq_len, device)

        if image is not None:
            image = self.video_processor.preprocess(image, height=height, width=width)
            image = image.to(device=device, dtype=cs.dtype)

        csb = cs
        p_attn_mask_b = p_attn_mask
        if do_cfg:
            csb = torch.cat([uc, cs], dim=0)
            p_attn_mask_b = torch.cat([unp_attn_mask, p_attn_mask], dim=0)
        if do_stg:
            csb = torch.cat([csb, cs], dim=0)
            p_attn_mask_b = torch.cat([p_attn_mask_b, p_attn_mask], dim=0)

        lats, cond_mask = self.prepare_latents(image, bs * num, height, width, frames, torch.float32, device, generator, lats)

        if do_cfg and cond_mask is not None:
            cond_mask = torch.cat([cond_mask, cond_mask])
        if do_stg and cond_mask is not None:
            cond_mask = torch.cat([cond_mask, cond_mask[-1:]]) # by gemini

        # 5. Prepare timesteps
        lat_frames = (frames - 1) // self.vae_temporal_compression_ratio + 1
        latH = height // self.vae_spatial_compression_ratio
        latW = width // self.vae_spatial_compression_ratio
        video_seqlen = lat_frames * latH * latW
        sigmas = np.linspace(1.0, 1 / steps, steps)
        mu = calculate_shift(video_seqlen, self.scheduler.config.base_image_seq_len, self.scheduler.config.max_image_seq_len,
                             self.scheduler.config.base_shift, self.scheduler.config.max_shift)
        timesteps, steps = retrieve_timesteps(self.scheduler, steps, device, timesteps, sigmas=sigmas, mu=mu)
        num_warmup_steps = max(len(timesteps) - steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Prepare micro-conditions
        lat_fps = fps / self.vae_temporal_compression_ratio
        rope_interp_scale = (1 / lat_fps, self.vae_spatial_compression_ratio, self.vae_spatial_compression_ratio)

        with self.progress_bar(total=steps) as progress_bar:
            for i, t in enumerate(timesteps):

                lat_in = torch.cat([lats] * num_conds) if do_cfg else lats
                lat_in = lat_in.to(cs.dtype)
                timestep = t.expand(lat_in.shape[0]) # broadcast to batch dimension compatible with ONNX/Core ML
                if cond_mask is not None: # image in
                    timestep = timestep.unsqueeze(-1) * (1 - cond_mask)

                noise_pred = self.transformer(lat_in, csb, timestep, p_attn_mask_b, lat_frames, latH, latW, rope_interp_scale, attention_kwargs, return_dict=False)[0]
                noise_pred = noise_pred.float()

                if do_stg:
                    noise_pert = noise_pred[-1:]
                if do_cfg:
                    noise_un, noise_c = noise_pred[:2].chunk(2)
                    noise_pred = noise_un + cfg_scale * (noise_c - noise_un)
                if do_stg:
                    noise_pred = noise_pred + stg_scale * (noise_c - noise_pert)
                    if do_rescaling:
                        factor = noise_c.std() / noise_pred.std()
                        factor = stg_rescale * factor + (1 - stg_rescale)
                        noise_pred = noise_pred * factor

                if cond_mask is not None: # image in
                    timestep = timestep[:1]
                    noise_pred = self._unpack_latents(noise_pred, lat_frames, latH, latW, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size)
                    lats = self._unpack_latents(lats, lat_frames, latH, latW, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size)
                    noise_pred = noise_pred[:, :, 1:]
                    noise_lats = lats[:, :, 1:]
                    pred_lats = self.scheduler.step(noise_pred, t, noise_lats, return_dict=False)[0]
                    lats = torch.cat([lats[:, :, :1], pred_lats], dim=2)
                    lats = self._pack_latents(lats, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size)
                else:
                    lats = self.scheduler.step(noise_pred, t, lats, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self.transformer = self.transformer.to('cpu'); clean_vram()
        
        if output_type == "latent":
            video = lats
        else:
            lats = self._unpack_latents(lats, lat_frames, latH, latW, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size)
            lats = self._denormalize_latents(lats, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor)
            lats = lats.to(cs.dtype)

            if not self.vae.config.timestep_conditioning:
                timestep = None
            else:
                noise = torch.randn(lats.shape, generator=generator, device=device, dtype=lats.dtype)
                if not isinstance(vae_t, list):
                    vae_t = [vae_t] * bs
                if vae_nscale is None:
                    vae_nscale = vae_t
                elif not isinstance(vae_nscale, list):
                    vae_nscale = [vae_nscale] * bs

                timestep = torch.tensor(vae_t, device=device, dtype=lats.dtype)
                vae_nscale = torch.tensor(vae_nscale, device=device, dtype=lats.dtype)[:, None, None, None, None]
                lats = (1 - vae_nscale) * lats + vae_nscale * noise

            video = self.vae.decode(lats, timestep, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)

        self.maybe_free_model_hooks() # Offload all models

        return video


def main():
    a = get_args(main_args())
    os.makedirs(a.out_dir, exist_ok=True)
    if not isset(a, 'seed'): a.seed = int((time.time()%1)*69696)
    g_ = torch.Generator(device).manual_seed(a.seed)
    if a.verbose: save_cfg(a, a.out_dir)
    if a.verbose: print('.. cfg', a.cfg_scale, '.. stg', a.stg_scale, '..', a.seed)

    mod_path = os.path.join(a.maindir, 'xtra/ltxv')
    if not os.path.isdir(mod_path): mod_path = 'a-r-r-o-w/LTX-Video-0.9.1-diffusers'
    pipe = LTXPipeline.from_pretrained(mod_path, torch_dtype=dtype)
    pipe.to(device)

    count = 0
    if isset(a, 'in_img'):
        assert os.path.exists(a.in_img), "!! Image(s) %s not found !!" % a.in_img
        img_paths = img_list(a.in_img) if os.path.isdir(a.in_img) else [a.in_img]
        count = len(img_paths)

    if isset(a, 'in_txt'):
        prompts = read_txt(a.in_txt)
        assert (count==len(prompts) or count==0), "%d images but %d prompts!" % (len(img_paths), len(prompts))
        count = len(prompts)
    else:
        assert count > 0, "No images and no prompts?"
        promptr = Captur(os.path.join(a.maindir, 'xtra/cogflor22l'))

    pbar = progbar(count)
    for i in range(count):
        gendict = {'generator': g_}
        if a.stg_scale > 0:
            gendict = {**gendict, 'stg_scale': a.stg_scale, 'stg_rescale': a.stg_rescale, 'stg_layers': [int(i) for i in a.stg_layers.split('-')]}
        log = ['%03d' % i] if count > 1 else []

        if isset(a, 'in_img'):
            img_path = img_paths[i % len(img_paths)]
            log += ['%s' % basename(img_path)]
            img = load_image(img_path)
            if a.size is not None: img = img.resize([int(x) for x in a.size.split('-')], Image.LANCZOS)
            gendict['image'] = img
            W, H = img.size
        else:
            W, H = calc_size(a.size, quant=32) if a.size is not None else [848, 480] # or 768,512

        if isset(a, 'in_txt'):
            prompt = prompts[i % len(prompts)]
        else:
            prompt = promptr(gendict['image'])
            if a.verbose: print(prompt, '\n\n')
        log = '-'.join(log + [txt_clean(prompt)[:42]])
        file_out = '%s-%d' % (log, a.seed)
        
        video = pipe(prompt, a.unprompt, W, H, a.frames, a.steps, a.cfg_scale, **gendict, vae_t=a.vae_t, vae_nscale=a.vae_nscale)
        export_to_video(video[0], os.path.join(a.out_dir, file_out + ".mp4"), fps=a.fps)
        pbar.upd(log, uprows=1)
   

if __name__ == "__main__":
    main()
