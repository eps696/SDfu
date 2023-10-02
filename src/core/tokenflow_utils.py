# edited from https://github.com/omerbt/TokenFlow

from typing import Type

import torch

def batch_cosine_sim(x, y):
    if type(x) is list:
        x = torch.cat(x, dim=0)
    if type(y) is list:
        y = torch.cat(y, dim=0)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    similarity = x @ y.T
    return similarity

def isinstance_str(x: object, cls_name: str):
    """ Checks whether x has any class *named* cls_name in its ancestry. Doesn't require access to the class's implementation """
    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    return False

def reg_var(diffusion_model, name, value):
    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"): # name may change!?
            setattr(module, name, value)

def get_var(diffusion_model, name):
    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock") and hasattr(module, name): # name may change!?
            return getattr(module, name)
            
def reg_time(model, t):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 't', t)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 't', t)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn2
    setattr(module, 't', t)

def reg_conv_injection(model, injection_schedule):
    def conv_forward(self):
        def forward(input_tensor, temb, scale=None): # !!! scale arg required for new diffusers
            hidden_states = input_tensor
            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if temb is not None:
                temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)
            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                source_batch_size = int(hidden_states.shape[0] // 3)
                # inject unconditional
                hidden_states[source_batch_size:2 * source_batch_size] = hidden_states[:source_batch_size]
                # inject conditional
                hidden_states[2 * source_batch_size:] = hidden_states[:source_batch_size]

            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            return output_tensor

        return forward

    conv_module = model.unet.up_blocks[1].resnets[1]
    conv_module.forward = conv_forward(conv_module)
    setattr(conv_module, 'injection_schedule', injection_schedule)

def reg_extended_attention_pnp(model, injection_schedule):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, seqlen, dim = x.shape
            h = self.heads
            fnum = batch_size // 3
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            q = self.to_q(x)
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)

            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                # inject unconditional
                q[fnum:2 * fnum] = q[:fnum]
                k[fnum:2 * fnum] = k[:fnum]
                # inject conditional
                q[2 * fnum:] = q[:fnum]
                k[2 * fnum:] = k[:fnum]

            k_source = k[:fnum]
            k_uncond = k[fnum:2 * fnum].reshape(1, fnum * seqlen, -1).repeat(fnum, 1, 1)
            k_cond = k[2 * fnum:].reshape(1, fnum * seqlen, -1).repeat(fnum, 1, 1)

            v_source = v[:fnum]
            v_uncond = v[fnum:2 * fnum].reshape(1, fnum * seqlen, -1).repeat(fnum, 1, 1)
            v_cond = v[2 * fnum:].reshape(1, fnum * seqlen, -1).repeat(fnum, 1, 1)

            q_source = self.head_to_batch_dim(q[:fnum])
            q_uncond = self.head_to_batch_dim(q[fnum:2 * fnum])
            q_cond = self.head_to_batch_dim(q[2 * fnum:])
            k_source = self.head_to_batch_dim(k_source)
            k_uncond = self.head_to_batch_dim(k_uncond)
            k_cond = self.head_to_batch_dim(k_cond)
            v_source = self.head_to_batch_dim(v_source)
            v_uncond = self.head_to_batch_dim(v_uncond)
            v_cond = self.head_to_batch_dim(v_cond)

            q_src = q_source.view(fnum, h, seqlen, dim // h)
            k_src = k_source.view(fnum, h, seqlen, dim // h)
            v_src = v_source.view(fnum, h, seqlen, dim // h)
            q_uncond = q_uncond.view(fnum, h, seqlen, dim // h)
            k_uncond = k_uncond.view(fnum, h, seqlen * fnum, dim // h)
            v_uncond = v_uncond.view(fnum, h, seqlen * fnum, dim // h)
            q_cond = q_cond.view(fnum, h, seqlen, dim // h)
            k_cond = k_cond.view(fnum, h, seqlen * fnum, dim // h)
            v_cond = v_cond.view(fnum, h, seqlen * fnum, dim // h)

            out_source_all = []
            out_uncond_all = []
            out_cond_all = []
            
            single_batch = fnum<=12
            b = fnum if single_batch else 1

            for frame in range(0, fnum, b):
                out_source = []
                out_uncond = []
                out_cond = []
                for j in range(h):
                    sim_source_b = torch.bmm(q_src[frame: frame+ b, j], k_src[frame: frame+ b, j].transpose(-1, -2)) * self.scale
                    sim_uncond_b = torch.bmm(q_uncond[frame: frame+ b, j], k_uncond[frame: frame+ b, j].transpose(-1, -2)) * self.scale
                    sim_cond = torch.bmm(q_cond[frame: frame+ b, j], k_cond[frame: frame+ b, j].transpose(-1, -2)) * self.scale

                    out_source.append(torch.bmm(sim_source_b.softmax(dim=-1), v_src[frame: frame+ b, j]))
                    out_uncond.append(torch.bmm(sim_uncond_b.softmax(dim=-1), v_uncond[frame: frame+ b, j]))
                    out_cond.append(torch.bmm(sim_cond.softmax(dim=-1), v_cond[frame: frame+ b, j]))

                out_source = torch.cat(out_source, dim=0)
                out_uncond = torch.cat(out_uncond, dim=0) 
                out_cond = torch.cat(out_cond, dim=0) 
                if single_batch:
                    out_source = out_source.view(h, fnum,seqlen, dim // h).permute(1, 0, 2, 3).reshape(h * fnum, seqlen, -1)
                    out_uncond = out_uncond.view(h, fnum,seqlen, dim // h).permute(1, 0, 2, 3).reshape(h * fnum, seqlen, -1)
                    out_cond = out_cond.view(h, fnum,seqlen, dim // h).permute(1, 0, 2, 3).reshape(h * fnum, seqlen, -1)
                out_source_all.append(out_source)
                out_uncond_all.append(out_uncond)
                out_cond_all.append(out_cond)
            
            out_source = torch.cat(out_source_all, dim=0)
            out_uncond = torch.cat(out_uncond_all, dim=0)
            out_cond = torch.cat(out_cond_all, dim=0)
                
            out = torch.cat([out_source, out_uncond, out_cond], dim=0)
            out = self.batch_to_head_dim(out)

            return to_out(out)

        return forward

    for _, module in model.unet.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"): # name may change!?
            module.attn1.forward = sa_forward(module.attn1)
            setattr(module.attn1, 'injection_schedule', [])

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)

def reg_extended_attention(model):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, seqlen, dim = x.shape
            h = self.heads
            fnum = batch_size // 3
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            q = self.to_q(x)
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)

            k_source = k[:fnum]
            k_uncond = k[fnum: 2*fnum].reshape(1, fnum * seqlen, -1).repeat(fnum, 1, 1)
            k_cond = k[2*fnum:].reshape(1, fnum * seqlen, -1).repeat(fnum, 1, 1)
            v_source = v[:fnum]
            v_uncond = v[fnum:2*fnum].reshape(1, fnum * seqlen, -1).repeat(fnum, 1, 1)
            v_cond = v[2*fnum:].reshape(1, fnum * seqlen, -1).repeat(fnum, 1, 1)

            q_source = self.head_to_batch_dim(q[:fnum])
            q_uncond = self.head_to_batch_dim(q[fnum: 2*fnum])
            q_cond = self.head_to_batch_dim(q[2 * fnum:])
            k_source = self.head_to_batch_dim(k_source)
            k_uncond = self.head_to_batch_dim(k_uncond)
            k_cond = self.head_to_batch_dim(k_cond)
            v_source = self.head_to_batch_dim(v_source)
            v_uncond = self.head_to_batch_dim(v_uncond)
            v_cond = self.head_to_batch_dim(v_cond)

            out_source = []
            out_uncond = []
            out_cond = []

            q_src = q_source.view(fnum, h, seqlen, dim // h)
            k_src = k_source.view(fnum, h, seqlen, dim // h)
            v_src = v_source.view(fnum, h, seqlen, dim // h)
            q_uncond = q_uncond.view(fnum, h, seqlen, dim // h)
            k_uncond = k_uncond.view(fnum, h, seqlen * fnum, dim // h)
            v_uncond = v_uncond.view(fnum, h, seqlen * fnum, dim // h)
            q_cond = q_cond.view(fnum, h, seqlen, dim // h)
            k_cond = k_cond.view(fnum, h, seqlen * fnum, dim // h)
            v_cond = v_cond.view(fnum, h, seqlen * fnum, dim // h)

            for j in range(h):
                sim_source_b = torch.bmm(q_src[:, j], k_src[:, j].transpose(-1, -2)) * self.scale
                sim_uncond_b = torch.bmm(q_uncond[:, j], k_uncond[:, j].transpose(-1, -2)) * self.scale
                sim_cond = torch.bmm(q_cond[:, j], k_cond[:, j].transpose(-1, -2)) * self.scale

                out_source.append(torch.bmm(sim_source_b.softmax(dim=-1), v_src[:, j]))
                out_uncond.append(torch.bmm(sim_uncond_b.softmax(dim=-1), v_uncond[:, j]))
                out_cond.append(torch.bmm(sim_cond.softmax(dim=-1), v_cond[:, j]))

            out_source = torch.cat(out_source, dim=0).view(h, fnum,seqlen, dim // h).permute(1, 0, 2, 3).reshape(h * fnum, seqlen, -1)
            out_uncond = torch.cat(out_uncond, dim=0).view(h, fnum,seqlen, dim // h).permute(1, 0, 2, 3).reshape(h * fnum, seqlen, -1)
            out_cond = torch.cat(out_cond, dim=0).view(h, fnum,seqlen, dim // h).permute(1, 0, 2, 3).reshape(h * fnum, seqlen, -1)

            out = torch.cat([out_source, out_uncond, out_cond], dim=0)
            out = self.batch_to_head_dim(out)

            return to_out(out)

        return forward

    for _, module in model.unet.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"): # name may change!?
            module.attn1.forward = sa_forward(module.attn1)

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)

def make_tokenflow_attention_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:

    class TokenFlowBlock(block_class):

        def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, timestep=None,
            cross_attention_kwargs=None, class_labels=None) -> torch.Tensor:
            
            batch_size, seqlen, dim = hidden_states.shape
            fnum = batch_size // 3 # framecount for current pass = batchcount for pivots, curr.batch for lats
            hidden_states = hidden_states.view(3, fnum, seqlen, dim)

            if self.use_ada_layer_norm:
                norm_hid_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hid_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype)
            else:
                norm_hid_states = self.norm1(hidden_states)

            norm_hid_states = norm_hid_states.view(3, fnum, seqlen, dim)
            if self.pivotal_pass: # pre pass
                if self.batch_pivots == True: # EXTERNAL OFFLOAD
                    self.pivot_hidden_states += [norm_hid_states]
                else:
                    self.pivot_hidden_states = norm_hid_states # [3, fnum, seqlen, dim]
            else: # gen pass
                idx1 = []
                idx2 = [] 
                b_ids = [self.batch_idx] # batch_ids
                if self.batch_idx > 0:
                    b_ids.append(self.batch_idx - 1)

                if self.batch_pivots == True: # EXTERNAL OFFLOAD
                    sim = batch_cosine_sim(norm_hid_states[0].reshape(-1, dim), self.pivot_hidden_states[self.layer_idx[0]][0][b_ids].reshape(-1, dim))
                else:
                    sim = batch_cosine_sim(norm_hid_states[0].reshape(-1, dim), self.pivot_hidden_states[0][b_ids].reshape(-1, dim))

                if len(b_ids) == 2:
                    sim1, sim2 = sim.chunk(2, dim=1) # [fnum * seqlen, seqlen]
                    idx1.append(sim1.argmax(dim=-1))
                    idx2.append(sim2.argmax(dim=-1))
                else:
                    idx1.append(sim.argmax(dim=-1))
                idx1 = torch.stack(idx1 * 3, dim=0) # [3, fnum * seqlen]
                idx1 = idx1.squeeze(1)
                if len(b_ids) == 2:
                    idx2 = torch.stack(idx2 * 3, dim=0) # [3, fnum * seqlen]
                    idx2 = idx2.squeeze(1)

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            if self.pivotal_pass: # pre pass
                self.attn_output = self.attn1(norm_hid_states.view(batch_size, seqlen, dim), encoder_hidden_states if self.only_cross_attention else None, 
                                              **cross_attention_kwargs) # [3 * fnum, seqlen, dim]
                if self.batch_pivots == True: # EXTERNAL OFFLOAD
                    self.attn_output = self.attn_output.view(3, batch_size//3, seqlen, dim) # reshape here for proper pivotal batching
                    self.attn_outs += [self.attn_output] # [3, fnum, seqlen, dim]
                else:
                    self.kf = self.attn_output # required to keep fnum dimension [attn_output itself drops it to 1 on the second lat batch]
            else: # gen pass
                if self.batch_pivots == True: # EXTERNAL OFFLOAD
                    self.attn_output = self.attn_outs[self.layer_idx[0]]
                    self.layer_idx[0] = (self.layer_idx[0] + 1) % len(self.attn_outs) # NEXT LAYER
                else:
                    batch_size_ = self.kf.shape[0]
                    self.attn_output = self.kf.view(3, batch_size_//3, seqlen, dim) # reshape here with updated batch from the prev layer
                self.attn_output = self.attn_output[:,b_ids] # [3, len(b_ids), seqlen, dim]
            if self.use_ada_layer_norm_zero: # not
                self.attn_output = gate_msa.unsqueeze(1) * self.attn_output

            # gather values from attn_output, using idx as indices, and get a tensor of shape [3, fnum, seqlen, dim]
            if not self.pivotal_pass: # gen pass
                if len(b_ids) == 2:
                    attn_1, attn_2 = self.attn_output[:, 0], self.attn_output[:, 1]
                    attn_output1 = attn_1.gather(dim=1, index=idx1.unsqueeze(-1).repeat(1, 1, dim))
                    attn_output2 = attn_2.gather(dim=1, index=idx2.unsqueeze(-1).repeat(1, 1, dim))
                    s = torch.arange(0, fnum).to(idx1.device) + b_ids[0] * fnum
                    # distance from the pivot
                    p1 = b_ids[0] * fnum + fnum // 2
                    p2 = b_ids[1] * fnum + fnum // 2
                    d1 = torch.abs(s - p1)
                    d2 = torch.abs(s - p2)
                    # weight
                    w1 = d2 / (d1 + d2)
                    w1 = torch.sigmoid(w1)
                    
                    w1 = w1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(3, 1, seqlen, dim)
                    attn_output1 = attn_output1.view(3, fnum, seqlen, dim)
                    attn_output2 = attn_output2.view(3, fnum, seqlen, dim)
                    attn_output = w1 * attn_output1 + (1 - w1) * attn_output2 # [3, fnum, seqlen, dim]
                else:
                    attn_output = self.attn_output[:,0].gather(dim=1, index=idx1.unsqueeze(-1).repeat(1, 1, dim)) # [3, fnum * seqlen, dim]
                attn_output = attn_output.reshape(batch_size, seqlen, dim)  # 3*fnum, seqlen, dim
            else: # pre pass
                attn_output = self.attn_output.reshape(batch_size, seqlen, dim) # reshape required for pivot batch
            hidden_states = hidden_states.reshape(batch_size, seqlen, dim)  # 3*fnum, seqlen, dim
            hidden_states = attn_output + hidden_states
            hidden_states = hidden_states.half() # !!! hack fix

            if self.attn2 is not None:
                norm_hid_states = self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)

                # 2. Cross-Attention
                attn_output = self.attn2(norm_hid_states, encoder_hidden_states, attention_mask=encoder_attention_mask, **cross_attention_kwargs)
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hid_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hid_states = norm_hid_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            ff_output = self.ff(norm_hid_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states

            return hidden_states

    return TokenFlowBlock


def set_tokenflow(model: torch.nn.Module):
    """ Sets the tokenflow attention blocks in a model """
    for _, module in model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"): # name may change!?
            make_tokenflow_block_fn = make_tokenflow_attention_block 
            module.__class__ = make_tokenflow_block_fn(module.__class__)
            # for older versions of diffusers
            if not hasattr(module, "use_ada_layer_norm_zero"):
                module.use_ada_layer_norm = False
                module.use_ada_layer_norm_zero = False
    return model
