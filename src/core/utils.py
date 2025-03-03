import os, sys
import numpy as np
from PIL import Image, ImageOps
import cv2
import re
import pickle
import collections
import math
import skimage
from einops import rearrange
import scipy
from scipy.interpolate import CubicSpline as CubSpline

import torch
import torch.nn.functional as F

from .txt2mask import Txt2Mask

is_mac = torch.backends.mps.is_available() and torch.backends.mps.is_built() # M1/M2 chip?
is_cuda = torch.cuda.is_available()
device = 'mps' if is_mac else 'cuda' if is_cuda else 'cpu'
dtype = torch.float16 if is_cuda or is_mac else torch.float32

try: # notebook/colab
    get_ipython().__class__.__name__
    maybe_colab = True
except: # normal console
    maybe_colab = False

def gpu_ram():
    return torch.cuda.get_device_properties(0).total_memory // (1024*1024*1023) if is_cuda else 0

def clean_vram():
    torch.cuda.empty_cache() if is_cuda else torch.mps.empty_cache() if is_mac and hasattr(torch.mps, 'empty_cache') else None

def isok(*itms): # all not None, len > 0
    ok = all([x is not None for x in itms])
    if ok: ok = ok and all([len(x) > 0 for x in itms if hasattr(x, '__len__')])
    return ok

def isset(a, *itms): # all exist, not None, not False, len > 0
    if not all([isinstance(itm, str) for itm in itms]):
        print('!! Wrong items:', *itms); return False
    oks = [True]
    for arg in itms:
        if not hasattr(a, arg) or getattr(a, arg) is None or getattr(a, arg) is False:
            oks += [False]
        elif hasattr(getattr(a, arg), '__len__'):
            oks += [True] if len(getattr(a, arg)) > 0 else [False]
        else:
            oks += [True]
    return all(oks)

def cvshow(img, name='t'):
    if not maybe_colab:
        if torch.is_tensor(img):
            img = torch.clip((img+1)*127.5, 0, 255).cpu().numpy().astype(np.uint8)
        cv2.imshow(name, img[:,:,::-1])
        cv2.waitKey(1)

def framestack(ins, frames, curve='linear', loop=True, rejoin=False):
    if len(ins) == 1:
        outs = torch.stack(list(ins) * frames)
    else:
        assert frames >= len(ins), "Frame count < input count"
        frames -= 1 # to include the last one
        if loop: ins = list(ins) + [ins[0]]
        outs = [ins[0]]
        steps = len(ins) - 1
        for i in range(steps):
            curstep_count = frames // steps + (1 if i < frames % steps else 0)
            for j in range(1, curstep_count):
                alpha = blend(j / curstep_count, curve)
                outs += [(1 - alpha) * ins[i] + alpha * ins[i+1]]
            outs.append(ins[i+1])
        outs = torch.stack(outs)
    if rejoin:
        outs = rearrange(outs, 'n b ... -> (b n) ...')
    return outs

def calc_size(size, quant=8, pad=False):
    if isinstance(size, str): size = [int(s) for s in size.split('-')]
    if len(size)==1: size = size * 2
    w0, h0 = size
    w, h = map(lambda x: x - x % quant, size)  # resize to integer multiple of 8
    if pad and list(size) != [w,h]:
        if w0 != w: w += quant
        if h0 != h: h += quant
    return w, h

def lerp(v0, v1, x):
    return (1.-x) * v0 + x * v1

def lerps(v0, v1, x): # for lists
    assert len(v0) == len(v1)
    return [(1.-x) * x0 + x * x1 for x0,x1 in zip(v0,v1)]

# https://github.com/itjustis/sdthings/blob/main/scripts/alivify.py - from @xsteenbrugge?
def slerp(v0, v1, x, DOT_THRESHOLD=0.9995):
    if x==0: return v0
    dot = torch.sum(v0 * v1 / (torch.norm(v0) * torch.norm(v1)))
    if torch.abs(dot) > DOT_THRESHOLD:
        v2 = lerp(v0, v1, x)
    else:
        theta_0 = torch.acos(dot)
        sin_theta_0 = torch.sin(theta_0)
        theta_t = theta_0 * x
        sin_theta_t = torch.sin(theta_t)
        s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1
    return v2

def slerp2(v0, v1, x):
    if x==0: return v0
    z1_norm = torch.norm(v0)
    z2_norm = torch.norm(v1)
    z2_normal = v1 * (z1_norm / z2_norm)
    vectors = []
    interp = v0 + (z2_normal - v0) * x
    interp_norm = torch.norm(interp)
    interpol_normal = interp * (z1_norm / interp_norm)
    return interpol_normal

def triblur(x, k=3, pow=1.0):
    # padding = (k-1) // 2 # minimum padding to conv
    padding = k # let's pad more to avoid border effects
    b,c,h,w = x.shape
    kernel = torch.linspace(-1,1,k+2)[1:-1].abs().neg().add(1).reshape(1,1,1,k).pow(pow).to(device=device, dtype=x.dtype)
    kernel = kernel / kernel.sum()
    x = x.reshape(b*c,1,h,w)
    x = F.pad(x, (padding,padding,padding,padding), mode='reflect')
    x = F.conv2d(x, kernel)
    x = F.conv2d(x, kernel.permute(0,1,3,2))
    p = (k+1)//2
    x = x[:, :, p:-p, p:-p] # crop extra padding
    x = x.reshape(b,c,h,w)
    return x

def mono16_to_dual8(x): # np array [h,w,c]
    x = (x * 65025.).astype(np.uint16)
    hi = (x >> 8).astype(np.uint8)  # High byte
    lo = (x & 0xFF).astype(np.uint8)  # Low byte
    zero = np.zeros_like(hi, dtype=np.uint8)
    return np.vstack((hi, lo, zero)).transpose(1,2,0)

def dual8_to_mono16(x): # np array [h,w,c]
    hi = x[:,:,0]
    lo = x[:,:,1]
    result = (hi.astype(np.uint16) << 8) | lo.astype(np.uint16)
    result = (result / 65025).astype(np.float32).clip(0,1)
    return result.reshape(x.shape[0], x.shape[1], 1).repeat(3, axis=2)

def load_img(path, size=None, tensor=True, quant=8, pad=False, gpu=True, dual8b=False):
    image = Image.open(path).convert('RGB')
    if isinstance(size, int): size = [size,size]
    if size is None: size = image.size
    w, h = calc_size(size, quant, pad)
    if [w,h] != list(image.size):
        if pad:
            result = Image.new(image.mode, (w,h))
            result.paste(image, (0,0))
            image = result
        else:
            resampl = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
            image = image.resize((w,h), resampl) # (w,h)
    if not tensor: return image, (w,h)
    if dual8b:
        image = dual8_to_mono16(np.array(image))
    else:
        image = np.array(image).astype(np.float32) / 255.
    image = torch.from_numpy(image).permute(2,0,1).unsqueeze(0)
    if gpu: image = image.to(device, dtype=dtype)
    return 2.*image - 1., (w,h)

def save_img(image, num, out_dir, prefix='', filepath=None, ext='jpg'):
    image = torch.clamp((image + 1.) / 2., min=0., max=1.)
    if image.device.type != 'cpu': image = image.detach().cpu()
    if image.dtype != torch.float32: image = image.float()
    image = image.permute(1, 2, 0).numpy() * 255
    if filepath is None: filepath = '%05d.%s' % (num, ext)
    Image.fromarray(image.astype(np.uint8)).save(os.path.join(out_dir, prefix + filepath), quality=95, subsampling=0)

def makemask(mask_str, image=None, invert_mask=False, threshold=0.35, tensor=True, model_path='models/xtra/clipseg/rd64-uni.pth'):
    if os.path.isfile(mask_str): 
        mask, _ = load_img(mask_str, tensor=False)
        mask = ImageOps.invert(mask.convert('L'))
    else: 
        if isinstance(image, str):
            image, _ = load_img(image, tensor=False)
        txt2mask = Txt2Mask(model_path, device=device)
        mask = txt2mask.segment(image, mask_str).to_mask(float(threshold))
        # image.putalpha(mask)
    if invert_mask: mask = ImageOps.invert(mask)
    if not tensor: return mask
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device, dtype=dtype)
    return mask

def unique_prefix(out_dir):
    dirlist = sorted(os.listdir(out_dir), reverse=True) # sort reverse alphabetically until we find max+1
    existing_name = next((f for f in dirlist if re.match('^(\d+)\..*\.jpg', f)), '00000.0.jpg')
    basecount = int(existing_name.split('.', 1)[0]) + 1
    return f'{basecount:05}'

def correct_colors(image_list, reference_image_path, image_callback = None): 
    reference_image = Image.open(reference_image_path)
    correction_target = cv2.cvtColor(np.asarray(reference_image), cv2.COLOR_RGB2LAB)
    for r in image_list:
        image, seed = r
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2LAB)
        image = skimage.exposure.match_histograms(image, correction_target, channel_axis=2)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_LAB2RGB).astype("uint8"))
        if image_callback is not None:
            image_callback(image, seed)
        else:
            r[0] = image

def basename(file):
    return os.path.splitext(os.path.basename(file))[0]

def file_list(path, ext=None):
    files = [os.path.join(path, f) for f in os.listdir(path)]
    if ext is not None: 
        files = [f for f in files if f.endswith(ext)]
    return sorted([f for f in files if os.path.isfile(f)])

def img_list(path, subdir=None):
    if subdir is True:
        files = [os.path.join(dp, f) for dp, dn, fn in os.walk(path) for f in fn]
    else:
        files = [os.path.join(path, f) for f in os.listdir(path)]
    files = [f for f in files if os.path.splitext(f.lower())[1][1:] in ['jpg', 'jpeg', 'png', 'ppm', 'tif']]
    files = [f for f in files if not '/__MACOSX/' in f.replace('\\', '/')] # workaround fix for macos phantom files
    return sorted([f for f in files if os.path.isfile(f)])

def blend(t, ip):
    if ip == 'bezier':
        return BezierBlend(t)
    elif ip == 'parametric':
        return ParametricBlend(t)
    elif ip == 'custom':
        return CustomBlend(t)
    else:
        return t

def BezierBlend(t):
  return t * t * (3.-2.*t)
def ParametricBlend(t):
    sqt = t*t
    return (sqt / (2.0 * (sqt - t) + 1.0))
def CustomBlend(x):
    return 1 - 2*x*(1-x) if x >= 0.5 else 2*x*(1-x)

def log_tokens(text, model):
    tokenized = ""
    discarded = ""
    usedTokens = 0
    tokens    = model.cond_stage_model.tokenizer._tokenize(text)
    totalTokens = len(tokens)
    for i in range(0, totalTokens):
        token = tokens[i].replace('</w>', '')
        if i < model.cond_stage_model.max_length:
            tokenized = tokenized + f"{token} "
            usedTokens += 1
        else:  # over max token length
            discarded = discarded + f"{token}"
    print(f".. {usedTokens} tokens: {tokenized}")
    if discarded != "": 
        print(f".. {totalTokens-usedTokens} discarded: {discarded}")

def read_latents(latents):
    if latents is not None and os.path.isfile(latents):
        key_latents = load_latents(latents)
    elif latents is not None and os.path.isdir(latents):
        key_latents = []
        lat_list = file_list(latents, 'pkl')
        for lat in lat_list: 
            key_latent = load_latents(lat)
            key_latents.append(key_latent)
        if isinstance(key_latents[0], list) or isinstance(key_latents[0], tuple):
            key_latents = list(map(list, zip(*key_latents)))
            for n in range(len(key_latents)):
                key_latents[n] = torch.cat(key_latents[n])
        else:
            key_latents = torch.cat(key_latents)
    else:
        print(' No input latents found'); exit()
    return key_latents
    
def load_latents(lat_file):
    with open(lat_file, 'rb') as f:
        key_lats = pickle.load(f)
    idx_file = os.path.splitext(lat_file)[0] + '.txt'
    if os.path.exists(idx_file): 
        with open(idx_file) as f:
            lat_idx = f.readline()
            lat_idx = [int(l.strip()) for l in lat_idx.split(',') if len(l.strip())>0]
        key_lats = list(key_lats) if isinstance(key_lats, list) or isinstance(key_lats, tuple) else [key_lats]
        for n, key_lat in enumerate(key_lats):
            key_lats[n] = torch.cat([key_lat[i % len(key_lat)].unsqueeze(0) for i in lat_idx])
    return key_lats

def save_cfg(args, dir='./', file='config.txt'):
    if dir != '':
        os.makedirs(dir, exist_ok=True)
    try: args = vars(args)
    except: pass
    if file is None:
        print_dict(args)
    else:
        with open(os.path.join(dir, file), 'w', encoding="utf-8") as cfg_file: # utf-8-sig maybe
            print_dict(args, cfg_file)

def print_dict(dict, file=None, path="", indent=''):
    for k in sorted(dict.keys()):
        if isinstance(dict[k], collections.abc.Mapping):
            if file is None:
                print(indent + str(k))
            else:
                file.write(indent + str(k) + ' \n')
            path = k if path=="" else path + "->" + k
            print_dict(dict[k], file, path, indent + '   ')
        else:
            if file is None:
                print('%s%s: %s' % (indent, str(k), str(dict[k])))
            else:
                file.write('%s%s: %s \n' % (indent, str(k), str(dict[k])))

# = = = = = = = = = = = = = = = = = = = = = = = = = = = 

def get_z(shape, rnd, uniform=False):
    if uniform:
        return rnd.uniform(0., 1., shape)
    else:
        return rnd.randn(*shape) # *x unpacks tuple/list to sequence

def make_lerps(z1, z2, num_steps): 
    vectors = []
    xs = [step / (num_steps - 1) for step in range(num_steps)]
    for x in xs:
        interpol = z1 + (z2 - z1) * x
        vectors.append(interpol)
    return np.array(vectors)

# interpolate on hypersphere
def make_slerps(z1, z2, num_steps):
    z1_norm = np.linalg.norm(z1)
    z2_norm = np.linalg.norm(z2)
    z2_normal = z2 * (z1_norm / z2_norm)
    vectors = []
    xs = [step / (num_steps - 1) for step in range(num_steps)]
    for x in xs:
        interplain = z1 + (z2 - z1) * x
        interp = z1 + (z2_normal - z1) * x
        interp_norm = np.linalg.norm(interp)
        interpol_normal = interplain * (z1_norm / interp_norm)
        # interpol_normal = interp * (z1_norm / interp_norm)
        vectors.append(interpol_normal)
    return np.array(vectors)

def cublerp(points, steps, fstep, looped=True):
    keys = np.array([i*fstep for i in range(steps)] + [steps*fstep])
    last_pt_num = 0 if looped is True else -1
    points = np.concatenate((points, np.expand_dims(points[last_pt_num], 0)))
    cspline = CubSpline(keys, points)
    return cspline(range(steps*fstep+1))

def latent_anima(shape, frames, transit, key_latents=None, uniform=False, cubic=False, start_lat=None, seed=None, looped=True, verbose=False):
    if key_latents is None:
        transit = int(max(1, min(frames//2, transit)))
    steps = max(1, math.ceil(frames / transit))
    log = ' timeline: %d steps by %d' % (steps, transit)

    if seed is None:
        seed = np.random.seed(int((time.time()%1) * 9999))
    rnd = np.random.RandomState(seed)
    
    # make key points
    if key_latents is None:
        key_latents = np.array([get_z(shape, rnd, uniform) for i in range(steps)])
    if start_lat is not None:
        key_latents[0] = start_lat

    latents = np.expand_dims(key_latents[0], 0)
    
    # populate lerp between key points
    if transit == 1:
        latents = key_latents
    else:
        if cubic:
            latents = cublerp(key_latents, steps, transit, looped)
            log += ', cubic'
        else:
            for i in range(steps):
                zA = key_latents[i]
                lat_num = (i+1)%steps if looped is True else min(i+1, steps-1)
                zB = key_latents[lat_num]
                if uniform is True:
                    interps_z = make_lerps(zA, zB, transit)
                else:
                    interps_z = make_slerps(zA, zB, transit)
                latents = np.concatenate((latents, interps_z))
    latents = np.array(latents)
    
    if verbose: print(log)
    if latents.shape[0] > frames: # extra frame
        latents = latents[1:]
    return latents
    
# # # = = = progress bar = = = # # #

import time
from shutil import get_terminal_size
import ipywidgets as ipy
import IPython

class ProgressIPy(object):
    def __init__(self, task_num=10, start_num=0, start=True):
        self.task_num = task_num - start_num
        self.pbar = ipy.IntProgress(min=0, max=self.task_num, bar_style='') # (value=0, min=0, max=max, step=1, description=description, bar_style='')
        self.labl = ipy.Label()
        IPython.display.display(ipy.HBox([self.pbar, self.labl]))
        self.completed = 0
        self.start_num = start_num
        if start:
            self.start()

    def start(self, task_num=None, start_num=None):
        if task_num is not None:
            self.task_num = task_num
        if start_num is not None:
            self.start_num = start_num
        self.labl.value = '{}/{}'.format(self.start_num, self.task_num + self.start_num)
        self.start_time = time.time()

    def upd(self, *p, **kw):
        self.completed += 1
        elapsed = time.time() - self.start_time + 0.0000000000001
        fps = self.completed / elapsed if elapsed>0 else 0
        if self.task_num > 0:
            finaltime = time.asctime(time.localtime(self.start_time + self.task_num * elapsed / float(self.completed)))
            fin = ' end %s' % finaltime[11:16]
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            self.labl.value = '{}/{}, rate {:.3g}s, time {}s, left {}s, {}'.format(self.completed + self.start_num, self.task_num + self.start_num, 1./fps, shortime(elapsed), shortime(eta), fin)
        else:
            self.labl.value = 'completed {}, time {}s, {:.1f} steps/s'.format(self.completed + self.start_num, int(elapsed + 0.5), fps)
        self.pbar.value += 1
        if self.completed == self.task_num: self.pbar.bar_style = 'success'
        return self.completed

    def reset(self, start_num=0, task_num=None):
        self.start_time = time.time()
        self.start_num = start_num
        if task_num is not None:
            self.task_num = task_num

class ProgressBar(object):
    '''A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    '''
    def __init__(self, count=0, start_num=0, bar_width=50, start=True):
        self.task_num = count - start_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        self.start_num = start_num
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal is small ({}), make it bigger for proper visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self, task_num=None, start_num=None):
        if task_num is not None:
            self.task_num = task_num
        if start_num is not None:
            self.start_num = start_num
        if self.task_num > 0:
            sys.stdout.write('[{}] {}/{} \n{}\n'.format(' ' * self.bar_width, self.start_num, self.task_num + self.start_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def upd(self, msg=None, uprows=0):
        self.completed += 1
        elapsed = time.time() - self.start_time + 0.0000000000001
        fps = self.completed / elapsed if elapsed>0 else 0
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            finaltime = time.asctime(time.localtime(self.start_time + self.task_num * elapsed / float(self.completed)))
            fin_msg = ' %ss left, end %s' % (shortime(eta), finaltime[11:16])
            if msg is not None: fin_msg += '  ' + str(msg)
            mark_width = int(self.bar_width * percentage)
            bar_chars = 'X' * mark_width + '-' * (self.bar_width - mark_width) # - - -
            sys.stdout.write('\033[%dA' % (uprows+2)) # cursor up 2 lines + extra if needed
            sys.stdout.write('\033[J')  # clean the output (remove extra chars since last display)
            try:
                sys.stdout.write('[{}] {}/{}, rate {:.3g}s, time {}s, left {}s \n{}\n'.format(
                    bar_chars, self.completed + self.start_num, self.task_num + self.start_num, 1./fps, shortime(elapsed), shortime(eta), fin_msg))
            except:
                sys.stdout.write('[{}] {}/{}, rate {:.3g}s, time {}s, left {}s \n{}\n'.format(
                    bar_chars, self.completed + self.start_num, self.task_num + self.start_num, 1./fps, shortime(elapsed), shortime(eta), '<< unprintable >>'))
        else:
            sys.stdout.write('completed {}, time {}s, {:.1f} steps/s'.format(self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()

    def reset(self, start_num=0, count=None, newline=False):
        self.start_time = time.time()
        self.start_num = start_num
        if count is not None:
            self.task_num = count - start_num
        if newline is True:
            sys.stdout.write('\n\n')

progbar = ProgressIPy if maybe_colab else ProgressBar

def time_days(sec):
    return '%dd %d:%02d:%02d' % (sec/86400, (sec/3600)%24, (sec/60)%60, sec%60)
def time_hrs(sec):
    return '%d:%02d:%02d' % (sec/3600, (sec/60)%60, sec%60)
def shortime(sec):
    if sec < 60:
        time_short = '%d' % (sec)
    elif sec < 3600:
        time_short  = '%d:%02d' % ((sec/60)%60, sec%60)
    elif sec < 86400:
        time_short  = time_hrs(sec)
    else:
        time_short = time_days(sec)
    return time_short

