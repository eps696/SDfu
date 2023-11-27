# Stable Diffusers for studies

<p align='center'><img src='_in/something.jpg' /></p>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eps696/SDfu/blob/master/SDfu_colab.ipynb)

This is yet another Stable Diffusion toolkit, aimed to be functional, clean & compact enough for various experiments. There's no GUI here, as the target audience are creative coders rather than post-Photoshop users. The latter may check [InvokeAI] or [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) as convenient production suites, or [ComfyUI] for node-based workflows.  

The toolkit is built on top of the [diffusers] library, with occasional additions from the others mentioned below. The following codebases are partially included here (to ensure compatibility and the ease of setup): [k-diffusion](https://github.com/crowsonkb/k-diffusion), [CLIPseg], [LPIPS](https://github.com/richzhang/PerceptualSimilarity).  
There was also a [similar repo](https://github.com/eps696/SD) (abandoned now), based on the [CompVis] and [Stability AI] libraries.  

Current functions:
* Text to image, with possible prompting **by reference images** via [IP adapter]
* Image edits (re- and in-painting)
* **Various interpolations** (between/upon images or text prompts, smoothed by [latent blending])
* Guidance with [ControlNet] (pose, depth, canny edges) and [Instruct pix2pix]
* **Smooth & stable video edit** with [TokenFlow]
* Text to video with **[AnimateDiff]** and [ZeroScope] models (smooth & unlimited, as in [ComfyUI])
* Ultra-fast generation with [LCM] model (not fully tested with all operations yet)

Fine-tuning with your images:
* Add subject (new token) with [textual inversion]
* Add subject (new token + Unet delta) with [custom diffusion]
* Add subject (Unet low rank delta) with [LoRA]

Other features:
* Memory efficient with `xformers` (hi res on 6gb VRAM GPU)
* **Multi guidance** technique for better interpolations
* Use of special models: inpainting, SD v2, SDXL, [Kandinsky]
* Masking with text via [CLIPseg]
* Weighted multi-prompts (with brackets or numerical weights)
* to be continued..  

## Setup

Install CUDA 11.8 if you're on Windows (seems not necessary on Linux with Conda).  
Setup the Conda environment:
```
conda create -n SD python=3.10 numpy pillow 
activate SD
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install xformers
```
NB: It's preferrable to install `xformers` library - to increase performance and to run SD in any resolution on the lower grade hardware (e.g. videocards with 6gb VRAM). However, it's not guaranteed to work with all the (quickly changing) versions of `pytorch`, hence it's separated from the rest of requirements. If you're on Windows, first ensure that you have Visual Studio 2019 installed. 

Run command below to download: Stable Diffusion [1.5](https://huggingface.co/CompVis/stable-diffusion), [1.5 Dreamlike Photoreal](https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0), [2-inpaint](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting), 
[2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1-base), [2.1-v](https://huggingface.co/stabilityai/stable-diffusion-2-1), [custom VAE](https://huggingface.co/stabilityai/sd-vae-ft-ema), [LCM], [ZeroScope], [AnimateDiff], [ControlNet], [instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix), [IP adapter] with CLIPVision, [CLIPseg] models (converted to `float16` for faster loading). Licensing info is available on their webpages.
```
python download.py
```

## Operations

Examples of usage:

* Generate an image from the text prompt:
```
python src/gen.py --in_txt "hello world" --size 1024-576
```
* Redraw directory of images:
```
python src/gen.py --in_img _in/pix -t "neon light glow" --strength 0.7
```
* Inpaint directory of images with inpainting model, turning humans into robots:
```
python src/gen.py -im _in/pix --mask "human, person" -t "steampunk robot" --model 2i
```
* Make a video (frame sequence), interpolating between the lines of the text file:
```
python src/latwalk.py -t yourfile.txt --size 1024-576
```
* Same, with drawing over a masked image:
```
python src/latwalk.py -t yourfile.txt -im _in/pix/alex-iby-G_Pk4D9rMLs.jpg --mask "human boy" --invert_mask -m 2i
```
* Same as above, with recursive pan/zoom motion (beware of possible imagery degradation on longer runs):
```
python src/recur.py -t yourfile.txt --fstep 5 --scale 0.01 -m 15drm
```
* Hallucinate a video, including your real images:
```
python src/latwalk.py -im _in/pix --cfg_scale 0 -f 1
```
Interpolations can be made smoother (and faster) by adding `--latblend X` option ([latent blending] technique, X in range 0~1). 
If needed, smooth the result further with [FILM](https://github.com/google-research/frame-interpolation).  
Models can be selected with `--model` option by either a shortcut (15, 15drm, 21, 21v, ..), a path on the [Hugging Face] website (e.g. `SG161222/Realistic_Vision_V2.0`, would be auto-downloaded for further use) or a local path to the downloaded file set (or `safetensors` file).  
Check other options and their shortcuts by running these scripts with `--help` option.  

There are also few Windows bat-files, slightly simplifying and automating the commands. 

### Prompts 

Text prompts may include brackets for weighting (like `(good) [bad] ((even better)) [[even worse]]`).  
More radical blending can be achieved with **multiguidance technique**, introduced here (interpolating predicted noise within diffusion denoising loop, instead of conditioning vectors). It can be used to draw images from complex prompts like `good prompt ~1 | also good prompt ~1 | bad prompt ~-0.5` with `--cguide` option, or for animations with `--lguide` option (further enhancing smoothness of [latent blending]). Note that it would slow down generation process.  

It's possible also to use **reference images** as prompts by providing the path with `--img_ref ..` option. If it's a directory, single generations or interpolations will pick the files one by one, while video generation will consume them all at once.  
*NB: WIP, for now only single generation and AnimateDiff supported.*


## Guide synthesis with [ControlNet] or [Instruct pix2pix]

* Generate an image from existing one, using its depth map as conditioning (extra guiding source):
```
python src/preproc.py -i _in/something.jpg --type depth -o _in/depth
python src/gen.py --control_mod depth --control_img _in/depth/something.jpg -im _in/something.jpg -t "neon glow steampunk" -f 1
```
One can replace `depth` in the commands above with `canny` (edges) or `pose` (if there are humans in the source).  
Option `-im ...` may be omitted to employ "pure" txt2img method, pushing the result further to the text prompt:
```
python src/preproc.py -i _in/something.jpg --type canny -o _in/canny
python src/gen.py --control_mod canny --control_img _in/canny/something.jpg -t "neon glow steampunk" --size 1024-512 --model 15drm
```
ControlNet options can be used for interpolations as well (fancy making videomapping over a building photo?):
```
python src/latwalk.py --control_mod canny --control_img _in/canny/something.jpg --control_scale 0.5 -t yourfile.txt --size 1024-512 --fstep 5
```
also with pan/zoom recursion:
```
python src/recur.py -cmod canny -cimg _in/canny/something.jpg -cts 0.5 -t yourfile.txt --size 1024-640 -fs 5 -is 12 --scale 0.02 -m 15drm
```

### More ways to edit images 

[Instruct pix2pix]:
```
python src/gen.py -im _in/pix --img_scale 2 -C 9 -t "turn human to puppet" --model 1p2p
```

[TokenFlow] (temporally stable!):
```
python src/tokenflow.py -im _in/yoursequence -t "rusty metallic sculpture" --batch_size 4 --batch_pivot --cpu
```
TokenFlow employs either `pnp` or `sde` method and can be used with various models & ControlNet options.  
*NB: this method handles all frames at once (that's why it's so stable). As such, it cannot consume long sequences by design. Pivots batching & CPU offloading (introduced in this repo) pushed the limits, yet didn't removed them. As an example, I managed to process only 300+ frames of 960x540 on a 3090 GPU in batches of 5 without OOM (or without going to the 10x slower shared RAM with new Nvidia drivers).*


## Text to Video

Generate video from a text prompt with [AnimateDiff] motion adapter (may combine it with any base SD model):
```
python src/anima.py -t "fiery dragon in a China shop" -m 15drm --frames 100 --loop
```
Process existing video:
```
python src/anima.py -t "rusty metallic sculpture" -iv yourvideo.mp4 -f 0.7 -m 15drm
```

Generate video from a text prompt with [ZeroScope] model (kinda obsolete):
```
python src/vid.py -t "fiery dragon in a China shop" --model vzs --frames 100 --loop
```
Process existing video:
```
python src/vid.py -t "combat in the dancehall" --in_vid yourvideo.mp4 --model vzs
```
NB: this model is limited to rather mundane stuff, don't expect any notable level of abstraction or fantasy here.



## Fine-tuning

* Train new token embedding for a specific subject (e.g. cat) with [textual inversion]:
```
python src/train.py --token mycat1 --term cat --data data/mycat1 -lr 0.001 --type text
```
* Finetune the model (namely, part of the Unet attention layers) with [LoRA]:
```
python src/train.py --data data/mycat1 -lr 0.0001 --type lora
```
* Train both token embedding & Unet attentions with [custom diffusion]:
```
python src/train.py --token mycat1 --term cat --data data/mycat1 --term_data data/cat --type custom
```
Add `--style` if you're training for a style rather than an object. Speed up [custom diffusion] with `--xformers` ([LoRA] takes care of it on its own); add `--low_mem` if you get OOM.   
Results of the trainings will be saved under `train` directory. 

Custom diffusion trains faster and can achieve impressive reproduction quality (including faces) with simple similar prompts, but it can lose the point on generation if the prompt is too complex or aside from the original category. To train it, you'll need both target reference images (`data/mycat1`) and more random images of similar subjects (`data/cat`). Apparently, you can generate the latter with SD itself.  
LoRA finetuning seems less precise while may affect wider spectrum of topics, and is a de-facto industry standard now.  
Textual inversion is more generic but stable. Also, its embeddings can be easily combined together on load.  

* Generate image with trained weights from [LoRA]:
```
python src/gen.py -t "cosmic beast cat" --load_lora mycat1-lora.pt
```
* Same with [custom diffusion]:
```
python src/gen.py -t "cosmic <mycat1> cat beast" --load_custom mycat1-custom.pt
```
* Same with [textual inversion] (you may provide a folder path to load few files at once):
```
python src/gen.py -t "cosmic <mycat1> cat beast" --load_token mycat1-text.pt
```
Note that you should add (mind the brackets) `<token> term ..` keywords to the prompt to activate learned subject with Text Inversion or Custom Diffusion. Put it in the beginning for learned objects, or at the end for styles. LoRA is not bound to such syntax.  

You can also run `python src/latwalk.py ...` with finetuned weights to make animations.


## Special model: LCM

One of the most impressive recent discoveries is a Latent Consistency Model ([LCM]) architecture. It replaces regular diffusion part by a more direct latent prediction with distilled model, and requires only 2~4 steps to run. Not tested for compatibility with the features above yet.  
Example of usage:
```
python src/gen.py -m lcm -t "hello world"
python src/gen.py -m lcm -im _in/pix -t "neon light glow" -f 0.4
```

## Special model: SDXL

SDXL is not integrated into SDfu core yet, for now it's a separate script, wrapping existing `diffusers` pipelines.  
Supported features: txt2img, img2img, inpaint, depth/canny controlnet, text interpolations, dual prompts (native).  
Unsupported (yet): latent blending, multi guidance, fine-tuning, weighted prompts.  
NB: The models (~8gb total) are auto-downloaded on the first use; you may download them yourself and set the path with `--models_dir ...` option.  
As an example, interpolate with ControlNet:
```
python src/sdxl.py -v -t yourfile.txt -cimg _in/something.jpg -cmod depth -cts 0.6 --size 1280-768 -fs 5
```

## Special model: Kandinsky 2.2

Another interesting model is [Kandinsky] 2.2, featuring txt2img, img2img, inpaint, depth-based controlnet and simple interpolations. Its architecture and pipelines differ from Stable Diffusion, so there's also a separate script for it, wrapping those pipelines. The options are similar to the above; run `python src/kand.py -h` to see unused ones. It also consumes only unweighted prompts (no brackets, etc).  
NB: The models (heavy!) are auto-downloaded on the first use; you may download them yourself and set the path with `--models_dir ...` option.  
As an example, interpolate with ControlNet:
```
python src/kand.py -v -t yourfile.txt -cimg _in/something.jpg -cts 0.6 --size 1280-720 -fs 5
```

## Credits

It's quite hard to mention all those who made the current revolution in visual creativity possible. Check the inline links above for some of the sources. 
Huge respect to the people behind [Stable Diffusion], [Hugging Face], and the whole open-source movement.

[Stable Diffusion]: <https://github.com/CompVis/stable-diffusion>
[diffusers]: <https://github.com/huggingface/diffusers>
[Hugging Face]: <https://huggingface.co>
[CompVis]: <https://github.com/CompVis/stable-diffusion>
[Stability AI]: <https://github.com/Stability-AI/stablediffusion>
[InvokeAI]: <https://github.com/invoke-ai/InvokeAI>
[Deforum]: <https://github.com/deforum-art/deforum-stable-diffusion>
[CLIPseg]: <https://github.com/timojl/clipseg>
[ControlNet]: <https://github.com/lllyasviel/ControlNet>
[Instruct pix2pix]: <https://github.com/timothybrooks/instruct-pix2pix>
[TokenFlow]: <https://github.com/omerbt/TokenFlow>
[textual inversion]: <https://textual-inversion.github.io>
[custom diffusion]: <https://github.com/adobe-research/custom-diffusion>
[LoRA]: <https://github.com/cloneofsimo/lora>
[latent blending]: <https://github.com/lunarring/latentblending>
[LCM]: <https://latent-consistency-models.github.io>
[Kandinsky]: <https://huggingface.co/kandinsky-community>
[AnimateDiff]: <https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-2>
[ZeroScope]: <https://huggingface.co/cerspense/zeroscope_v2_576w>
[Potat]: <https://huggingface.co/camenduru/potat1>
[ComfyUI]: <https://github.com/comfyanonymous/ComfyUI>
[IP adapter]: <https://huggingface.co/h94/IP-Adapter>
