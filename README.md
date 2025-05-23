# Stable Diffusers for studies

<p align='center'><img src='_in/something.jpg' /></p>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eps696/SDfu/blob/master/SDfu_colab.ipynb)

This is yet another Stable Diffusion toolkit, aimed to be functional, clean & compact enough for various experiments. There's no GUI here, as the target audience are creative coders rather than post-Photoshop users. The latter may check [InvokeAI] or [Fooocus] as convenient production suites, or [ComfyUI] for flexible node-based workflows.  

The toolkit is built on top of the [diffusers] library, with occasional additions from the others mentioned below. The following codebases are partially included here (to ensure compatibility and the ease of setup): [Insightface](https://github.com/deepinsight/insightface), [CLIPseg], [LPIPS](https://github.com/richzhang/PerceptualSimilarity).  

Current functions:
* Text to image, with possible prompting **by reference images** via [IP adapters]
* Image edits (re- and in-painting)
* **Various interpolations** (between/upon images or text prompts, smoothed by [latent blending])
* Guidance with [ControlNet] (depth, depth-anything, pose, canny edges)
* **Video generation** with [CogVideoX], [LTXV], [AnimateDiff] and [ZeroScope] models (virtually unlimited for the latter two)
* Smooth & stable video edit with [TokenFlow]
* **Ultra-fast generation** with [TCD Scheduler] or [SDXL-Lightning] model (combined with other features)
* Ultra-fast generation with [LCM] model (not fully tested with all operations yet)

Fine-tuning with your images:
* Add subject (new token) with [Textual Inversion]
	* Same with **inventing novel imagery** with [ConceptLab]
* Add subject (new token + Unet delta) with [Custom Diffusion]
* Add subject (Unet low rank delta) with [LoRA]

Other features:
* Memory efficient with `xformers` (hi res on 6gb VRAM GPU)
* **Multi guidance** technique for better interpolations
* [Self-attention guidance] for better coherence and details
* Use of special models: inpainting, SD v2, SDXL, [Kandinsky]
* Masking with text via [CLIPseg]
* Weighted multi-prompts (with brackets or numerical weights)
* to be continued..  

## Setup

Example of setup on Windows:  
First install Visual Studio, CUDA 12.1, and Miniconda. Then run (in Conda prompt, from the repo directory):
```
conda create -n SD python=3.11
activate SD
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.27.post2 optimum-quanto==0.2.4
pip install -r requirements.txt
```

Example of similar setup on Mac / MPS (without CUDA):  
```
conda create -n SD python=3.11
activate SD
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
pip install -r requirements.txt
```

Run command below to download: Stable Diffusion [1.5](https://huggingface.co/CompVis/stable-diffusion), [1.5 Dreamlike Photoreal](https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0), [2-inpaint](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting), 
[custom VAE](https://huggingface.co/stabilityai/sd-vae-ft-ema), [LCM], [ZeroScope], [AnimateDiff] v3, [ControlNet], [IP adapters] with CLIPVision, [CLIPseg] models (converted to `float16` for faster loading).  
Please pay attention to the licensing info for these and other referenced models which is available on their webpages.
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
python src/recur.py -t yourfile.txt --fstep 5 --scale 0.01
```
* Hallucinate a video, including your real images:
```
python src/latwalk.py -im _in/pix --cfg_scale 0 -f 1
```
Interpolations can be made smoother (and faster) by adding `--latblend X` option ([latent blending] technique, X in range 0~1). 
If needed, smooth the result further with [FILM](https://github.com/google-research/frame-interpolation).  
Models can be selected with `--model` option by either a shortcut (15, 15drm, 21, ..), a path on the [Hugging Face] website (e.g. `SG161222/Realistic_Vision_V2.0`, would be auto-downloaded for further use) or a local path to the downloaded file set (or `safetensors` file).  
Coherence in details may be enhanced by [Self-Attention Guidance] with argument `--sag_scale X` (~1.5x slower, best with `ddpm` sampler). It works with per-frame generation and [AnimateDiff], but not for latent blending (yet).  
Check other options and their shortcuts by running these scripts with `--help` option.  

One of the most impressive recent advances is **ultrafast** Consistency generation approach (used in [LCM], [TCD Scheduler] and [SDXL-Lightning] techniques). It replaces regular diffusion part by the more direct latent prediction with distilled model, and requires very few (4 or more) steps to run. 
To use [TCD Scheduler] with any SD 1.5 base model, add `--sampler tcd --load_lora h1t/TCD-SD15-LoRA --cfg_scale 1 -s X` options where X is low (starting from 4). 
The quality seems to be sensitive to the prompt elaboration.

There are also few Windows bat-files, slightly simplifying and automating the commands. 

### Prompts 

Text prompts may include brackets for weighting (like `(good) [bad] ((even better)) [[even worse]]`).  
More radical blending can be achieved with **multiguidance technique**, introduced here (interpolating predicted noise within diffusion denoising loop, instead of conditioning vectors). It can be used to draw images from complex prompts like `good prompt ~1 | also good prompt ~1 | bad prompt ~-0.5` with `--cguide` option, or for animations with `--lguide` option (further enhancing smoothness of [latent blending]). Note that it would slow down generation process.  

It's possible also to use **reference images** as visual prompts with [IP adapters] technique by providing the path with `--img_ref ..` option. For a single reference, you can use either a single image, or any file set with `--allref` option. For an ordered scenario, you should provide a directory with image files or subdirectories (with images) to pick them one by one. The latter is preferrable, as the referencing quality is better when using 3-5 images than a single one.
For instance, this would make a smooth interpolation over a directory of images as visual prompts:  
```
python src/latwalk.py --img_ref _in/pix --latblend 0.8 --size 1024-576
```
One can select and/or combine various IP adapters for finer results (joining related parameters with `+`).  
Possible adapters: `plus`, `face-full`, `faceid-plus` or the file path.  
Possible types: `face` for face-based adapters; `full`, `scene` or `style` for the others.
```
python src/gen.py --img_ref _in/pix --ipa plus+faceid-plus --ip_type full+face
```
NB: For the time of writing, none of the face-based adapters can guarantee real portrait similarity. Try using [Custom Diffusion] finetuning (below) if you need a human-level face recognizability. Otherwise, just post-process results with third-party face-swapping methods.


## Guide synthesis with [ControlNet]

* Generate an image from existing one, using its depth map as conditioning (control source for extra guiding):
```
python src/preproc.py -i _in/something.jpg --type depth -o _in/depth
python src/gen.py --control_mod depth --control_img _in/depth/something.jpg -im _in/something.jpg -t "neon glow steampunk" -f 1
```
Posssible control types/modes are: `depth` ([MiDaS](https://github.com/isl-org/MiDaS)), `canny` (edges), `pose` (if there are human figures in the source) or `deptha` ([Depth Anything 2](https://github.com/DepthAnything/Depth-Anything-V2), very precise method). For the latter, depth maps are saved/loaded as dual-band 8bit png files to keep the float16 precision.  

Option `-im ...` may be omitted to employ "pure" txt2img method, pushing the result closer to the text prompt:
```
python src/preproc.py -i _in/something.jpg --type deptha -o _in/deptha
python src/gen.py --control_mod deptha --control_img _in/deptha/something.jpg -t "neon glow steampunk" --size 1024-512
```
ControlNet options can be used for interpolations as well (fancy making videomapping over a building photo?):
```
python src/latwalk.py --control_mod canny --control_img _in/canny/something.jpg --control_scale 0.5 -t yourfile.txt --size 1024-512 --fstep 5
```
also with pan/zoom recursion:
```
python src/recur.py -cmod canny -cnimg _in/canny/something.jpg -cts 0.5 -t yourfile.txt --size 1024-640 -fs 5 -is 12 --scale 0.02
```
Finally, one can select and/or combine various Controlnets for finer results (joining related parameters with `+`):
```
python src/gen.py -cmod depth+deptha -cnimg _in/depth/something.jpg+_in/deptha/something.jpg -cts 0.3+0.2 -t "neon glow steampunk" -f 1
```

### Video editing with [TokenFlow]

```
python src/tokenflow.py -im _in/yoursequence -t "rusty metallic sculpture" --batch_size 4 --batch_pivot --cpu
```
TokenFlow employs either `pnp` or `sde` method and can be used with various models & ControlNet options.  
*NB: this method handles all frames at once (that's why it's so stable). As such, it cannot consume long sequences by design. Pivots batching & CPU offloading (introduced in this repo) pushed the limits, yet didn't removed them. As an example, I managed to process only 300+ frames of 960x540 on a 3090 GPU in batches of 5 without video-memory overflow.*


## Video Synthesis

Generate a video from the text prompt (the more it's detailed the better!) with **[LTXV]** model:
```
python src/ltxv.py --frames 121 -ST 1.5 -t "late afternoon light casting long shadows,a cyclist athlet pedaling down a scenic mountain track"
```
Animate an image (you may add text prompt as well, just ensure that it's very detailed and strongly relating to the image):
```
python src/ltxv.py -ST 1.5 -im _in/pix/bench2.jpg
```

Generate a video from the text prompt (make it as detailed as possible!) with **[CogVideoX]** model:
```
python src/cogx.py --frames 101 --loop -t "late afternoon light casting long shadows,a cyclist athlet pedaling down a scenic mountain track"
```
Redraw existing video:
```
python src/cogx.py -iv yourvideo.mp4 -f 0.8 -t "decaying metallic sculpture, rusted swirls of iron oxide, jagged edges worn smooth"
```
Continue existing video, using last 9 video frames (or 3 in VAE latent space) as overlap for further generation (experimental):
```
python src/cogx.py -iv yourvideo.mp4 --frames 101 --overlap 9 -t "chaotic battle of prehistoric creatures at birtdhay party in a scientific lab" 
```
Generate a video from a directory of images with 50 frames between keyframes (very experimental):
```
python src/cogx.py -im yourimagedir --fstep 50 -t "decaying metallic sculpture, rusted swirls of iron oxide, jagged edges worn smooth"
```
NB: Generation of longer sequences is an abnormal use of the current CogX model, and often leads to degraded quality and leaked stock watermark appearance (especially with image-to-video model), so use it with care. Option `--rot_emb` (recalculate positional embeddings for full length) may give better temporal consistency but deteriorated image quality. More reliable way to achieve necessary length may be to prolong previous video pieces in a few 49-frames steps. You can also try `--dyn_cfg` option for text-to-video generations (works better for shorter pieces).


Generate a video from a text prompt with **[AnimateDiff]** motion adapter (may combine it with any base SD model):
```
python src/anima.py -t "fiery dragon in a China shop" --frames 100 --loop
```
Process existing video:
```
python src/anima.py -t "rusty metallic sculpture" -iv yourvideo.mp4 -f 0.7
```
Generate a video interpolation over a text file (as text prompts) and a directory of images (as visual prompts):
```
python src/anima.py -t yourfile.txt -imr _in/pix --frames 200 
```

Generate a video from a text prompt with **[ZeroScope]** model (kinda obsolete):
```
python src/vid.py -t "fiery dragon in a China shop" --model vzs --frames 100 --loop
```
Process existing video:
```
python src/vid.py -t "combat in the dancehall" --in_vid yourvideo.mp4 --model vzs
```
NB: this model is limited to rather mundane stuff, don't expect any notable level of abstraction or fantasy here.


## Fine-tuning

* Train new token embedding for a specific subject (e.g. cat) with [Textual Inversion]:
```
python src/train.py --token mycat1 --term cat --data data/mycat1 -lr 0.001 --type text
```
* Finetune the model (namely, part of the Unet attention layers) with [LoRA]:
```
python src/train.py --data data/mycat1 -lr 0.0001 --type lora
```
* Train both token embedding & Unet attentions with [Custom Diffusion]:
```
python src/train.py --token mycat1 --term cat --data data/mycat1 --term_data data/cat --type custom
```
Add `--style` if you're training for a style rather than an object. Speed up [Custom Diffusion] with `--xformers` ([LoRA] takes care of it on its own); add `--low_mem` if you get OOM.   
Results of the trainings will be saved under `train` directory. 

Custom diffusion trains faster and can achieve impressive reproduction quality (including faces) with simple similar prompts, but it can lose the point on generation if the prompt is too complex or aside from the original category. To train it, you'll need both target reference images (`data/mycat1`) and more random images of similar subjects (`data/cat`). Apparently, you can generate the latter with SD itself.  
LoRA finetuning seems less precise while may affect wider spectrum of topics, and is a de-facto industry standard now.  
Textual inversion is more generic but stable. Also, its embeddings can be easily combined together on load.  

One can also train new token embedding for a novel unusual subject within a class, employing the trick from [ConceptLab] (see their webpage for details):
```
python src/trainew.py --token mypet --term pet
```


* Generate an image with trained weights from [LoRA]:
```
python src/gen.py -t "cosmic beast cat" --load_lora mycat1-lora.pt
```
* Same with [Custom Diffusion]:
```
python src/gen.py -t "cosmic <mycat1> cat beast" --load_custom mycat1-custom.pt
```
* Same with [Textual Inversion] (you may provide a folder path to load few files at once):
```
python src/gen.py -t "cosmic <mycat1> cat beast" --load_token mycat1-text.pt
```
Note that you should add (mind the brackets) `<token> term ..` keywords to the prompt to activate learned subject with Text Inversion or Custom Diffusion. Put it in the beginning for learned objects, or at the end for styles. LoRA is not bound to such syntax.  

You can also run `python src/latwalk.py ...` with finetuned weights to make animations.


## Special model: LCM

Ultrafast Latent Consistency Model ([LCM]) with only 2~4 steps to run; supported only for image generation (not for video!).  
Examples of usage:
```
python src/gen.py -m lcm -t "hello world"
python src/gen.py -m lcm -im _in/pix -t "neon light glow" -f 0.5
python src/gen.py -m lcm -cmod depth -cnimg _in/depth/something.jpg -im _in/something.jpg -t "neon glow steampunk" -f 1
python src/latwalk.py -m lcm -t yourfile.txt
python src/latwalk.py -m lcm -t yourfile.txt -lb 0.75 -s 8
```

## Special model: SDXL

SDXL is a high quality HD model which is mostly used these days.  
Supported features: txt2img, img2img, image references, depth/canny controlnet, text interpolations with latent blending, dual prompts (native).  
Unsupported (yet): video generation, multi guidance, fine-tuning, weighted prompts.  
NB: The models (~8gb total) are auto-downloaded on the first use; you may download them yourself and set the path with `--models_dir ...` option.  
As an example, interpolate with ControlNet and Latent Blending:
```
python src/sdxl.py -v -t yourfile.txt -cnimg _in/something.jpg -cmod depth -cts 0.6 --size 1280-768 -fs 5 -lb 0.75
```
Methods for **ultrafast** generation with only few steps:
* distilled model [SDXL-Lightning]. Use it with `--lightning -s X` option where X = 2, 4 or 8. 
Pro: best quality; contra: requires special model.
* [TCD Scheduler]. Use it with `--sampler TCD --load_lora h1t/TCD-SDXL-LoRA --cfg_scale 1 -s X` options where X is low (starting from 4). 
Pro: applicable to any SDXL model; contra: quality may be worse (sensitive to the prompts).

Generate a video with SDXL model and [AnimateDiff] motion adapter (beware: sensitive to complex prompts):
```
python src/sdxl.py -v -t "fiery dragon in a China shop" -ad guoyww/animatediff-motion-adapter-sdxl-beta -sm euler -s 23 --size 1024-576
```
Technically, AnimateDiff-XL supports fast [SDXL-Lightning] models and [TCD Scheduler], but the results are very poor. 


## Special model: Kandinsky 2.2

Another interesting model is [Kandinsky] 2.2, featuring txt2img, img2img, inpaint, depth-based controlnet and simple interpolations. Its architecture and pipelines differ from Stable Diffusion, so there's also a separate script for it, wrapping those pipelines. The options are similar to the above; run `python src/kand.py -h` to see unused ones. It also consumes only unweighted prompts (no brackets, etc).  
NB: The models (heavy!) are auto-downloaded on the first use; you may download them yourself and set the path with `--models_dir ...` option.  
As an example, interpolate with ControlNet:
```
python src/kand.py -v -t yourfile.txt -cnimg _in/something.jpg -cts 0.6 --size 1280-720 -fs 5
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
[Fooocus]: <https://github.com/lllyasviel/Fooocus>
[CLIPseg]: <https://github.com/timojl/clipseg>
[ControlNet]: <https://github.com/lllyasviel/ControlNet>
[TokenFlow]: <https://github.com/omerbt/TokenFlow>
[Textual Inversion]: <https://textual-inversion.github.io>
[Custom Diffusion]: <https://github.com/adobe-research/custom-diffusion>
[LoRA]: <https://github.com/cloneofsimo/lora>
[latent blending]: <https://github.com/lunarring/latentblending>
[LCM]: <https://latent-consistency-models.github.io>
[Kandinsky]: <https://huggingface.co/kandinsky-community>
[LTXV]: <https://huggingface.co/Lightricks/LTX-Video>
[CogVideoX]: <https://github.com/THUDM/CogVideo>
[AnimateDiff]: <https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-2>
[ZeroScope]: <https://huggingface.co/cerspense/zeroscope_v2_576w>
[ComfyUI]: <https://github.com/comfyanonymous/ComfyUI>
[IP adapters]: <https://huggingface.co/h94/IP-Adapter>
[SDXL-Lightning]: <https://huggingface.co/ByteDance/SDXL-Lightning>
[TCD Scheduler]: <https://mhh0318.github.io/tcd/>
[Self-Attention Guidance]: <https://github.com/KU-CVLAB/Self-Attention-Guidance>
[ConceptLab]: <https://kfirgoldberg.github.io/ConceptLab>
[Instruct pix2pix]: <https://github.com/timothybrooks/instruct-pix2pix>
[instruct-pix2pix]: <https://huggingface.co/timbrooks/instruct-pix2pix>
