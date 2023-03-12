# Stable Diffusers for studies

<p align='center'><img src='_in/something.jpg' /></p>

This is yet another Stable Diffusion compilation, aimed to be functional, clean & compact enough for various experiments. There's no GUI here, as the target audience are creative coders rather than post-Photoshop users. The latter may check [InvokeAI] or [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) as a convenient production tool, or [Deforum] for precisely controlled animations.  

The code is based on the [diffusers] library, with occasional additions from the others mentioned below. The following codebases are partially included here (to ensure compatibility and the ease of setup): [k-diffusion](https://github.com/crowsonkb/k-diffusion), [CLIPseg].  
There is also a [similar repo](https://github.com/eps696/SD), based on the [CompVis] and [Stability AI] libraries (may be less up-to-date).  

Current functions:
* Text to image
* Image re- and in-painting
* Various interpolations (between or upon images or text prompts)

Fine-tuning with your images:
* Add subject (new token) with [textual inversion]
* Add subject (new token + Unet delta) with [custom diffusion]

Other features:
* Memory efficient with `xformers` (hi res on 6gb VRAM GPU)
* Use of special depth/inpainting and v2 models
* Masking with text via [CLIPseg]
* Weighted multi-prompts (with brackets or numerical weights)
* to be continued..  

More stuff and Colab version will follow. 

## Setup

Install CUDA 11.6. Setup the Conda environment:
```
conda create -n SD python=3.10 numpy pillow 
activate SD
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```
Install `xformers` library to increase performance. It makes possible to run SD in any resolution on the lower grade hardware (e.g. videocards with 6gb VRAM). If you're on Windows, first ensure that you have Visual Studio 2019 installed. 
```
pip install git+https://github.com/facebookresearch/xformers.git
```
Run command below to download Stable Diffusion [1.5](https://huggingface.co/CompVis/stable-diffusion), [1.5-inpaint](https://huggingface.co/runwayml/stable-diffusion-inpainting), [2-inpaint](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting), [2-depth](https://huggingface.co/stabilityai/stable-diffusion-2-depth), [2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1-base), [2.1-v](https://huggingface.co/stabilityai/stable-diffusion-2-1), [custom VAE](https://huggingface.co/stabilityai/sd-vae-ft-ema), [CLIPseg] models (converted to `float16` for faster loading). Licensing info is available on their webpages.
```
python download.py
```

## Operations

Examples of usage:

* Generate an image from the text prompt:
```
python src/gen.py -t "hello world" --size 1024-576
```
* Redraw directory of images, keeping the basic forms intact:
```
python src/gen.py -im _in/pix -t "neon light glow" --model 2d
```
* Inpaint directory of images with RunwayML model, turning humans into robots:
```
python src/gen.py -im _in/pix --mask "human, person" -t "steampunk robot" --model 15i
```
* Make a video (frame sequence), interpolating between the lines of the text file:
```
python src/latwalk.py -t yourfile.txt --size 1024-576
```
* Same, with drawing over a masked image:
```
python src/latwalk.py -t yourfile.txt -im _in/pix/alex-iby-G_Pk4D9rMLs.jpg --mask "human boy" --invert_mask -m 2i
```
* Hallucinate a video, including your real images:
```
python src/latwalk.py -im _in/pix --cfg_scale 0 --strength 1
```
Check other options and their shortcuts by running these scripts with `--help` option.  
Interpolated videos may be further smoothed out with [FILM](https://github.com/google-research/frame-interpolation).  

There are also Windows bat-files, slightly simplifying and automating the commands. 

## Fine-tuning

* Train new token embedding for a specific subject (e.g. cat) with [textual inversion]:
```
python src/train.py --token mycat1 --term cat --data data/mycat1 -lr 0.001
```
* Do the same with [custom diffusion]:
```
python src/train.py --token mycat1 --term cat --data data/mycat1 --term_data data/cat --low_mem
```
Results of the trainings will be saved under `train` directory. 

Custom diffusion trains faster and can achieve impressive reproduction quality (including faces) with simple similar prompts, but it can entirely lose the point if the prompt is complex or aside from the original category. To train it, you'll need both target reference images (`data/mycat1`) and more random images of similar subjects (`data/cat`). Apparently, you can generate the latter with SD itself.  
Textual inversion is more generic but stable. Also, its embeddings can be easily combined without additional retraining.  

* Generate image with embedding and delta weights from [custom diffusion]. The token is hardcoded in the file, so you need only to point the path to it:
```
python src/gen.py -t "cosmic <mycat1> beast" --delta_ckpt mycat1.ckpt
```
* Same with [textual inversion] (you may provide a folder path to load few files at once):
```
python src/gen.py -t "cosmic <mycat1> beast" --token_emb mycat1.pt
```
Besides special tokens (e.g. `<depthmap>`), text prompts may include weights, controlled as numbers (like `good prompt :1 | also good prompt :1 | bad prompt :-0.5`) or with brackets (like `(good) [bad] ((even better)) [[even worse]]`).  

You can also run `python src/latwalk.py ...` with such arguments to make animations.


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
[textual inversion]: <https://textual-inversion.github.io>
[custom diffusion]: <https://github.com/adobe-research/custom-diffusion>
