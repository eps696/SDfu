import os
from tqdm import tqdm
import urllib.request
import zipfile
import shutil

def get_model(url, root="./models", file='', name='', unzip=True):
    os.makedirs(root, exist_ok=True)
    download_target = os.path.join(root, os.path.basename(url.split('?')[0]))
    file_target     = os.path.join(root, file)

    if os.path.isfile(file_target):
        print(f" .. {file_target} exists .. skipping {name}")
        return
    elif os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    print(f" downloading {name}")
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=64, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))

    if unzip:
        with zipfile.ZipFile(download_target, 'r') as zf:
            zf.extractall(root)
        os.remove(download_target)

def main():
    get_model('https://www.dropbox.com/s/9wuhum8w0iqs4o7/sdfu-v15-full-fp16.zip?dl=1', 'models/v1', \
              'text/pytorch_model.bin', 'complete SD 1.5 model')
    get_model("https://www.dropbox.com/s/7hwh27xfzcy7a2g/sdfu-v15drm-unetx-fp16.zip?dl=1", 'models/v1', \
              'unet15drm/diffusion_pytorch_model.bin', 'SD Dreamlike Photoreal Unet & text encoder')
    # get_model("https://www.dropbox.com/s/bzmjnslk2sgsbt9/sdfu-v15i-unet-fp16.zip?dl=1", 'models/v1', \
              # 'unet15i/diffusion_pytorch_model.bin', 'SD 1.5-inpainting Unet')

    get_model("https://www.dropbox.com/s/38876tjuklvwq82/sdfu-v21-full-fp16.zip?dl=1", 'models/v2', \
              'unet21/diffusion_pytorch_model.bin', 'complete SD 2.1 model')
    get_model("https://www.dropbox.com/s/10gbecrugca1ydv/sdfu-v21v-full-fp16.zip?dl=1", 'models/v2v', \
              'unet21v/diffusion_pytorch_model.bin', 'complete SD 2.1-v model')
    get_model("https://www.dropbox.com/s/r5fa1mdxpw9e8k2/sdfu-v2i-unet-fp16.zip?dl=1", 'models/v2', \
              'unet2i/diffusion_pytorch_model.bin', 'SD 2-inpainting model Unet')
    # get_model("https://www.dropbox.com/s/4visx4qcscirvob/sdfu-v2d-unet-fp16.zip?dl=1", 'models/v2', \
              # 'unet2d/diffusion_pytorch_model.bin', 'SD 2-depth model Unet & depth')

    get_model("https://www.dropbox.com/s/ht7qbruw6mxz8le/sdfu-imageenc-fp16.zip?dl=1", 'models', \
              'image/ip-adapter_sd15.bin', 'IP adapter & CLIPVision models')

    get_model("https://www.dropbox.com/s/3sql6gsjmtvw2zo/sdfu-lcm-fp16.zip?dl=1", 'models', \
              'lcm/unet/diffusion_pytorch_model.safetensors', 'LCM few-steps model')

    get_model("https://www.dropbox.com/s/uyaidznqjaot7hw/sdfu-video-unet-fp16.zip?dl=1", 'models/v2', \
              'unetvzs/diffusion_pytorch_model.bin', 'Zeroscope model Unet')

    get_model("https://www.dropbox.com/scl/fi/4gorn5lf9owygizhgwuy6/sdfu-animatediff-fp16.zip?rlkey=mh54vl9ngre9n898pcgsr8fry&dl=1", 'models', \
              'anima/diffusion_pytorch_model.safetensors', 'AnimateDiff motion model')

    get_model("https://www.dropbox.com/s/qhe1zpbubjr3t75/sdfu-controlnet.zip?dl=1", 'models', \
              'control/depth/diffusion_pytorch_model.bin', 'ControlNet models')

    get_model("https://www.dropbox.com/s/n1z21ds5eauzk4m/sdfu-ip2p-unet-fp16.zip?dl=1", 'models/v1', \
              'unet1p2p/diffusion_pytorch_model.bin', 'Instruct pix2pix Unet')

    get_model("https://www.dropbox.com/s/z9uycihl6tybx9y/sdfu-v1-vaes-fp16.zip?dl=1", 'models/v1', \
              'vae-ft-ema/diffusion_pytorch_model.bin', 'SD 1.x extra VAE models')

    get_model("https://www.dropbox.com/scl/fi/blwzoyu9abp4q6wfq4hoj/rd64-uni.pth?rlkey=mhfp4jlles5oio1sn1eneyx4l&dl=1", 'models/xtra/clipseg', \
              'rd64-uni.pth', 'CLIPseg model', unzip=False)


if __name__ == '__main__':
    main()
