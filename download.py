import os
from tqdm import tqdm
import urllib.request
import zipfile
import shutil

def get_model(url: str, root: str = "./models", unzip=True):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url.split('?')[0])
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

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
    print(' downloading complete SD 1.5 model')
    get_model("https://www.dropbox.com/s/9wuhum8w0iqs4o7/sdfu-v15-full-fp16.zip?dl=1", 'models/v1')
    print(' downloading SD Dreamlike Photoreal Unet & text encoder')
    get_model("https://www.dropbox.com/s/7hwh27xfzcy7a2g/sdfu-v15drm-unetx-fp16.zip?dl=1", 'models/v1')
    # print(' downloading SD 1.5-inpainting Unet')
    # get_model("https://www.dropbox.com/s/bzmjnslk2sgsbt9/sdfu-v15i-unet-fp16.zip?dl=1", 'models/v1')

    print(' downloading complete SD 2.1 model')
    get_model("https://www.dropbox.com/s/38876tjuklvwq82/sdfu-v21-full-fp16.zip?dl=1", 'models/v2')
    print(' downloading complete SD 2.1-v model')
    get_model("https://www.dropbox.com/s/10gbecrugca1ydv/sdfu-v21v-full-fp16.zip?dl=1", 'models/v2v')
    print(' downloading SD 2-inpainting model Unet')
    get_model("https://www.dropbox.com/s/r5fa1mdxpw9e8k2/sdfu-v2i-unet-fp16.zip?dl=1", 'models/v2')
    print(' downloading SD 2-depth model Unet & depth')
    get_model("https://www.dropbox.com/s/4visx4qcscirvob/sdfu-v2d-unet-fp16.zip?dl=1", 'models/v2')

    print(' downloading SD 1.x extra VAE models')
    get_model("https://www.dropbox.com/s/z9uycihl6tybx9y/sdfu-v1-vaes-fp16.zip?dl=1", 'models/v1')

    print(' downloading ControlNet models')
    get_model("https://www.dropbox.com/s/qhe1zpbubjr3t75/sdfu-controlnet.zip?dl=1", 'models')

    print(' downloading CLIPseg model')
    get_model("https://www.dropbox.com/s/c0tduhr4g0al1cq/rd64-uni.pth?dl=1", 'models/clipseg', unzip=False)


if __name__ == '__main__':
    main()
