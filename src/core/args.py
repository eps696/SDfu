import argparse

samplers = ['ddim', 'pndm', 'lms', 'euler', 'euler_a', 'uni', 'dpm', 'ddpm',  'lcm', 'tcd', 'orig']
models = ['lcm', '15', '15drm', '2i', '21', 'vzs'] # lcm, 15, 15drm are uncensored
un = ""
un = "low quality, poorly drawn, out of focus, blurry, tiled, segmented, oversaturated"
# un += ", letters, text, titles, graffiti, typography, watermarks, writings"
# un += ", human, people, man, girl, face"
# un += ", ugly, deformed, disfigured, mutated, mutilated, bad anatomy, malformed hands, extra limbs"

def unprompt(args):
    una = args.unprompt
    return un if una is None else '' if una=='no' else una if una[-1]=='.' else un + una if una[0]==',' else ', '.join([una, un])

def main_args():
    parser = argparse.ArgumentParser(conflict_handler = 'resolve')
    # inputs & paths
    parser.add_argument('-t',  '--in_txt',  default='', help='Text string or file to process')
    parser.add_argument('-pre', '--pretxt', default='', help='Prefix for input text')
    parser.add_argument('-post','--postxt', default='', help='Postfix for input text')
    parser.add_argument('-im', '--in_img',  default=None, help='input image or directory with images (overrides width and height)')
    parser.add_argument('-M',  '--mask',    default=None, help='Path to input mask for inpainting mode (overrides width and height)')
    parser.add_argument('-un','--unprompt', default=None, help='Negative prompt to be used as a neutral [uncond] starting point')
    parser.add_argument('-o',  '--out_dir', default="_out", help="Output directory for generated images")
    parser.add_argument('-md', '--maindir', default='./models', help='Main SD models directory')
    # ip adapters
    parser.add_argument('-imr','--img_ref', default=None, help="Reference image[s] or directory[s] with images, separated by '+'")
    parser.add_argument('-imw', '--imgref_weight', default='0.3', help="Weight[s] for the reference image(s), separated by '+'")
    parser.add_argument('-ip', '--ipa',     default='', help="IP adapter model name[s], separated by '+'")
    parser.add_argument('-ipt','--ip_type', default='full', help="IP adapter type[s] = full, scene, style, face; separated by '+'")
    parser.add_argument('-ar',  '--allref', default='', help='y = apply all reference images at once, n = pick one by one')
    # mandatory params
    parser.add_argument('-m',  '--model',   default='15drm', help="SD model to use")
    parser.add_argument('-sm', '--sampler', default='ddim', choices=samplers)
    parser.add_argument(       '--vae',     default='ema', help='orig, ema, mse')
    parser.add_argument('-C','--cfg_scale', default=7.5, type=float, help="prompt guidance scale")
    parser.add_argument('-f', '--strength', default=1, type=float, help="strength of image processing. 0 = preserve img, 1 = replace it completely")
    parser.add_argument('-if', '--img_scale', default=None, type=float, help='image guidance scale for Instruct pix2pix. None = disabled it')
    parser.add_argument('-eta','--eta',     default=0., type=float)
    parser.add_argument('-s',  '--steps',   default=37, type=int, help="number of diffusion steps")
    parser.add_argument('-b',  '--batch',   default=1, type=int, help="batch size")
    parser.add_argument(   '--vae_batch',   default=8, type=int, help="batch size for VAE decoding")
    parser.add_argument('-n',  '--num',     default=1, type=int, help="Repeat prompts N times")
    parser.add_argument('-S',  '--seed',    type=int, help="image seed")
    # finetuned stuff
    parser.add_argument('-rt', '--load_token', default=None, help="path to the text inversion embeddings file")
    parser.add_argument('-rd', '--load_custom', default=None, help="path to the custom diffusion delta checkpoint")
    parser.add_argument('-rl', '--load_lora', default=None, help="path to the LoRA file")
    # controlnet
    parser.add_argument('-cmod', '--control_mod', default=None, help="Path[s] to the ControlNet models, separated by '+'")
    parser.add_argument('-cnimg','--control_img', default=None, help="Path[s] to the ControlNet driving images, separated by '+'")
    parser.add_argument('-cts', '--control_scale', default='0.7', help="ControlNet effect scale[s], separated by '+'")
    # misc
    parser.add_argument('-cg', '--cguide',  action='store_true', help='Use noise guidance for interpolation, instead of cond lerp')
    parser.add_argument('-fu',  '--freeu',  action='store_true', help='Use FreeU enhancement (Fourier representations in Unet)')
    parser.add_argument('-sag','--sag_scale', default=0, type=float, help="Self-attention guidance scale")
    parser.add_argument('-sz', '--size',    default=None, help="image size, multiple of 8")
    parser.add_argument('-lo', '--lowmem',  action='store_true', help='Offload subnets onto CPU for higher resolution [slower]')
    parser.add_argument('-inv', '--invert_mask', action='store_true')
    parser.add_argument('-v',  '--verbose', action='store_true')

    return parser
