import argparse

samplers = ['klms', 'uni', 'pndm', 'dpm', 'euler_a', 'dpm2_a',   'ddim', 'euler']
models = ['15', '15drm', '15i', '2i', '21', '21v'] # !! only 15 is uncensored !!
unprompt = ""
# unprompt = "low quality, poorly drawn, out of focus, blurry, tiled, segmented" # , oversaturated, [[[ugly]]]
# unprompt += ", letters, text, titles, graffiti, typography, watermarks, writings"
# unprompt += ", human, people, man, girl, face"
# unprompt += ", ugly, deformed, disfigured, mutated, mutilated, bad anatomy, malformed hands, extra limbs"

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
    # mandatory params
    parser.add_argument('-m',  '--model',   default='15', choices=models, help="model version")
    parser.add_argument('-sm', '--sampler', default='ddim', choices=samplers)
    parser.add_argument(       '--vae',     default='ema', help='orig, ema, mse')
    parser.add_argument('-C','--cfg_scale', default=7.5, type=float, help="prompt guidance scale")
    parser.add_argument('-f', '--strength', default=0.75, type=float, help="strength of image processing. 0 = preserve img, 1 = replace it completely")
    parser.add_argument(      '--ddim_eta', default=0., type=float)
    parser.add_argument('-s',  '--steps',   default=50, type=int, help="number of diffusion steps")
    parser.add_argument('--precision',      default='autocast')
    parser.add_argument('-b',  '--batch',   default=1, type=int, help="batch size")
    parser.add_argument('-n',  '--num',     default=1, type=int, help="Repeat prompts N times")
    parser.add_argument('-S',  '--seed',    type=int, help="image seed")
    # finetuned stuff
    parser.add_argument('-rt', '--load_token', default=None, help="path to the text inversion embeddings file")
    parser.add_argument('-rd', '--load_custom', default=None, help="path to the custom diffusion delta checkpoint")
    parser.add_argument('-rl', '--load_lora', default=None, help="path to the LoRA file")
    # controlnet
    parser.add_argument('-cmod', '--control_mod', default=None, help="path to the ControlNet model")
    parser.add_argument('-cimg', '--control_img', default=None, help="path to the ControlNet driving image (contour, pose, etc)")
    parser.add_argument('-cts', '--control_scale', default=0.7, type=float, help="ControlNet effect scale")
    # misc
    parser.add_argument('-cg', '--cguide',  action='store_true', help='Use noise guidance for interpolation, instead of cond lerp')
    parser.add_argument('-sz', '--size',    default=None, help="image size, multiple of 8")
    parser.add_argument('-inv', '--invert_mask', action='store_true')
    parser.add_argument('-v',  '--verbose', action='store_true')

    return parser
