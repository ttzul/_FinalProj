from dim.core.net import VGG16
import cv2
import torch 
import argparse
from dim.core.deploy import inference_img_whole
import numpy as np

# FROM dim.core.deploy 
def my_torch_load(fname):
    try:
        ckpt = torch.load(fname, encoding='ISO-8859-1')
        return ckpt
    except Exception as e:
        print("Load Error:{}/nTry Load Again...".format(e))
        class C:
            pass
        def c_load(ss):
            return pickle.load(ss, encoding='latin1')
        def c_unpickler(ss):
            return pickle.Unpickler(ss, encoding='latin1')
        c = C
        c.load = c_load
        c.Unpickler = c_unpickler
        ckpt = torch.load(args.resume, pickle_module=c, encoding='ISO-8859-1')
        return ckpt


def args_init():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.cuda = True
    args.stage = 1
    args.crop_or_resize = "whole"
    args.resume = "dim/model/stage1_skip_sad_52.9.pth"
    return args

def model_load(args):
    # load the model
    model = VGG16(args)
    ckpt = my_torch_load(args.resume)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model = model.cuda()
    return model

def matting(args, image, trimap):
    model = model_load(args)
    args.size_h = image.shape[0]
    args.size_w = image.shape[1]
    args.max_size = 1600

    with torch.no_grad():
        pred_mattes = inference_img_whole(args, model, image, trimap)
    
    return pred_mattes

def composing(fg, alpha_matte, bg=None):
    # extend the alpha_matte to 3 channels
    alpha_matte = np.expand_dims(alpha_matte, axis=2)
    if bg is None:
        # return an RGBA image
        composing_res = fg * alpha_matte
        composing_alpha = alpha_matte * 255
        composing_res = np.concatenate((composing_res, composing_alpha), axis=2)
    else:
        bg = cv2.resize(bg, (fg.shape[1], fg.shape[0]), interpolation=cv2.INTER_LINEAR)
        composing_res = fg * alpha_matte + bg * (1 - alpha_matte)

    return composing_res

        
    

if __name__ == "__main__":
    image_fg = cv2.imread("dim/result/example/image/troll.png", cv2.IMREAD_COLOR)
    trimap = cv2.imread("dim/result/example/trimap/troll.png", cv2.IMREAD_GRAYSCALE)
    image_bg = cv2.imread("dim/result/example/image/dandelion-1335575_1920_1.png")
    args = args_init()
    alpha_matte = matting(args, image_fg, trimap)
    res = composing(image_fg, alpha_matte)
    cv2.imwrite("dim/result/example/pred/troll.png", res)