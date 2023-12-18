from tqdm import tqdm
import network
import utils
import os
import argparse

from datasets import VOCSegmentation, Cityscapes
from torchvision import transforms as T

import torch
import torch.nn as nn

from PIL import Image
from glob import glob

import numpy as np
from trimap_generator.trimap_class import trimap,Dilation,Erosion


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--input", type=str, required=True,
                        help="path to a single image or image directory")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of training set')

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet50',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--ckpt", default="weight/best_deeplabv3plus_resnet50_voc_os16.pth", type=str,
                        help="resume from checkpoint")
    
    ## Generate Trimap
    parser.add_argument("--size", type=int, default=15,
                        help="Erosion and Dilation Size") # Unknown Region Thickness
    parser.add_argument("--defg", type=object, default=Erosion,
                        help="None/Dilation/Erosion")
    parser.add_argument("--num_iter", type=object, default=10,
                        help="Dilation/Erosion num iters")
    
    return parser

def segment(opts):
    
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        decode_fn = Cityscapes.decode_target

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    image_files = []
    if os.path.isdir(opts.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input, '**/*.%s'%(ext)), recursive=True)
            if len(files)>0:
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)
    
    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)


    transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    
    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files):
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext)-1]
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0) # To tensor of NCHW
            img = img.to(device)
            
            pred = model(img).max(1)[1].cpu().numpy()[0] # HW
            colorized_preds = decode_fn(pred).astype('uint8')
            colorized_preds = Image.fromarray(colorized_preds)

    return colorized_preds



#======Temp ======
if __name__ == '__main__':
    opts = get_argparser().parse_args()

    # Segment
    img = segment(opts)
    img.save(f'./image/seg_img/{os.path.basename(opts.input)}')

    # Change to Mask (0 or 255)
    img_np = np.array(img)
    img_np[img_np.sum(axis=2) > 0] = 255
    gray_img_np = np.mean(img_np, axis=-1) #channel 3 to channel 2

    #Generate trimap
    trimap_img = trimap(opts,gray_img_np)
    img = Image.fromarray(trimap_img)
    img = img.convert('L')
    img.save(f'./image/trimap_img/{os.path.basename(opts.input)}')

    
