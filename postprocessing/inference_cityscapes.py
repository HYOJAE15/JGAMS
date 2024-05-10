import argparse
import os.path as osp

import cv2
import numpy as np

from glob import glob
from tqdm import tqdm

from mmseg.apis import init_model, inference_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='inference with cityscapes format code'
    )
    # parser.add_argument('config', help='config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        'img_dir',
        help=('image directory for inference'))
    parser.add_argument(
        'save_dir',
        help=('directory to save result'))
    # parser.add_argument(
    #     'dataset_type',
    #     help=('data type to be inference'))
    parser.add_argument(
        'img_format',
        help=('format of img'))


    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    
    # model = init_model(args.config, args.checkpoint, device='cuda:0')

    img_list = glob(osp.join(args.img_dir, f'*.{args.img_format}'))

    print(img_list[0])

    save_dir = args.save_dir



if __name__ == '__main__':
    main()
