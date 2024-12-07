# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
test pretrained model.
Author: aiboy.wei@outlook.com .
'''

from data.load_data import CHARS, CHARS_DICT, LPRDataLoader
from PIL import Image, ImageDraw, ImageFont
from model.LPRNet_quantize import build_lprnet_quantize
# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import cv2
import os
import matplotlib.pyplot as plt
from torch.ao.quantization import QuantStub, DeQuantStub

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--test_img_dirs', default="./data/test", help='the test images path')
    parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--test_batch_size', default=100, help='testing batch size.')
    parser.add_argument('--phase_train', default=False, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=False, type=bool, help='Use cuda to train model')
    parser.add_argument('--show', default=False, type=bool, help='show test image and its predict result or not.')
    parser.add_argument('--quantized_model', default='./weights/Final_LPRNet_model.pth', help="apply post training static quantization on the model")

    args = parser.parse_args()

    return args

class QuantizedLPRNet(nn.Module):
    def __init__(self, model):
        super(QuantizedLPRNet, self).__init__()
        self.quant = QuantStub()  # Handles input quantization
        self.model = model
        self.dequant = DeQuantStub()  # Handles output dequantization

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

def load_pretrained_model(model_path):
    # Load the pretrained LPRNet model
    model = build_lprnet_quantize(lpr_max_len=8, phase=False, class_num=68, dropout_rate=0.5)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def apply_quantization(model, calibration_data):
    # Wrap the model with QuantStub and DeQuantStub
    model = QuantizedLPRNet(model)
    
    # Define the quantization configuration
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Prepare the model for quantization
    model_prepared = torch.ao.quantization.prepare(model, inplace = False)
    
    # Calibrate the model using a representative dataset
    model_prepared.eval()
    with torch.no_grad():
      model_prepared(calibration_data)
    
    # Convert to quantized version
    model_quantized = torch.ao.quantization.convert(model_prepared, inplace = False)
    return model_quantized

def calibrate_quantized_model(model_path, datasets, datapoints_len = 100):
  model = load_pretrained_model(model_path)
  calibration_data,_,_ = next(iter(DataLoader(datasets, datapoints_len, shuffle = True, collate_fn = collate_fn)))
  quantized_model = apply_quantization(model, calibration_data)
  return quantized_model


def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.float32)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)

def test():
    args = get_parser()


    test_img_dirs = os.path.expanduser(args.test_img_dirs)
    test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len)
    
    lprnet = calibrate_quantized_model(args.quantized_model, test_dataset)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    try:
        Greedy_Decode_Eval(lprnet, test_dataset, args)
    finally:
        # cv2.destroyAllWindows()
        pass

def Greedy_Decode_Eval(Net, datasets, args):
    # TestNet = Net.eval()
    epoch_size = len(datasets) // args.test_batch_size
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    t1 = time.time()
    for i in range(epoch_size):
        # load train data
        images, labels, lengths = next(batch_iterator)
        start = 0
        targets = []
        for length in lengths:
            label = labels[start:start+length]
            targets.append(label)
            start += length
        targets = np.array([el.numpy() for el in targets])
        imgs = images.numpy().copy()

        if args.cuda:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        # forward
        prebs = Net(images)
        # greedy decode
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label: # dropout repeate label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
        for i, label in enumerate(preb_labels):
            # show image and its predict label
            if args.show:
                show(imgs[i], label, targets[i])
            if len(label) != len(targets[i]):
                Tn_1 += 1
                continue
            if (np.asarray(targets[i]) == np.asarray(label)).all():
                Tp += 1
            else:
                Tn_2 += 1
    Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
    print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
    t2 = time.time()
    print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))

def show(img, label, target):
    """Save or display the image inline instead of using cv2.imshow"""
    img = img.transpose(1, 2, 0)  # Convert CHW to HWC
    img = (img * 255).astype(np.uint8)  # Convert to uint8

    plt.figure(figsize=(5, 2))
    plt.imshow(img)
    plt.title(f'Predicted: {label}, Target: {target}')
    plt.axis('off')
    plt.show()

# def show(img, label, target):
#     img = np.transpose(img, (1, 2, 0))
#     img *= 128.
#     img += 127.5
#     img = img.astype(np.uint8)

#     lb = ""
#     for i in label:
#         lb += CHARS[i]
#     tg = ""
#     for j in target.tolist():
#         tg += CHARS[int(j)]

#     flag = "F"
#     if lb == tg:
#         flag = "T"
#     # img = cv2.putText(img, lb, (0,16), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 0, 255), 1)
#     img = cv2ImgAddText(img, lb, (0, 0))
#     cv2.imshow("test", img)
#     print("target: ", tg, " ### {} ### ".format(flag), "predict: ", lb)
#     cv2.waitKey()
#     cv2.destroyAllWindows()

def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=12):
    if (isinstance(img, np.ndarray)):  # detect opencv format or not
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("data/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
    draw.text(pos, text, textColor, font=fontText)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    test()
