import time
from data.load_data import CHARS, CHARS_DICT, LPRDataLoader
from PIL import Image, ImageDraw, ImageFont
from model.LPRNet import build_lprnet
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

from mlc.build_autotune import compile
import tvm

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
    parser.add_argument('--log_file', default = './tuning_log.json', help = "log_file to compile optimized MLC model")
    parser.add_argument('--optimized_model', default='./weights/Final_LPRNet_model.pth', help='MLC optimized model')

    args = parser.parse_args()

    return args

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
  datasets = LPRDataLoader(test_img_dirs.split(','), [94, 24], 8)

  epoch_size = len(datasets) // 1
  batch_iterator = iter(DataLoader(datasets, 1, shuffle=True, num_workers=8, collate_fn=collate_fn))
  Tp = 0
  Tn_1 = 0
  Tn_2 = 0
  t1 = time.time()
  module = compile(args.optimized_model, args.log_file)
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

      images = Variable(images)

      module.set_input("data", tvm.nd.array(images.numpy()))

      # Run inference
      module.run()

      # Get output
      prebs = module.get_output(0).asnumpy()
      # greedy decode
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

if __name__ == "__main__":
    test()
