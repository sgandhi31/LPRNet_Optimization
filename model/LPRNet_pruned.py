import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import time
import copy
from LPRNet import LPRNet
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='pruned the model')
    parser.add_argument('--pretrained_model', default = 'weights/Final_LPRNet.pth', help = "load the model to prune")
    parser.add_argument('--pruning_ratio', type=float, default = 0.5, help = "set the pruning ratio")
    parser.add_argument('--verbose', default = False, type = bool, help = 'get non zero parameters')
    parser.add_argument('--save',default = 'unstructured_pruned_LPRNet_model_0.5.pth', help = 'save the pruned model')
    return parser.parse_args()

def load_pretrained_model(checkpoint_path):
    lprnet = LPRNet(lpr_max_len=8, phase=False, class_num=68, dropout_rate=0.5)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    if 'state_dict' in checkpoint:
        lprnet.load_state_dict(checkpoint['state_dict'])
    elif 'lprnet' in checkpoint:
        lprnet.load_state_dict(checkpoint['lprnet'])
    else:
        lprnet.load_state_dict(checkpoint)
    lprnet.eval()
    return lprnet


def get_model_size(model):
    num_nonzeros = sum(param.count_nonzero().item() for param in model.parameters())
    print(f"No. of Non-Zero Parameters: {num_nonzeros}")
    return num_nonzeros

def unstructured_pruning(model, prune_ratio):
    pruned_model = copy.deepcopy(model)
    for module in pruned_model.modules():
      if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
          prune.l1_unstructured(module, name="weight", amount=prune_ratio)
    return pruned_model
    # pruned_models = []
    # for prune_ratio in prune_ratios:
    #     pruned_model = copy.deepcopy(model)
    #     for module in pruned_model.modules():
    #         if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
    #             prune.l1_unstructured(module, name="weight", amount=prune_ratio)
    #     pruned_models.append(pruned_model)

# apply pruning
def apply_pruning():
  args = get_parser()
  original_model = load_pretrained_model(args.pretrained_model)
  print("LPRNet model loaded successfully!")
  pruned_model = unstructured_pruning(original_model, args.pruning_ratio)
  print('model has been pruned')
  original_size = get_model_size(original_model)
  # Remove pruning reparameterization
  for module in pruned_model.modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        prune.remove(module, "weight")
  if args.verbose:
    # Get pruned model size
    print(f"Pruned model with ratio {args.pruning_ratio}:")
    pruned_size = get_model_size(pruned_model)

    # Calculate and print the reduction percentage
    reduction_percentage = (original_size - pruned_size) / original_size * 100
    print(f"Parameter reduction: {reduction_percentage:.2f}%")

  if args.save:
    torch.save(pruned_model.state_dict(), args.save)

if __name__=='__main__':
  apply_pruning()
