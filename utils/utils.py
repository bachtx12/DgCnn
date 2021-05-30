import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../models'))
def copy_parameters(model, pretrained, verbose=True, part_seg=False):
    feat_dict = model.state_dict()
    #load pre_trained self-supervised
    pretrained_dict = pretrained
    print(feat_dict.keys())
    print(pretrained_dict.keys())
    predict = {}
    if part_seg:
        for k, v in pretrained_dict.items():
            if k == 'mlp1.0.weight' or k == 'mlp1.0.bias':
                predict[k.replace('mlp1.0', 'conv1')] = v
            elif k in ['mlp1.1.weight', 'mlp1.1.bias', 'mlp1.1.running_mean', 'mlp1.1.running_var', 'mlp1.1.num_batches_tracked']:
                predict[k.replace('mlp1.1', 'bn1')]=v
            elif k == 'mlp2.3.weight' or k == 'mlp2.3.bias':
                predict[k.replace('mlp2.3', 'conv2')]=v 
            elif k in ['mlp2.4.weight', 'mlp2.4.bias', 'mlp2.4.running_mean', 'mlp2.4.running_var', 'mlp2.4.num_batches_tracked']:
                predict[k.replace('mlp2.4', 'bn2')]=v
            else:
                predict[k]=v
        # pretrained_dict = {k.replace('mlp1.0', 'conv1'): v for k, v in pretrained_dict.items() if k == 'mlp1.0.weight' or k == 'mlp1.0.bias'}
        # pretrained_dict = {k.replace('mlp1.1', 'bn1'): v for k, v in pretrained_dict.items() if k in ['mlp1.1.weight', 'mlp1.1.bias', 'mlp1.1.running_mean', 'mlp1.1.running_var', 'mlp1.1.num_batches_tracked']}
        # pretrained_dict = {k.replace('mlp2.3', 'conv2'): v for k, v in pretrained_dict.items() if k == 'mlp2.3.weight' or k == 'mlp2.3.bias'}
        # pretrained_dict = {k.replace('mlp2.4', 'bn2'): v for k, v in pretrained_dict.items() if k in ['mlp2.4.weight', 'mlp2.4.bias', 'mlp2.4.running_mean', 'mlp2.4.running_var', 'mlp2.4.num_batches_tracked']}
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if
    #                 k in feat_dict and pretrained_dict[k].size() == feat_dict[k].size()}
        pretrained_dict = {k: v for k, v in predict.items() if
                    k in feat_dict and predict[k].size() == feat_dict[k].size()}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                k in feat_dict and pretrained_dict[k].size() == feat_dict[k].size()}
    
    if verbose:
        print('=' * 27)
        print('Restored Params and Shapes:')
        for k, v in pretrained_dict.items():
            print(k, ': ', v.size())
        print('=' * 68)
    feat_dict.update(pretrained_dict)
    model.load_state_dict(feat_dict)
    return model
def to_one_hot(y, num_class):
    new_y = torch.eye(num_class)[y.cpu().data.numpy(), ]
    if y.is_cuda:
        return new_y.cuda()
    return new_y
def calculate_sem_IoU(pred_np, seg_np):
    I_all = np.zeros(13)
    U_all = np.zeros(13)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(13):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    return I_all / U_all
def calculate_shape_IoU(pred_np, seg_np, label, class_choice):
    label = label.squeeze()
    shape_ious = []
    for shape_idx in range(seg_np.shape[0]):
        if not class_choice:
            start_index = index_start[label[shape_idx]]
            num = seg_num[label[shape_idx]]
            parts = range(start_index, start_index + num)
        else:
            parts = range(seg_num[label[0]])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious
