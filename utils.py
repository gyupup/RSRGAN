import torch
from pprint import pprint
import numpy as np


def anchor_base_RCN():
    #anchor_RCN
    anchor_base = np.array([[0,0,48,48]])
    #_, _, height, width = features.shape
    height = 60
    width = 76
    feat_stride = 8

    shift_x = torch.arange(0, width * feat_stride, feat_stride)
    shift_y = torch.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),shift_y.ravel(), shift_x.ravel()), axis=1)#[0 0 0 0] [0 4 0 4]

    anchor = shift + anchor_base #4560 4
    anchor = anchor.astype(np.float32)
    anchor = torch.from_numpy(anchor) #4560 4
    
    return anchor


def anchor_base_SRGAN():
    #anchor_RCN
    anchor_base = np.array([[0,0,20,20]])
    #_, _, height, width = features.shape
    height = 4
    width = 4
    feat_stride = 9

    shift_x = torch.arange(0, width * feat_stride, feat_stride)
    shift_y = torch.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),shift_y.ravel(), shift_x.ravel()), axis=1)#[0 0 0 0] [0 3 0 3]

    anchor = shift + anchor_base #4560 4
    anchor = anchor.astype(np.int32)
    #anchor = torch.from_numpy(anchor) #4560 4
    
    return anchor

def loc2bbox( loc):   #输入预测框和偏移量  计算回归框 

    src_bbox = np.array([[0,0,80,80]])

    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    h = np.exp(dh) * src_height[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]

    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox


























