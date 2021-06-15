import torch
from pprint import pprint
import numpy as np



def optimizer(lr,model):
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * 2}]
            else:
                params += [{'params': [value], 'lr': lr}]
    optimizer = torch.optim.Adam(params)
    return optimizer

def if_init(anchor,bbox):  #anchor 18420 4  bbox 2 1 4
    #计算交集
    # top left
    tl = np.maximum(anchor[:, :2], bbox[:,:, :2]) 
    # bottom right
    br = np.minimum(anchor[:, 2:], bbox[:,:, 2:])

    area_i = torch.prod((br - tl), axis=2) * (tl < br).all(axis=2)
    return area_i

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
    height = 3
    width = 3
    feat_stride = 9

    shift_x = torch.arange(0, width * feat_stride, feat_stride)
    shift_y = torch.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),shift_y.ravel(), shift_x.ravel()), axis=1)#[0 0 0 0] [0 3 0 3]

    anchor = shift + anchor_base #4560 4
    anchor = anchor.astype(np.int32)
    #anchor = torch.from_numpy(anchor) #4560 4
    
    return anchor


def anchor_base_front():
    #anchor_RCN
    anchor_base = np.array([[0,0,25,25]])
    #_, _, height, width = features.shape
    height = 8
    width = 8
    feat_stride = 3

    shift_x = torch.arange(0, width * feat_stride, feat_stride)
    shift_y = torch.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),shift_y.ravel(), shift_x.ravel()), axis=1)#[0 0 0 0] [0 3 0 3]

    anchor = shift + anchor_base #4560 4
    anchor = anchor.astype(np.int32)
    #anchor = torch.from_numpy(anchor) #4560 4
    
    return anchor
















def heapify(arr, n, i,index_arr): 
    largest = i  
    l = 2 * i + 1     # left = 2*i + 1 
    r = 2 * i + 2     # right = 2*i + 2 
  
    if l < n and arr[i] < arr[l]: 
        largest = l 
  
    if r < n and arr[largest] < arr[r]: 
        largest = r 
  
    if largest != i: 
        arr[i],arr[largest] = arr[largest],arr[i]  # 交换
        index_arr[i],index_arr[largest] = index_arr[largest],index_arr[i]
        heapify(arr, n, largest,index_arr) 
  
def heapSort(arr,index_arr): 
    n = len(arr) 

    # Build a maxheap. 
    for i in range(int(n/2-1), -1, -1): 
        heapify(arr, n, i,index_arr) 
  
    # 一个个交换元素
    for i in range(n-1, n-101, -1): 
        arr[i], arr[0] = arr[0], arr[i]   # 交换
        index_arr[i],index_arr[0] = index_arr[0],index_arr[i]
        heapify(arr, i, 0,index_arr) 
  



def heap_sort(arr,l):

#索引列表
    index_arr = list()
    for i in range(l):
        index_arr.append(i)
    heapSort(arr,index_arr) 
    index_arr.reverse()

    return index_arr












