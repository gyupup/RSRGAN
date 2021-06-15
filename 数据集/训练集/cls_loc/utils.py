import torch
import numpy as np





def _smooth_l1_loss(x, t, in_weight, sigma): #x pred_loc  t gt_loc
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):

    in_weight = torch.zeros(gt_loc.shape).cuda()#64 4

    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label

    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1#64 4 根据label对应，label=1对应的行为1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= (((gt_label > 0).sum()+0.01).float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss


def bbox2loc( dst_bbox):  #输入预测框和真实框  输出偏移量
    # src_bbox 4 4
    #dst_bbox 4 4
    
    src_bbox = np.tile(np.array([0,0,80,80]),(dst_bbox.shape[0],1))
    src_bbox = src_bbox.astype(np.float32)

    #长宽
    height = src_bbox[:, 2] - src_bbox[:, 0]  #80
    width = src_bbox[:, 3] - src_bbox[:, 1]   #80
    #中心
    ctr_y = src_bbox[:, 0] + 0.5 * height    #40
    ctr_x = src_bbox[:, 1] + 0.5 * width    #40 

    #真实目标框的长宽
    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]    
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    #真实目标框的中心
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    #确保height和width不为0
    #eps = np.finfo(height.dtype).eps#获取该类型的最小值 确保不为0
    eps = 0.00001
    height = np.maximum(height, eps)#80
    width = np.maximum(width, eps)#80

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width

    dh = np.log((base_height+0.00001) / height)
    dw = np.log((base_width+0.00001) / width)



    loc = np.vstack((dy, dx, dh, dw)).transpose()
    return loc

def loc2bbox( loc):   #输入预测框和偏移量  计算回归框 

    loc = loc.detach().cpu().numpy()

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






