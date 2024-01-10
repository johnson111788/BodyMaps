import os, json, time, torch, csv, numpy as np
from scipy import ndimage
from pynvml.smi import nvidia_smi
from pathlib import Path

from utils.logger import logger

default_weight_dict = {
    "spleen": 1,
    "kidney_right": 1,
    "kidney_left": 1,
    "gallbladder": 1,
    "esophagus": 1,
    "liver": 1 ,
    "stomach": 1,
    "aorta": 1,
    "inferior_vena_cava": 1, # inferior_vena_cava postcava
    "portal_vein_and_splenic_vein": 1,
    "pancreas": 1,
    "adrenal_gland_right": 1,
    "adrenal_gland_left": 1,
    "duodenum": 1,
    "hepatic_vessel": 1, # no
    "lung_right": 1, # merge
    "lung_left": 1, # merge
    "colon": 1,
    "intestine": 1, # no
    "rectum": 1, # no
    "urinary_bladder": 1, # bladder
    "prostate": 1, # no
    "femur_left": 1,
    "femur_right": 1,
    "celiac_trunk": 1, # no
}


def dice_score(preds, labels, spe_sen=False):  # on GPU
    ### preds: w,h,d; label: w,h,d
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    preds = torch.where(preds > 0.5, 1., 0.)
    predict = preds.contiguous().view(1, -1)
    target = labels.contiguous().view(1, -1)

    tp = torch.sum(torch.mul(predict, target))
    fn = torch.sum(torch.mul(predict!=1, target))
    fp = torch.sum(torch.mul(predict, target!=1))
    tn = torch.sum(torch.mul(predict!=1, target!=1))

    den = torch.sum(predict) + torch.sum(target) + 1

    dice = 2 * tp / den
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    specificity = tn/(fp + tn)


    # print(dice, recall, precision)
    if spe_sen:
        return dice, recall, precision, specificity
    else:
        return dice, recall, precision


def get_mask_edges(mask):
    b_mask = mask ==1
    # convert to bool array
    if isinstance(b_mask, torch.Tensor):
        b_mask = b_mask.cpu().numpy()  # Move to CPU and convert to numpy if it's a CUDA tensor
    edges = ndimage.binary_erosion(b_mask) ^ b_mask
    return edges

def get_surface_distance(mask1,mask2,spacing):
    edges1 = get_mask_edges(mask1)
    edges2 = get_mask_edges(mask2)
    dis = ndimage.distance_transform_edt(~edges1, sampling=spacing)
    return np.asarray(dis[edges2])

def surface_dice(mask1,mask2,spacing,tolerance):
    dis1 = get_surface_distance(mask1,mask2,spacing)
    dis2 = get_surface_distance(mask2,mask1,spacing)
    boundary_complete = len(dis1) + len(dis2)
    boundary_correct = np.sum(dis1 <= tolerance) + np.sum(dis2 <= tolerance)
    nsd = boundary_correct / boundary_complete
    return nsd

def get_testcase(V_tlimit, A_tlimit):
    test_cases_infor=[*csv.DictReader(open(V_tlimit))] + [*csv.DictReader(open(A_tlimit))]
    for num, test_case_infor in enumerate(test_cases_infor):
        test_cases_infor[num]['name'] = test_case_infor['name']+'.nii.gz'
    return test_cases_infor

def check_dir(file_path):
    file_path = Path(file_path)
    files = [f for f in file_path.iterdir() if ".nii.gz" in str(f)]
    if len(files) != 0:
        return False
    return True