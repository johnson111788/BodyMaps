import numpy as np
import pandas as pd
import cc3d, fastremap
from scipy import ndimage

from monai.data import decollate_batch
from monai.transforms import Compose, Invertd



TEMPLATE={
    '01': [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
    '01_2': [1,3,4,5,6,7,11,14],
    '02': [1,3,4,5,6,7,11,14],
    '03': [6],
    '04': [6,27], # post process
    '05': [2,3,26,32], # post process
    '06': [1,2,3,4,6,7,11,16,17],
    '07': [6,1,3,2,7,4,5,11,14,18,19,12,13,20,21,23,24],
    '08': [6, 2, 3, 1, 11],
    '09': [1,2,3,4,5,6,7,8,9,11,12,13,14,21,22],
    '12': [6,21,16,17,2,3],  
    '13': [6,2,3,1,11,8,9,7,4,5,12,13,25], 
    '14': [1,2,3,4,5,6,7,8,9,10,12,13],
    '10_03': [6, 27], # post process
    '10_06': [30],
    '10_07': [11, 28], # post process
    '10_08': [15, 29], # post process
    '10_09': [1],
    '10_10': [31],
    '15': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17], ## total segmentation
    'all': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32], 
    'target':[1,2,3,4,6,7,8,9,11], ## target organ index
    'assemble':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
}

ORGAN_NAME = ['Spleen', 'Right Kidney', 'Left Kidney', 'Gall Bladder', 'Esophagus', 
                'Liver', 'Stomach', 'Arota', 'Postcava', 'Portal Vein and Splenic Vein',
                'Pancreas', 'Right Adrenal Gland', 'Left Adrenal Gland', 'Duodenum', 'Hepatic Vessel',
                'Right Lung', 'Left Lung', 'Colon', 'Intestine', 'Rectum', 
                'Bladder', 'Prostate', 'Left Head of Femur', 'Right Head of Femur', 'Celiac Truck',
                'Kidney Tumor', 'Liver Tumor', 'Pancreas Tumor', 'Hepatic Vessel Tumor', 'Lung Tumor', 'Colon Tumor', 'Kidney Cyst']

TUMOR_SIZE = {
    'Kidney Tumor': 80, 
    'Liver Tumor': 20, 
    'Pancreas Tumor': 100, 
    'Hepatic Vessel Tumor': 80, 
    'Lung Tumor': 30, 
    'Colon Tumor': 100, 
    'Kidney Cyst': 20
}

TUMOR_NUM = {
    'Kidney Tumor': 5, 
    'Liver Tumor': 20, 
    'Pancreas Tumor': 1, 
    'Hepatic Vessel Tumor': 10, 
    'Lung Tumor': 10, 
    'Colon Tumor': 3, 
    'Kidney Cyst': 20
}

TUMOR_ORGAN = {
    'Kidney Tumor': [2,3], 
    'Liver Tumor': [6], 
    'Pancreas Tumor': [11], 
    'Hepatic Vessel Tumor': [15], 
    'Lung Tumor': [16,17], 
    'Colon Tumor': [18], 
    'Kidney Cyst': [2,3]
}


def organ_post_process(pred_mask, organ_list,args):
    post_pred_mask = np.zeros(pred_mask.shape)

    for b in range(pred_mask.shape[0]):
        for organ in organ_list:
            if organ == 11: # both process pancreas and Portal vein and splenic vein
                post_pred_mask[b,10] = extract_topk_largest_candidates(pred_mask[b,10], 1) # for pancreas
                if 10 in organ_list:
                    post_pred_mask[b,9] = PSVein_post_process(pred_mask[b,9], post_pred_mask[b,10])
                    # post_pred_mask[b,9] = pred_mask[b,9]
                # post_pred_mask[b,organ-1] = extract_topk_largest_candidates(pred_mask[b,organ-1], 1)
            elif organ == 16:
                try:
                    left_lung_mask, right_lung_mask = lung_post_process(pred_mask[b])
                    post_pred_mask[b,16] = left_lung_mask
                    post_pred_mask[b,15] = right_lung_mask
                except IndexError:
                    print('this case does not have lungs!')
                    shape_temp = post_pred_mask[b,16].shape
                    post_pred_mask[b,16] = np.zeros(shape_temp)
                    post_pred_mask[b,15] = np.zeros(shape_temp)

                right_lung_size = np.sum(post_pred_mask[b,15],axis=(0,1,2))
                left_lung_size = np.sum(post_pred_mask[b,16],axis=(0,1,2))
                
                print('left lung size: '+str(left_lung_size))
                print('right lung size: '+str(right_lung_size))

                total_anomly_slice_number=0

                if right_lung_size>left_lung_size:
                    if right_lung_size/left_lung_size > 4:
                        mid_point = int(right_lung_mask.shape[0]/2)
                        left_region = np.sum(right_lung_mask[:mid_point,:,:],axis=(0,1,2))
                        right_region = np.sum(right_lung_mask[mid_point:,:,:],axis=(0,1,2))
                        
                        if (right_region+1)/(left_region+1)>4:
                            print('this case only has right lung')
                            post_pred_mask[b,15] = right_lung_mask
                            post_pred_mask[b,16] = np.zeros(right_lung_mask.shape)
                        elif (left_region+1)/(right_region+1)>4:
                            print('this case only has left lung')
                            post_pred_mask[b,16] = right_lung_mask
                            post_pred_mask[b,15] = np.zeros(right_lung_mask.shape)
                        else:
                            print('need anomaly detection')
                            print('start anomly detection at right lung')
                            try:
                                left_lung_mask,right_lung_mask,total_anomly_slice_number = anomly_detection(
                                    pred_mask,post_pred_mask[b,15],b,total_anomly_slice_number)
                                post_pred_mask[b,16] = left_lung_mask
                                post_pred_mask[b,15] = right_lung_mask
                                right_lung_size = np.sum(post_pred_mask[b,15],axis=(0,1,2))
                                left_lung_size = np.sum(post_pred_mask[b,16],axis=(0,1,2))
                                while right_lung_size/left_lung_size>4 or left_lung_size/right_lung_size>4:
                                    print('still need anomly detection')
                                    if right_lung_size>left_lung_size:
                                        left_lung_mask,right_lung_mask,total_anomly_slice_number = anomly_detection(
                                        pred_mask,post_pred_mask[b,15],b,total_anomly_slice_number)
                                    else:
                                        left_lung_mask,right_lung_mask,total_anomly_slice_number = anomly_detection(
                                        pred_mask,post_pred_mask[b,16],b,total_anomly_slice_number)
                                    post_pred_mask[b,16] = left_lung_mask
                                    post_pred_mask[b,15] = right_lung_mask
                                    right_lung_size = np.sum(post_pred_mask[b,15],axis=(0,1,2))
                                    left_lung_size = np.sum(post_pred_mask[b,16],axis=(0,1,2))
                                print('lung seperation complete')
                            except IndexError:
                                left_lung_mask, right_lung_mask = lung_post_process(pred_mask[b])
                                post_pred_mask[b,16] = left_lung_mask
                                post_pred_mask[b,15] = right_lung_mask
                                print("cannot seperate two lungs, writing csv")


                else:
                    if left_lung_size/right_lung_size > 4:
                        mid_point = int(left_lung_mask.shape[0]/2)
                        left_region = np.sum(left_lung_mask[:mid_point,:,:],axis=(0,1,2))
                        right_region = np.sum(left_lung_mask[mid_point:,:,:],axis=(0,1,2))
                        if (right_region+1)/(left_region+1)>4:
                            print('this case only has right lung')
                            post_pred_mask[b,15] = left_lung_mask
                            post_pred_mask[b,16] = np.zeros(left_lung_mask.shape)
                        elif (left_region+1)/(right_region+1)>4:
                            print('this case only has left lung')
                            post_pred_mask[b,16] = left_lung_mask
                            post_pred_mask[b,15] = np.zeros(left_lung_mask.shape)
                        else:

                            print('need anomly detection')
                            print('start anomly detection at left lung')
                            try:
                                left_lung_mask,right_lung_mask,total_anomly_slice_number = anomly_detection(
                                    pred_mask,post_pred_mask[b,16],b,total_anomly_slice_number)
                                post_pred_mask[b,16] = left_lung_mask
                                post_pred_mask[b,15] = right_lung_mask
                                right_lung_size = np.sum(post_pred_mask[b,15],axis=(0,1,2))
                                left_lung_size = np.sum(post_pred_mask[b,16],axis=(0,1,2))
                                while right_lung_size/left_lung_size>4 or left_lung_size/right_lung_size>4:
                                    print('still need anomly detection')
                                    if right_lung_size>left_lung_size:
                                        left_lung_mask,right_lung_mask,total_anomly_slice_number = anomly_detection(
                                        pred_mask,post_pred_mask[b,15],b,total_anomly_slice_number)
                                    else:
                                        left_lung_mask,right_lung_mask,total_anomly_slice_number = anomly_detection(
                                        pred_mask,post_pred_mask[b,16],b,total_anomly_slice_number)
                                    post_pred_mask[b,16] = left_lung_mask
                                    post_pred_mask[b,15] = right_lung_mask
                                    right_lung_size = np.sum(post_pred_mask[b,15],axis=(0,1,2))
                                    left_lung_size = np.sum(post_pred_mask[b,16],axis=(0,1,2))

                                print('lung seperation complete')
                            except IndexError:
                                left_lung_mask, right_lung_mask = lung_post_process(pred_mask[b])
                                post_pred_mask[b,16] = left_lung_mask
                                post_pred_mask[b,15] = right_lung_mask
                                print("cannot seperate two lungs, writing csv")

                print('find number of anomaly slice: '+str(total_anomly_slice_number))



                
            elif organ == 17:
                continue ## the le
            elif organ in [1,2,3,4,5,6,7,8,9,12,13,14,18,19,20,21,22,23,24,25]: ## rest organ index
                post_pred_mask[b,organ-1] = extract_topk_largest_candidates(pred_mask[b,organ-1], 1)
            elif organ in [28,29,30,31,32]:
                post_pred_mask[b,organ-1] = extract_topk_largest_candidates(pred_mask[b,organ-1], TUMOR_NUM[ORGAN_NAME[organ-1]], area_least=TUMOR_SIZE[ORGAN_NAME[organ-1]])
            elif organ in [26,27]:
                organ_mask = merge_and_top_organ(pred_mask[b], TUMOR_ORGAN[ORGAN_NAME[organ-1]])
                post_pred_mask[b,organ-1] = organ_region_filter_out(pred_mask[b,organ-1], organ_mask)
                post_pred_mask[b,organ-1] = extract_topk_largest_candidates(post_pred_mask[b,organ-1], TUMOR_NUM[ORGAN_NAME[organ-1]], area_least=TUMOR_SIZE[ORGAN_NAME[organ-1]])
                print('filter out')
            else:
                post_pred_mask[b,organ-1] = pred_mask[b,organ-1]
    return post_pred_mask,total_anomly_slice_number

def lung_overlap_post_process(pred_mask):
    new_mask = np.zeros(pred_mask.shape, np.uint8)
    new_mask[pred_mask==1] = 1
    label_out = cc3d.connected_components(new_mask, connectivity=26)

    areas = {}
    for label, extracted in cc3d.each(label_out, binary=True, in_place=True):
        areas[label] = fastremap.foreground(extracted)
    candidates = sorted(areas.items(), key=lambda item: item[1], reverse=True)
    num_candidates = len(candidates)
    if num_candidates!=1:
        print('start separating two lungs!')
        ONE = int(candidates[0][0])
        TWO = int(candidates[1][0])


        print('number of connected components:'+str(len(candidates)))
        a1,b1,c1 = np.where(label_out==ONE)
        a2,b2,c2 = np.where(label_out==TWO)
        
        left_lung_mask = np.zeros(label_out.shape)
        right_lung_mask = np.zeros(label_out.shape)

        if np.mean(a1) < np.mean(a2):
            left_lung_mask[label_out==ONE] = 1
            right_lung_mask[label_out==TWO] = 1
        else:
            right_lung_mask[label_out==ONE] = 1
            left_lung_mask[label_out==TWO] = 1
        erosion_left_lung_size = np.sum(left_lung_mask,axis=(0,1,2))
        erosion_right_lung_size = np.sum(right_lung_mask,axis=(0,1,2))
        print('erosion left lung size:'+str(erosion_left_lung_size))
        print('erosion right lung size:'+ str(erosion_right_lung_size))
        return num_candidates,left_lung_mask, right_lung_mask
    else:
        print('current iteration cannot separate lungs, erosion iteration + 1')
        ONE = int(candidates[0][0])
        print('number of connected components:'+str(len(candidates)))
        lung_mask = np.zeros(label_out.shape)
        lung_mask[label_out == ONE]=1
        lung_overlapped_mask_size = np.sum(lung_mask,axis=(0,1,2))
        print('lung overlapped mask size:' + str(lung_overlapped_mask_size))

        return num_candidates,lung_mask

def find_best_iter_and_masks(lung_mask):
    iter=1
    print('current iteration:' + str(iter))
    struct2 = ndimage.generate_binary_structure(3, 3)
    erosion_mask= ndimage.binary_erosion(lung_mask, structure=struct2,iterations=iter)
    candidates_and_masks = lung_overlap_post_process(erosion_mask)
    while candidates_and_masks[0]==1:
        iter +=1
        print('current iteration:' + str(iter))
        erosion_mask= ndimage.binary_erosion(lung_mask, structure=struct2,iterations=iter)
        candidates_and_masks = lung_overlap_post_process(erosion_mask)
    print('check if components are valid')
    left_lung_erosion_mask = candidates_and_masks[1]
    right_lung_erosion_mask = candidates_and_masks[2]
    left_lung_erosion_mask_size = np.sum(left_lung_erosion_mask,axis = (0,1,2))
    right_lung_erosion_mask_size = np.sum(right_lung_erosion_mask,axis = (0,1,2))
    while left_lung_erosion_mask_size/right_lung_erosion_mask_size>4 or right_lung_erosion_mask_size/left_lung_erosion_mask_size>4:
        print('components still have large difference, erosion interation + 1')
        iter +=1
        print('current iteration:' + str(iter))
        erosion_mask= ndimage.binary_erosion(lung_mask, structure=struct2,iterations=iter)
        candidates_and_masks = lung_overlap_post_process(erosion_mask)
        while candidates_and_masks[0]==1:
            iter +=1
            print('current iteration:' + str(iter))
            erosion_mask= ndimage.binary_erosion(lung_mask, structure=struct2,iterations=iter)
            candidates_and_masks = lung_overlap_post_process(erosion_mask)
        left_lung_erosion_mask = candidates_and_masks[1]
        right_lung_erosion_mask = candidates_and_masks[2]
        left_lung_erosion_mask_size = np.sum(left_lung_erosion_mask,axis = (0,1,2))
        right_lung_erosion_mask_size = np.sum(right_lung_erosion_mask,axis = (0,1,2))
    print('erosion done, best iteration: '+str(iter))



    print('start dilation')
    left_lung_erosion_mask = candidates_and_masks[1]
    right_lung_erosion_mask = candidates_and_masks[2]

    erosion_part_mask = lung_mask - left_lung_erosion_mask - right_lung_erosion_mask
    left_lung_dist = np.ones(left_lung_erosion_mask.shape)
    right_lung_dist = np.ones(right_lung_erosion_mask.shape)
    left_lung_dist[left_lung_erosion_mask==1]=0
    right_lung_dist[right_lung_erosion_mask==1]=0
    left_lung_dist_map = ndimage.distance_transform_edt(left_lung_dist)
    right_lung_dist_map = ndimage.distance_transform_edt(right_lung_dist)
    left_lung_dist_map[erosion_part_mask==0]=0
    right_lung_dist_map[erosion_part_mask==0]=0
    left_lung_adding_map = left_lung_dist_map < right_lung_dist_map
    right_lung_adding_map = right_lung_dist_map < left_lung_dist_map
    
    left_lung_erosion_mask[left_lung_adding_map==1]=1
    right_lung_erosion_mask[right_lung_adding_map==1]=1

    left_lung_mask = left_lung_erosion_mask
    right_lung_mask = right_lung_erosion_mask
    print('dilation complete')
    left_lung_mask_fill_hole = ndimage.binary_fill_holes(left_lung_mask)
    right_lung_mask_fill_hole = ndimage.binary_fill_holes(right_lung_mask)
    left_lung_size = np.sum(left_lung_mask_fill_hole,axis=(0,1,2))
    right_lung_size = np.sum(right_lung_mask_fill_hole,axis=(0,1,2))
    print('new left lung size:'+str(left_lung_size))
    print('new right lung size:' + str(right_lung_size))
    return left_lung_mask_fill_hole,right_lung_mask_fill_hole

def anomly_detection(pred_mask,post_pred_mask,batch,anomly_num):
    total_anomly_slice_number = anomly_num
    df = get_dataframe(post_pred_mask)

    lung_df = df[df['array_sum']!=0]
    lung_df['SMA20'] = lung_df['array_sum'].rolling(20,min_periods=1,center=True).mean()
    lung_df['STD20'] = lung_df['array_sum'].rolling(20,min_periods=1,center=True).std()
    lung_df['SMA7'] = lung_df['array_sum'].rolling(7,min_periods=1,center=True).mean()
    lung_df['upper_bound'] = lung_df['SMA20']+2*lung_df['STD20']
    lung_df['Predictions'] = lung_df['array_sum']>lung_df['upper_bound']
    lung_df['Predictions'] = lung_df['Predictions'].astype(int)
    lung_df.dropna(inplace=True)
    anomly_df = lung_df[lung_df['Predictions']==1]
    anomly_slice = anomly_df['slice_index'].to_numpy()
    anomly_value = anomly_df['array_sum'].to_numpy()
    anomly_SMA7 = anomly_df['SMA7'].to_numpy()

    print('decision made')
    if len(anomly_df)!=0:
        print('anomaly point detected')
        print('check if the anomaly points are real')
        real_anomly_slice = []
        for i in range(len(anomly_df)):
            if anomly_value[i] > anomly_SMA7[i]+200:
                print('the anomaly point is real')
                real_anomly_slice.append(anomly_slice[i])
                total_anomly_slice_number+=1
        
        if len(real_anomly_slice)!=0:

            print('anomaly detection plot created')
            for s in real_anomly_slice:
                pred_mask[batch,15,:,:,s]=0
                pred_mask[batch,16,:,:,s]=0
            left_lung_mask, right_lung_mask = lung_post_process(pred_mask[batch])
            left_lung_size = np.sum(left_lung_mask,axis=(0,1,2))
            right_lung_size = np.sum(right_lung_mask,axis=(0,1,2))
            print('new left lung size:'+str(left_lung_size))
            print('new right lung size:' + str(right_lung_size))
            return left_lung_mask,right_lung_mask,total_anomly_slice_number
        else: 
            print('the anomaly point is not real, start separate overlapping')
            left_lung_mask,right_lung_mask = find_best_iter_and_masks(post_pred_mask)
            return left_lung_mask,right_lung_mask,total_anomly_slice_number


    print('overlap detected, start erosion and dilation')
    left_lung_mask,right_lung_mask = find_best_iter_and_masks(post_pred_mask)

    return left_lung_mask,right_lung_mask,total_anomly_slice_number

def get_dataframe(post_pred_mask):
    target_array = post_pred_mask
    target_array_sum = np.sum(target_array,axis=(0,1))
    slice_index = np.arange(target_array.shape[-1])
    df = pd.DataFrame({'slice_index':slice_index,'array_sum':target_array_sum})
    return df
            
def merge_and_top_organ(pred_mask, organ_list):
    ## merge 
    out_mask = np.zeros(pred_mask.shape[1:], np.uint8)
    for organ in organ_list:
        out_mask = np.logical_or(out_mask, pred_mask[organ-1])
    ## select the top k, for righr left case
    out_mask = extract_topk_largest_candidates(out_mask, len(organ_list))

    return out_mask

def organ_region_filter_out(tumor_mask, organ_mask):
    ## dialtion
    organ_mask = ndimage.binary_closing(organ_mask, structure=np.ones((5,5,5)))
    organ_mask = ndimage.binary_dilation(organ_mask, structure=np.ones((5,5,5)))
    ## filter out
    tumor_mask = organ_mask * tumor_mask

    return tumor_mask


def PSVein_post_process(PSVein_mask, pancreas_mask):
    xy_sum_pancreas = pancreas_mask.sum(axis=0).sum(axis=0)
    z_non_zero = np.nonzero(xy_sum_pancreas)
    if len(z_non_zero[0])!= 0:
        z_value = np.min(z_non_zero) ## the down side of pancreas
        new_PSVein = PSVein_mask.copy()
        new_PSVein[:,:,:z_value] = 0
    else:
        new_PSVein = PSVein_mask.copy()
    return new_PSVein

def lung_post_process(pred_mask):
    new_mask = np.zeros(pred_mask.shape[1:], np.uint8)
    new_mask[pred_mask[15] == 1] = 1
    new_mask[pred_mask[16] == 1] = 1
    label_out = cc3d.connected_components(new_mask, connectivity=26)
    
    areas = {}
    for label, extracted in cc3d.each(label_out, binary=True, in_place=True):
        areas[label] = fastremap.foreground(extracted)
    candidates = sorted(areas.items(), key=lambda item: item[1], reverse=True)

    ONE = int(candidates[0][0])
    TWO = int(candidates[1][0])

    # raise
    # print(candidates.shape)
    # print(len(candidates))
    # raise
    a1,b1,c1 = np.where(label_out==ONE)
    a2,b2,c2 = np.where(label_out==TWO)
    
    left_lung_mask = np.zeros(label_out.shape)
    right_lung_mask = np.zeros(label_out.shape)

    if np.mean(a1) < np.mean(a2):
        left_lung_mask[label_out==ONE] = 1
        right_lung_mask[label_out==TWO] = 1
    else:
        right_lung_mask[label_out==ONE] = 1
        left_lung_mask[label_out==TWO] = 1
    
    left_lung_mask_fill_hole = ndimage.binary_fill_holes(left_lung_mask)
    right_lung_mask_fill_hole = ndimage.binary_fill_holes(right_lung_mask)
    return left_lung_mask_fill_hole, right_lung_mask_fill_hole

def extract_topk_largest_candidates(npy_mask, organ_num, area_least=0):
    ## npy_mask: w, h, d
    ## organ_num: the maximum number of connected component
    out_mask = np.zeros(npy_mask.shape, np.uint8)
    t_mask = npy_mask.copy()
    keep_topk_largest_connected_object(t_mask, organ_num, area_least, out_mask, 1)

    return out_mask


def keep_topk_largest_connected_object(npy_mask, k, area_least, out_mask, out_label):
    labels_out = cc3d.connected_components(npy_mask, connectivity=26)
    areas = {}
    for label, extracted in cc3d.each(labels_out, binary=True, in_place=True):
        areas[label] = fastremap.foreground(extracted)
    candidates = sorted(areas.items(), key=lambda item: item[1], reverse=True)

    for i in range(min(k, len(candidates))):
        if candidates[i][1] > area_least:
            out_mask[labels_out == int(candidates[i][0])] = out_label

def invert_transform(invert_key = str,batch = None, input_transform = None ):
    post_transforms = Compose([
        Invertd(
            keys=invert_key,
            transform=input_transform,
            orig_keys="image",
            nearest_interp=True,
            to_tensor=True,
        ),
    ])
    BATCH = [post_transforms(i) for i in decollate_batch(batch)]
    return BATCH
