import os, time, csv, glob, warnings, torch
import numpy as np, nibabel as nib
warnings.filterwarnings("ignore")
from monai.transforms import AsDiscrete

from utils.utils import dice_score, surface_dice, default_weight_dict

from multiprocessing import Pool, Manager


class_map_w1k = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "aorta",
    8: "inferior_vena_cava",
    9: "pancreas",
    10: "adrenal_gland_right",
    11: "adrenal_gland_left",
    12: "duodenum",
    13: "colon",
    14: "intestine",
    15: "celiac_trunk",
}

def read_label(lbl_dir):
    array = nib.load(lbl_dir)
    pixdim = array.header['pixdim']
    spacing_mm = tuple(pixdim[1:4])
    array = array.get_fdata() 

    return array, spacing_mm


def metrics_computer(lbl_dir, pred_path, out_path, organ_dice_results, organ_nsd_results):
    post_label = AsDiscrete(to_onehot=16)

    csv_dsc = open(os.path.join(out_path, 'dice_results.csv'), 'a')
    fieldnames = ["name"] + list(class_map_w1k.values())
    csv_dsc_writer = csv.DictWriter(csv_dsc, fieldnames=fieldnames)

    csv_nsd = open(os.path.join(out_path, 'nsd_results.csv'), 'a')
    csv_nsd_writer = csv.DictWriter(csv_nsd, fieldnames=fieldnames)

    case_name = os.path.basename(pred_path).split('.')[0]
    start_time = time.time()
    pred = nib.load(pred_path)
    pred = pred.get_fdata() 
    
    lbl, spacing_mm = read_label(os.path.join(lbl_dir, 'label',case_name+'.nii.gz'))
    
    lbl = post_label(lbl[np.newaxis,:])
    pred = post_label(pred[np.newaxis,:])
    
    dice_case_result = {"name": case_name}
    nsd_case_result = {"name": case_name}
    
    for class_idx, class_name in zip(class_map_w1k.keys(), class_map_w1k.values()):
        # dice, nsd = cal_dice_nsd(pred[class_idx], lbl[class_idx], spacing_mm, 1)  # unpack the returned tuple and only take the dice score
        dice, _, _ = dice_score(torch.from_numpy(pred[class_idx]), torch.from_numpy(lbl[class_idx]))  # unpack the returned tuple and only take the dice score
        dice = dice.item() if torch.is_tensor(dice) else dice  # convert tensor to Python native data type if it's a tensor
        nsd = surface_dice(torch.from_numpy(pred[class_idx]), torch.from_numpy(lbl[class_idx]), spacing_mm, 1)  # using retrieved spacing here

        if np.sum(lbl[class_idx]) != 0:

            dice_case_result[class_name] = round(dice, 3)
            tmp=organ_dice_results[class_name]
            tmp.append(round(dice, 3))
            organ_dice_results.update({class_name:tmp})


            nsd_case_result[class_name] = round(nsd, 3)
            tmp=organ_nsd_results[class_name]
            tmp.append(round(nsd, 3))
            organ_nsd_results.update({class_name:tmp})
        else:
            dice_case_result[class_name] = np.NaN 
            nsd_case_result[class_name] = np.NaN
    
    csv_dsc_writer.writerows([dice_case_result])
    csv_nsd_writer.writerows([nsd_case_result])
    csv_dsc.close()
    csv_nsd.close()

    
def metrics_engine(dataset_path, out_path, pred_path_list, num_workers):

    organ_dice_results = Manager().dict()
    organ_nsd_results = Manager().dict()
    for i in class_map_w1k.values():
        organ_dice_results[i] = []
        organ_nsd_results[i] = []

    csv_dsc = open(os.path.join(out_path, 'dice_results.csv'), 'a')
    fieldnames = ["name"] + list(class_map_w1k.values())
    csv_dsc_writer = csv.DictWriter(csv_dsc, fieldnames=fieldnames)
    csv_dsc_writer.writeheader()
    csv_dsc.close()

    csv_nsd = open(os.path.join(out_path, 'nsd_results.csv'), 'a')
    csv_nsd_writer = csv.DictWriter(csv_nsd, fieldnames=fieldnames)
    csv_nsd_writer.writeheader()
    csv_nsd.close()
    
    pool = Pool(processes=num_workers)
    for index, pred_path in enumerate(pred_path_list):
        pool.apply_async(metrics_computer, (dataset_path, pred_path, out_path,organ_dice_results, organ_nsd_results))
    pool.close()
    pool.join()

    avg_dsc = {"name": "avg"}
    avg_nsd = {"name": "avg"}
    wavg_dsc = {"name": "weighted avg"}
    wavg_nsd = {"name": "weighted avg"}
    for i in organ_dice_results.keys():
        avg_dsc.update({i:round(np.array(organ_dice_results[i]).mean(), 3) }) 
        avg_nsd.update({i:round(np.array(organ_nsd_results[i]).mean(), 3)})
        wavg_dsc.update({i:round(np.array(organ_dice_results[i]).mean()*default_weight_dict[i], 3)})
        wavg_nsd.update({i:round(np.array(organ_nsd_results[i]).mean()*default_weight_dict[i], 3)})

    csv_dsc = open(os.path.join(out_path, 'dice_results.csv'), 'a')
    fieldnames = ["name"] + list(class_map_w1k.values())
    csv_dsc_writer = csv.DictWriter(csv_dsc, fieldnames=fieldnames)

    csv_nsd = open(os.path.join(out_path, 'nsd_results.csv'), 'a')
    csv_nsd_writer = csv.DictWriter(csv_nsd, fieldnames=fieldnames)
    
    csv_dsc_writer.writerows([avg_dsc])
    csv_nsd_writer.writerows([avg_nsd])
    csv_dsc_writer.writerows([wavg_dsc])
    csv_nsd_writer.writerows([wavg_nsd])
    csv_dsc.close()
    csv_nsd.close()
    
    
    # calculate weighted mDSC & weighted mNSD
    wavg_dsc_value = [wavg_dsc[i] for i in wavg_dsc.keys() if i != "name" and not np.isnan(wavg_dsc[i])]
    wavg_nsd_value = [wavg_nsd[i] for i in wavg_nsd.keys() if i != "name" and not np.isnan(wavg_nsd[i])]
    wmean_dsc = round(np.array(wavg_dsc_value).sum(), 3) 
    wmean_nsd = round(np.array(wavg_nsd_value).sum(), 3)  
    
    # calculate mDSC & weighted mNSD
    avg_dsc_value = [avg_dsc[i] for i in avg_dsc.keys() if i != "name" and not np.isnan(avg_dsc[i])]
    avg_nsd_value = [avg_nsd[i] for i in avg_nsd.keys() if i != "name" and not np.isnan(avg_nsd[i])]
    mean_dsc = round(np.array(avg_dsc_value).mean(), 3) 
    mean_nsd = round(np.array(avg_nsd_value).mean(), 3)  

    with open(os.path.join(out_path, 'scores.txt'),'a+') as score_file:
        score_file.writelines("wmDSC wmNSD")
        score_file.writelines("\n")
        score_file.writelines(" ".join([str(wmean_dsc), str(wmean_nsd)]))
        score_file.writelines("\n")
        score_file.writelines("mDSC mNSD")
        score_file.writelines("\n")
        score_file.writelines(" ".join([str(mean_dsc), str(mean_nsd)]))
        
    return wmean_dsc, wmean_nsd, mean_dsc, mean_nsd



def cal_metric(out_path, pred_path, dataset_path, num_workers):
    pred_path_list = glob.glob(pred_path+'/*.nii.gz')

    mean_wdsc, mean_wnds, mean_dsc, mean_nsd = metrics_engine(dataset_path, out_path, pred_path_list, num_workers)
    
    return mean_wdsc, mean_wnds, mean_dsc, mean_nsd

    

