import os, sys
import numpy as np
sys.path.append("..") 

from monai.transforms import (
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)
from monai.data import DataLoader, Dataset, list_data_collate
from monai.config import KeysCollection
from monai.transforms import LoadImaged
from monai.transforms.transform import MapTransform


class_map_25organ_w1k = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    6: "liver",
    7: "stomach",
    8: "aorta",
    9: "inferior_vena_cava",
    11: "pancreas",
    12: "adrenal_gland_right",
    13: "adrenal_gland_left",
    14: "duodenum",
    18: "colon",
    19: "intestine",
    25: "celiac_trunk",
}

taskmap_set = {
    'w1k':class_map_25organ_w1k,
}


def path_loader_w1k(args):
    test_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear"),
            ), 
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            ToTensord(keys=["image"]),
        ]
    )
    
    
    ## test dict part
    all_test_imgs = os.listdir(args.dataset_path)
    pred_imgs = os.listdir(args.out_path)
    test_imgs = [img for img in all_test_imgs if img not in pred_imgs]
    data_dicts_test = [{'image': os.path.join(args.dataset_path, test_img), "name": test_img.split('.')[0]} for test_img in test_imgs]
    print('data_dicts_test',len(data_dicts_test))
    test_dataset = Dataset(data=data_dicts_test, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)

    return test_loader, test_transforms