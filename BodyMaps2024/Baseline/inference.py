from tqdm import tqdm
import os, warnings, argparse, torch, numpy as np, nibabel as nib
warnings.filterwarnings("ignore")

from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet

from dataset.dataloader import path_loader_w1k, taskmap_set
from model.Universal_model import Universal_model
from utils.utils import TEMPLATE, organ_post_process, invert_transform


def validation(model, ValLoader, test_transforms, args):
    model.eval()
    selected_class_map = taskmap_set[args.map_type]
    
    output_directory = args.out_path
    os.makedirs(output_directory, exist_ok=True)
    
    for index, batch in enumerate(tqdm(ValLoader)):
        image, name = batch["image"].to(args.device), batch["name"]

        name = name.item() if isinstance(name, torch.Tensor) else name  # Convert to Python str if it's a Tensor
        original_affine = batch["image_meta_dict"]["affine"][0].numpy()
        
        with torch.no_grad():
            val_outputs = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=args.overlap, mode='gaussian')
            val_outputs = val_outputs.sigmoid()
            hard_val_outputs = (val_outputs>0.5).float()
        
        organ_list_all = TEMPLATE['all'] # post processing all organ
        pred_hard_post, _ = organ_post_process(hard_val_outputs.cpu().numpy(), organ_list_all,args)
        pred_hard_post = torch.tensor(pred_hard_post)
        
        batch["pred"] = pred_hard_post
        batch = invert_transform('pred', batch, test_transforms)
        pred = batch[0]['pred']

        organ_seg_save_path = output_directory
        final_pred = np.zeros(pred.shape[1:])
        for index, i in enumerate(selected_class_map.keys()):
            final_pred[pred[i-1]==1] = index+1
        
        nib.save(
                nib.Nifti1Image(final_pred.astype(np.uint8), original_affine), os.path.join(organ_seg_save_path, f'{name[0]}.nii.gz')
        )


def process(args):
    rank = 0
    args.device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(args.device)

    # prepare the 3D model
    if args.model_backbone == 'segresnet':
        model = SegResNet(
                    blocks_down=[1, 2, 2, 4],
                    blocks_up=[1, 1, 1],
                    init_filters=16,
                    in_channels=1,
                    out_channels=32,
                    dropout_prob=0.0,
                    )
    else:
        model = Universal_model(img_size=(args.roi_x, args.roi_y, args.roi_z),
                        in_channels=1,
                        out_channels=32,
                        backbone=args.model_backbone,
                        encoding='word_embedding'
                        )

    model.to(args.device)

    #Load pre-trained weights
    store_dict = model.state_dict()
    checkpoint = torch.load(args.pretrain)
    load_dict = checkpoint['net']

    for key, value in load_dict.items():
        name = '.'.join(key.split('.')[1:])
        store_dict[name] = value

    model.load_state_dict(store_dict)
    print('Use pretrained weights')
    model.cuda()
    torch.backends.cudnn.benchmark = True

    test_loader, test_transforms = path_loader_w1k(args)

    validation(model, test_loader, test_transforms, args)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device")

    parser.add_argument('--dataset_path', default='PATH_TO_IMGS')
    parser.add_argument('--out_path', default='PATH_TO_PREDICTION')
    parser.add_argument('--pretrain', default='PATH_TO_TRAINED_WEIGHT')

    parser.add_argument('--map_type', default='w1k')
    parser.add_argument('--model_backbone', default='unet', choices=['unet', 'swinunetr', 'dints', 'segresnet'])

    parser.add_argument('--overlap', default=0.5, type=float, help='overlap for sliding_window_inference')
    parser.add_argument('--num_workers', default=8, type=int, help='workers numebr for DataLoader')
    parser.add_argument('--a_min', default=-175, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')

    args = parser.parse_args()
    
    process(args=args)

if __name__ == "__main__":
    main()



    
