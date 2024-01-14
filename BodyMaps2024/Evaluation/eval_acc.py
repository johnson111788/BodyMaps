import os, shutil, argparse, torch
from tqdm import tqdm

from utils.efficiency import efficiency
from utils.gpu_auc import gpu_auc
from utils.cal_metric import cal_metric

from utils.utils import get_testcase, check_dir
from utils.logger import add_file_handler_to_logger, logger


join = os.path.join


def eval(args):
    try:
        
        team_outpath = join(args.save_path, args.name)        
        os.makedirs(team_outpath, exist_ok=True)
        mean_wdsc, mean_wnds, mean_dsc, mean_nsd = cal_metric(out_path=team_outpath, pred_path=join(team_outpath, './outputs/'), dataset_path=args.data_path, num_workers=10)
        
        with open(os.path.join(team_outpath, 'final_scores.txt'),'a') as score_file:
            score_file.writelines("docker_name, wmDSC, wmNSD")
            score_file.writelines("\n")
            score_file.writelines(",".join([args.name, str(mean_dsc), str(mean_nsd)]))

    except Exception as e:
        logger.exception(e)
        
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bodymaps'
                        , help='name')
    parser.add_argument('--data_path', default='PATH_TO_DATA'
                        , help='all test cases')
    parser.add_argument('--save_path', default='PATH_TO_SAVE'
                        , help='evaluation results will be saved in this folder')

    args = parser.parse_args()
    
    add_file_handler_to_logger(name=args.name+"_ACC", dir_path="logs/", level="DEBUG")
    
    eval(args)