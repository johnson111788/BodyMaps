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
        logger.info('Loading docker:', args.docker)
        os.system('singularity build {} docker-archive://{} '.format(join(args.docker_path, args.docker.replace('.tar','.sif')), join(args.docker_path, args.docker)))
        
        name = args.docker.split('.')[0]
        team_outpath = join(args.save_path, name)
        os.makedirs(team_outpath, exist_ok=True)
        
        time_flag=0
        
        logger.info(f"Evaluating {name}")
        for test_case_infor in tqdm(test_cases_infor):
            case = test_case_infor['name']
            
            if not check_dir(args.temp_in):
                logger.error("please check inputs folder", args.temp_in)
                raise
            shutil.copy(join(args.data_path, 'img', case), args.temp_in)
            
            logger.info(f"{case} start predict...")

            json_path = join(team_outpath, case + ".json")
            runtime = efficiency(args.docker_path, name, json_path, args.time_interval, 0, args.sleep_time)
            logger.info(f"{case} finished!")

            os.remove(join(args.temp_in, case))
            
            if runtime > float(test_case_infor['spacing'])*float(test_case_infor['z_size'])-args.time_thresh:
                time_flag+=1

            if time_flag > len(test_cases_infor) * 0.2:
                raise RuntimeError(f'20% of the case runtime exceeds upper limit for {args.docker}, so the program is interrupted')
        
        csv_path = join(team_outpath, 'efficiency.csv')
        mean_runtime, mean_AUC = gpu_auc(team_outpath, csv_path, args.time_interval)

        shutil.move(args.temp_out, team_outpath)
        os.mkdir(args.temp_out)
        
        torch.cuda.empty_cache()
        shutil.rmtree(args.temp_in)
        os.mkdir(args.temp_in)
        
        mean_wdsc, mean_wnds, mean_dsc, mean_nsd = cal_metric(out_path=team_outpath, pred_path=join(team_outpath, args.temp_out), dataset_path=args.data_path, num_workers=8)
        
        final_score = (mean_dsc*mean_nsd)/(mean_runtime*mean_AUC)
        with open(os.path.join(team_outpath, 'final_scores.txt'),'a') as score_file:
            score_file.writelines("docker_name, wmDSC, wmNSD, Runtime, AUC")
            score_file.writelines("\n")
            score_file.writelines(",".join([name, str(mean_dsc), str(mean_nsd), str(mean_runtime), str(mean_AUC)]))

    except Exception as e:
        logger.exception(e)
        
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--docker', default='bodymaps.tar'
                        , help='docker name')
    parser.add_argument('--docker_path', default='PATH_TO_DOCKER'
                        , help='put docker in this folder')
    parser.add_argument('--data_path', default='PATH_TO_DATA'
                        , help='all test cases')
    parser.add_argument('--save_path', default='PATH_TO_SAVE'
                        , help='evaluation results will be saved in this folder')

    parser.add_argument('--temp_in', default='./inputs/'
                        , help='temporarily save test cases in this folder to for loop evaluation')
    parser.add_argument('--temp_out', default='./outputs/',
                        help='temporarily save predicted cases in this folder to for loop evaluation')

    parser.add_argument('--V_tlimit', default='./utils/VENOUS_time_limit.csv'
                        , help='temporarily save test cases in this folder to for loop evaluation')
    parser.add_argument('--A_tlimit', default='./utils/ARTERIAL_time_limit.csv',
                        help='temporarily save predicted cases in this folder to for loop evaluation')

    parser.add_argument("-time_interval", default=0.1, help="time_interval")
    parser.add_argument("-sleep_time", default=5, help="sleep time")
    
    parser.add_argument('--time_thresh', default=50)
    args = parser.parse_args()
    
    add_file_handler_to_logger(name=args.docker, dir_path="logs/", level="DEBUG")
    
    test_cases_infor = get_testcase(args.V_tlimit, args.A_tlimit)

    os.makedirs(args.temp_in, exist_ok=True)
    os.makedirs(args.temp_out, exist_ok=True)
    os.system("chmod -R 777 outputs/")

    eval(args)