import json,csv, glob, matplotlib, os
import numpy as np, matplotlib.pyplot as plt

join = os.path.join
matplotlib.use("Agg")

from utils.logger import logger


def gpu_auc(team_outpath, csv_path, time_interval):
    
    jsonl = sorted(glob.glob(team_outpath + "/*.json"))
    alldata = []
    all_time = []
    all_MaxGPU = []
    all_AUC = []
    for item in jsonl:
        csv_l = []
        name = item.split(os.sep)[-1].split(".")[0]
        csv_l.append(name + '.nii.gz')
        zitem = item
        with open(zitem) as f:
            try:
                js = json.load(f)
            except Exception as error:
                logger.error(f"{item} have error")
                logger.exception(error)
            if "time" not in js:
                logger.error(f"{item} don't have time!!!!")
                logger.info(f"Manually compute {item}")
                time = time_interval * len(js["gpu_memory"])
            else:
                time = js["time"]
            csv_l.append(np.round(time,2))

            mem = js["gpu_memory"]
            x = [item * time_interval for item in range(len(mem))]
            plt.cla()
            plt.xlabel("Time (s)", fontsize="large")
            plt.ylabel("GPU Memory (MB)", fontsize="large")
            plt.plot(x, mem, "b", ms=10, label="a")
            plt.savefig(zitem.replace(".json", "_GPU-Time.png"), dpi=300)
            count_set = set(mem)

            max_mem = max(count_set)
            auc = np.round(sum(mem) * time_interval)
            csv_l.append(np.round(max_mem))
            csv_l.append(np.round(auc))

            all_time.append(np.round(time,2))
            all_MaxGPU.append(np.round(max_mem))
            all_AUC.append(np.round(sum(mem) * time_interval))

        alldata.append(csv_l)

    f = open(csv_path, "w",newline='')
    writer = csv.writer(f)
    writer.writerow(["Name", "Time", "MaxGPU_Mem", "AUC_GPU_Time"])
    for i in alldata:
        writer.writerow(i)

    mean_time = np.array(all_time).mean()
    mean_MaxGPU = np.array(all_MaxGPU).mean()
    mean_AUC = np.array(all_AUC).mean()
    writer.writerow(["avg", str(mean_time), str(mean_MaxGPU), str(mean_AUC)])
    f.close()

    return mean_time, mean_AUC
