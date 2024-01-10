import json, os, time
from multiprocessing import Manager, Process
from pynvml.smi import nvidia_smi

from utils.logger import logger


def daemon_process(gpu_list, time_interval, gpu_index=1):
    while True:
        nvsmi = nvidia_smi.getInstance()
        dictm = nvsmi.DeviceQuery("memory.free, memory.total")
        gpu_memory = (
            dictm["gpu"][gpu_index]["fb_memory_usage"]["total"] - dictm["gpu"][gpu_index]["fb_memory_usage"]["free"]
        )
        gpu_list.append(gpu_memory)
        time.sleep(time_interval)


def save_result(T, json_path, gpu_list_):
    if os.path.exists(json_path):
        with open(json_path) as f:
            js = json.load(f)
    else:
        js = {"gpu_memory": []}

    with open(json_path, "w") as f:
        js["gpu_memory"] = gpu_list_
        
        json.dump(js, f, indent=4)

    with open(json_path, "r") as f:
        js = json.load(f)
    with open(json_path, "w") as f:
        js["time"] = T
        json.dump(js, f, indent=4)
    time.sleep(2)


def efficiency(docker_path, docker_name, json_path, time_interval, gpu, sleep_time):
    
    try:
        manager = Manager()
        gpu_list = manager.list()

        p1 = Process(target=daemon_process, args=(gpu_list, time_interval, gpu,))
        p1.daemon = True
        p1.start()
        start_time = time.time()
        os.system('SINGULARITYENV_CUDA_VISIBLE_DEVICES={} singularity exec --nv -B $PWD/inputs/:/workspace/inputs/ -B $PWD/outputs/:/workspace/outputs/ {}{}.sif bash /workspace/predict.sh'.format(gpu, docker_path, docker_name))
        end_time = time.time()

        T = end_time - start_time
        gpu_list = list(gpu_list)
        gpu_list_copy = gpu_list.copy()
        
        save_result(T, json_path, gpu_list_copy)
        time.sleep(sleep_time)
        p1.terminate()
        p1.join()
        return T
        
    except Exception as error:
        logger.exception(error)