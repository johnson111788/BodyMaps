# Segmentation Efficiency Evaluation


## Usage

### 1. Setup models
```shell
git clone https://github.com/johnson111788/BodyMaps.git
cd BodyMaps2024/Baseline
conda create -n bodymaps24 && conda activate bodymaps24
conda install python=3.9
pip install tqdm
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install nibabel
pip install monai[all]==0.9.0
pip install connected-components-3d
pip install fastremap
pip install pynvml
pip install loguru
```


### 2. Install Docker and Singularity

Please refer to [Docker](https://docs.docker.com/engine/install/) and [Singularity](https://sylabs.io/guides/3.7/user-guide/quick_start.html#quick-installation-steps) for installation.


### 3. Evaluation


Set `docker`, `docker_path`, `data_path`, and `save_path` in `eval.py` and run

```shell
python eval.py
```

## Q&A

Q: How the `Area under GPU memory-time curve` and `Area under CPU utilization-time curve` is computed?

> A: We record the GPU memory and GPU utilization every 0.1s. The `Area under GPU memory-time curve` and `Area under CPU utilization-time curve` are the cumulative values along running time.

## Info

The class-index mapping in W-1K is as follows:

| Class | Index | Class | Index |
|----------|----------|----------|----------|
|   Spleen  |   1  |   Pancreas  |  9  |
|   Right kidney  |   2  |   Right adrenal gland  |  10  |
|   Left kidney  |   3  |   Left adrenal gland  |  11  |
|   Gallbladder  |   4  |   Duodenum  |  12  |
|   Liver  |   5  |   Colon  |    13  |
|   Stomach  |   6  |   Intestine  |   14  |
|   Aorta  |   7  |   Celiac Trunk  |   15  |
|   Inferior Vena Cava  |   8  |     |     |
