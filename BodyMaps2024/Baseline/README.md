# BodyMaps24 Baseline

The official baseline model for [Body Maps: Towards 3D Atlas of Human Body](https://codalab.lisn.upsaclay.fr/competitions/16919), derived from [CLIP-Driven Universal Model](https://github.com/ljwztc/CLIP-Driven-Universal-Model).

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
```

### 2. Inference

```shell
python inference.py --pretrain "PATH_TO_WEIGHT" --model_backbone unet --out_path /workspace/outputs/ --dataset_path /workspace/inputs/
```

For more usage, please refer to [the repositry of Universal-Model](https://github.com/ljwztc/CLIP-Driven-Universal-Model).


### 3. Evaluation

Please refer to [BodyMaps24 Evaluation Code](https://github.com/johnson111788/BodyMaps/tree/master/BodyMaps2024/Evaluation).

### 4. Build a Docker Image

Build a docker image of the model by

```shell
docker build -t teamname .
```

The configuration of the docker image is in `Dokcerfile`. It will call `predict.sh` when starting the docker image.

NOTE: Please create empty folders named `outputs` and `intputs` in the same directory of `Dockerfile` before building the docker image.


### 5. Run and Save the Docker Image

To debug the docker image, you can run the docker image by

```shell
docker container run --gpus "device=0" --shm-size 64G --name teamname --rm -v $PWD/BodyMaps2024_Test/:/workspace/inputs/ -v $PWD/teamname_outputs/:/workspace/outputs/ teamname:latest /bin/bash -c "sh predict.sh"
```

If there's no error, you can save the docker image by

```shell
docker save teamname > teamname.tar
```
and send the `teamname.tar` to the organizer.

We will run the docker image by singularity owing to the safety issue:

```shell
singularity build teamname.sif docker-archive://teamname.tar
singularity exec --nv -B $PWD/BodyMaps2024_Test/img/:/workspace/inputs/ -B $PWD/teamname_outputs/:/workspace/outputs/ teamname.sif bash /workspace/predict.sh
```