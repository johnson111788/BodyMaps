FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

WORKDIR /workspace
COPY ./   /workspace

RUN pip install pip -U

RUN pip install tqdm
RUN pip install nibabel
RUN pip install monai[all]==0.9.0
RUN pip install connected-components-3d
RUN pip install fastremap

