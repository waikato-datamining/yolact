ARG PYTORCH="1.2"
ARG CUDA="10.0"
ARG CUDNN="7"

ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-get update && \
    apt-get install -y --no-install-recommends libglib2.0-0 libsm6 libxrender-dev libxext6 && \
    git clone https://github.com/dbolya/yolact.git /yolact && \
    cd /yolact && \
    git reset --hard f54b0a5b17a7c547e92c4d7026be6542f43862e7 && \
    pip install Cython && \
    pip install opencv-python "pillow<7.0.0" pycocotools matplotlib torchvision==0.4.0 && \
    rm -Rf /root/.cache/pip && \
    rm -rf /var/lib/apt/lists/*

COPY config.py /yolact/data
COPY yolact_train /usr/bin
COPY yolact_eval /usr/bin
ENV PYTHONPATH=/yolact
