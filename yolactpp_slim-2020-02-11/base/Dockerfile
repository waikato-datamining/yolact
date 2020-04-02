ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt-get update && \
    apt-get install -y software-properties-common wget && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y python3.7 python3.7-dev && \
    ln -s /usr/bin/python3.7 /usr/bin/python && \
    apt-get install -y --no-install-recommends libglib2.0-0 libsm6 libxrender-dev libxext6 && \
    apt-get install -y build-essential git && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py && \
    python -m pip install Cython && \
    python -m pip install numpy && \
    python -m pip install opencv-python==4.1.1.26 "pillow<7.0.0" pycocotools matplotlib torchvision==0.4.0 torch==1.4.0 python_image_complete wai.annotations planar && \
    git clone https://github.com/dbolya/yolact.git /yolactpp && \
    cd /yolactpp && \
    git reset --hard f54b0a5b17a7c547e92c4d7026be6542f43862e7 && \
    rm -Rf /root/.cache/pip && \
    rm -rf /var/lib/apt/lists/*

# copy version of setup without torch check whether cuda is present
# (fails at build time, due to graphics card not getting passed through)
COPY setup.py /yolactpp/external/DCNv2/
RUN cd /yolactpp/external/DCNv2 && \
    python setup.py build develop && \
    rm -Rf /root/.cache/pip

COPY config.py /yolactpp/data
COPY predict.py /yolactpp
COPY yolactpp_* /usr/bin/
ENV PYTHONPATH=/yolactpp
