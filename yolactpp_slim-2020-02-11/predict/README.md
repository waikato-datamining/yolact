# Inference with yolact++ slim (2020-02-11)

Uses [YOLACT++](https://github.com/dbolya/yolact/) for object detection (masks).
Uses Python3.7 instead of Anaconda for the base image, hence *slim*.

## Version

YOLACT++ github repo hash (commit 322):

```
f54b0a5b17a7c547e92c4d7026be6542f43862e7
```

Timestamp:

```
2020-02-11
```

## Docker

### Build local image

* Build image `yolactpp_slim_predict` from Docker file (from within `yolactpp_slim-2020-02-11/predict`):

  ```
  docker build -t yolactpp_slim_predict .
  ```

* Run image `yolactpp_slim_predict` in interactive mode (i.e., using `bash`):

  ```
  docker run --runtime=nvidia --shm-size 8G -ti \
    -v /path_to/local_disk/containing_data:/path_to/mount/inside/docker_container \
    -e YOLACTPP_CONFIG=/data/config/model-01.py \
    yolactpp_slim_predict bash
  ```

  `/local:/container` maps a local disk directory into a directory inside the container


### Pre-built images

* Build

  ```
  docker build -t yolact/yolactpp_slim_predict:2020-02-11 .
  ```

* Tag

  ```
  docker tag \
    yolact/yolactpp_slim_predict:2020-02-11 \
    public-push.aml-repo.cms.waikato.ac.nz:443/yolact/yolactpp_slim_predict:2020-02-11
  ```

* Push

  ```
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/yolact/yolactpp_slim_predict:2020-02-11
  ```

  If error `no basic auth credentials` occurs, then run (enter user/password when prompted):

  ```
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```

* Pull

  If image is available in aml-repo and you just want to use it, you can pull using following 
  command and then [run](#run).

  ```
  docker pull public.aml-repo.cms.waikato.ac.nz:443/yolact/yolactpp_slim_predict:2020-02-11
  ```

  If error `no basic auth credentials` occurs, then run (enter user/password when prompted):

  ```
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```

  Then tag by running:

  ```
  docker tag \
    public.aml-repo.cms.waikato.ac.nz:443/yolact/yolactpp_slim_predict:2020-02-11 \
    yolact/yolactpp_slim_predict:2020-02-11
  ```

* <a name="run">Run</a>

  ```
  docker run --runtime=nvidia --shm-size 8G \
    -v /local:/container \
    -e YOLACTPP_CONFIG=/data/config/model-01.py \
    -it yolact/yolactpp_slim_predict:2020-02-11 \
    --config=external_config --trained_model=weights/MODELNAME.pth \
    --score_threshold=0.15 --top_k=200 \
    --output_polygons --output_minrect \
    --prediction_in /predictions/in/ --prediction_out /predictions/out/    
  ```

  `/local:/container` maps a local disk directory into a directory inside the container.
  Typically, you would map the `weights` (pre-trained models) and the data (annotations, 
  log, etc):

  ```
  -v /some/where/dataset01:/data -v /some/where/yolactpp/pretrained:/yolactpp/weights
  ```
