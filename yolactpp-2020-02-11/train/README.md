# Training yolact++ (2020-02-11)

Uses [YOLACT++](https://github.com/dbolya/yolact/) for object detection (masks).

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

* Build image `yolactpp_train` from Docker file (from within `yolactpp-2020-02-11/train`):

  ```
  docker build -t yolactpp_train .
  ```

* Run image `yolactpp_train` in interactive mode (i.e., using `bash`):

  ```
  docker run --runtime=nvidia --shm-size 8G -ti \
    -v /path_to/local_disk/containing_data:/path_to/mount/inside/docker_container \
    -e YOLACTPP_CONFIG=/data/config/model-01.py \
    yolactpp_train bash
  ```

### Pre-built images

* Build

  ```
  docker build -t yolact/yolactpp:2020-02-11_train .
  ```

* Tag

  ```
  docker tag \
    yolact/yolactpp:2020-02-11_train \
    public-push.aml-repo.cms.waikato.ac.nz:443/yolact/yolactpp:2020-02-11_train
  ```

* Push

  ```
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/yolact/yolactpp:2020-02-11_train
  ```

  If error `no basic auth credentials` occurs, then run (enter user/password when prompted):

  ```
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```

* Pull

  If image is available in aml-repo and you just want to use it, you can pull using following 
  command and then [run](#run).

  ```
  docker pull public.aml-repo.cms.waikato.ac.nz:443/yolact/yolactpp:2020-02-11_train
  ```

  If error `no basic auth credentials` occurs, then run (enter user/password when prompted):

  ```
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```

  Then tag by running:

  ```
  docker tag \
    public.aml-repo.cms.waikato.ac.nz:443/yolact/yolactpp:2020-02-11_train \
    yolact/yolactpp:2020-02-11_train
  ```

* <a name="run">Run</a>

  ```
  docker run --runtime=nvidia --shm-size 8G \
    -v /local:/container \
    -e YOLACTPP_CONFIG=/data/config/model-01.py \
    -it yolact/yolactpp:2020-02-11_train \
    --config=external_config --log_folder=/data/log \
    --validation_epoch 100    
  ```

  `/local:/container` maps a local disk directory into a directory inside the container.
  Typically, you would map the `weights` (pre-trained models) and the data (annotations, 
  log, etc):

  ```
  -v /some/where/dataset01:/data -v /some/where/yolactpp/pretrained:/yolactpp/weights
  ```