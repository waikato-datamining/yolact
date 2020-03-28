# yolact (2020-02-11)

Uses [YOLACT](https://github.com/dbolya/yolact/) for object detection (masks).

## Version

YOLACT github repo hash (commit 322):

```
f54b0a5b17a7c547e92c4d7026be6542f43862e7
```

Timestamp:

```
2020-02-11
```

## Docker

### Build local image

* Build image yolact from Docker file (from within `yolact-2020-02-11/base`):

  ```
  docker build -t yolact .
  ```

* Run image `yolact` in interactive mode (i.e., using `bash`) as container `yolact_container`:

  ```
  docker run --runtime=nvidia --name yolact_container -ti -v \
    /path_to/local_disk/containing_data:/path_to/mount/inside/docker_container \
    yolact bash
  ```

### Pre-built images

* Build

  ```
  docker build -t yolact/yolact:2020-02-11 .
  ```

* Tag

  ```
  docker tag \
    yolact/yolact:2020-02-11 \
    public-push.aml-repo.cms.waikato.ac.nz:443/yolact/yolact:2020-02-11
  ```

* Push

  ```
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/yolact/yolact:2020-02-11
  ```

  If error `no basic auth credentials` occurs, then run (enter user/password when prompted):

  ```
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```

* Pull

  If image is available in aml-repo and you just want to use it, you can pull using following 
  command and then [run](#run).

  ```
  docker pull public.aml-repo.cms.waikato.ac.nz:443/yolact/yolact:2020-02-11
  ```

  If error `no basic auth credentials` occurs, then run (enter user/password when prompted):

  ```
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```

  Then tag by running:

  ```
  docker tag \
    public.aml-repo.cms.waikato.ac.nz:443/yolact/yolact:2020-02-11 \
    yolact/yolact:2020-02-11
  ```

* <a name="run">Run</a>

  ```
  docker run --runtime=nvidia --shm-size 8G -v /local:/container -it yolact/yolact:2020-02-11
  ```

  `/local:/container` maps a local disk directory into a directory inside the container.
  Typically, you would map the `weights` (pre-trained models) and the data (annotations, 
  log, etc):

  ```
  -v /some/where/dataset01:/data -v /some/where/yolact/pretrained:/yolact/weights
  ```


## Usage

* Instead of modifying the `/yolact/data/config.py` file itself, an external
  Python module can be loaded in via the `YOLACT_CONFIG` environment variable.
  This module must have the following two variables defined:

    * `external_dataset`
    * `external_config`

  These will get mapped by the modified `config.py` in the image to 
  `external_dataset` and `external_config`.

  See [external_config_example.py](external_config_example.py) for an
  example module.

  For example, if your configuration will be available in the docker container
  as `/data/config/model-01.py` then you export the following environment variable:

  ```
  export YOLACT_CONFIG=/data/config/model-01.py
  ```

* Train

  ```
  yolact_train --config=external_config --log_folder=/data/log \
    --validation_epoch 100
  ```

* Evaluate

  ```
  yolact_eval --config=external_config --trained_model=weights/MODELNAME.pth \
    --score_threshold=0.15 --top_k=200 \
    --images=/predictions/in/:/predictions/out/
  ```

* Predict

  ```
  yolact_predict --config=external_config --trained_model=weights/MODELNAME.pth \
    --score_threshold=0.15 --top_k=200 \
    --output_polygons --output_minrect \
    --prediction_in /predictions/in/ --prediction_out /predictions/out/    
  ```

## Additional configurations

* You can use `YOLACT_CONFIG2` and `YOLACT_CONFIG3` to supply two more
  configurations.
  
* Use `--config=external_config2` and `--config=external_config3` respectively
  when referring to them.

* Within the configuration file itself, by sure to reference the dataset correctly
  via `'dataset': external_dataset2,` or `'dataset': external_dataset2,`
