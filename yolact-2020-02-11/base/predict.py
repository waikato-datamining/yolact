import argparse
import cv2
from datetime import datetime
import os
import time
import torch
import torch.backends.cudnn as cudnn
import traceback
from image_complete import auto
from yolact import Yolact
from data import cfg, set_cfg
from layers.output_utils import postprocess
from utils import timer
from utils.augmentations import FastBaseTransform
from wai.annotations.core import ImageInfo
from wai.annotations.roi import ROIObject
from wai.annotations.roi.io import ROIWriter
from wai.annotations.image_utils import mask_to_polygon, polygon_to_minrect, polygon_to_lists

SUPPORTED_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]
""" supported file extensions (lower case). """

MAX_INCOMPLETE = 3
""" the maximum number of times an image can return 'incomplete' status before getting moved/deleted. """


def predictions_to_rois(dets_out, width, height, top_k, score_threshold,
                        output_polygons, mask_threshold, mask_nth, output_minrect,
                        view_margin, fully_connected):
    """
    Turns the predictions into ROI objects
    :param dets_out: the predictions
    :param width: the width of the image
    :type width: int
    :param height: the height of the image
    :type height: int
    :param top_k: the maximum number of top predictions to use
    :type top_k: int
    :param score_threshold: the minimum score predictions have to have
    :type score_threshold: float
    :param output_polygons: whether the model predicts masks and polygons should be stored in the CSV files
    :type output_polygons: bool
    :param mask_threshold: the threshold to use for determining the contour of a mask
    :type mask_threshold: float
    :param mask_nth: to speed up polygon computation, use only every nth row and column from mask
    :type mask_nth: int
    :param output_minrect: when predicting polygons, whether to output the minimal rectangles around the objects as well
    :type output_minrect: bool
    :param view_margin: the margin in pixels to use around the masks
    :type view_margin: int
    :param fully_connected: whether regions of 'high' or 'low' values should be fully-connected at isthmuses
    :type fully_connected: str
    :return: the list of ROIObjects
    :rtype: list
    """

    result = []

    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(dets_out, width, height, crop_masks=False, score_threshold=score_threshold)
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:top_k]
        if output_polygons:
            classes, scores, boxes, masks = [x[idx].cpu().numpy() for x in t]
        else:
            classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < score_threshold:
            num_dets_to_consider = j
            break

    # the class labels
    if isinstance(cfg.dataset.class_names, list):
        class_labels = cfg.dataset.class_names
    else:
        class_labels = [cfg.dataset.class_names]

    if num_dets_to_consider > 0:
        # After this, mask is of size [num_dets, h, w, 1]
        if output_polygons:
            masks = masks[:num_dets_to_consider, :, :, None]
        for j in range(num_dets_to_consider):
            x0, y0, x1, y1 = boxes[j, :]
            x0n = x0 / width
            y0n = y0 / height
            x1n = x1 / width
            y1n = y1 / height
            label = classes[j]
            score = scores[j]
            label_str = class_labels[classes[j]]
            px = None
            py = None
            pxn = None
            pyn = None
            bw = None
            bh = None
            if output_polygons:
                px = []
                py = []
                pxn = []
                pyn = []
                mask = masks[j,:,:][:,:,0]
                poly = mask_to_polygon(mask, mask_threshold=mask_threshold, mask_nth=mask_nth, view=(x0, y0, x1, y1), view_margin=view_margin, fully_connected=fully_connected)
                if len(poly) > 0:
                    px, py = polygon_to_lists(poly[0], swap_x_y=True, normalize=False, as_string=True)
                    pxn, pyn = polygon_to_lists(poly[0], swap_x_y=True, normalize=True, img_width=width, img_height=height, as_string=True)
                    if output_minrect:
                        bw, bh = polygon_to_minrect(poly[0])
            roiobj = ROIObject(x0, y0, x1, y1, x0n, y0n, x1n, y1n, label, label_str, score=score,
                               poly_x=px, poly_y=py, poly_xn=pxn, poly_yn=pyn,
                               minrect_w=bw, minrect_h=bh)
            result.append(roiobj)

    return result


def predict(model, input_dir, output_dir, tmp_dir, top_k, score_threshold, delete_input,
            output_polygons, mask_threshold, mask_nth, output_minrect,
            view_margin, fully_connected):
    """
    Loads the model/config and performs predictions.

    :param model: the model to use
    :type model: Yolact
    :param input_dir: the directory to check for images to use for prediction
    :type input_dir: str
    :param output_dir: the directory to store the results in (predictions and/or images)
    :type output_dir: str
    :param tmp_dir: the directory to store the results initially, before moving them into the actual output directory
    :type tmp_dir: str
    :param top_k: the top X predictions (= objects) to parse
    :type top_k: int
    :param score_threshold: the score threshold to use
    :type score_threshold: float
    :param delete_input: whether to delete the images from the input directory rather than moving them to the output directory
    :type delete_input: bool
    :param fast_nms: use a faster, but not entirely correct version of NMS
    :type fast_nms: bool
    :param cross_class_nms: compute NMS cross-class or per-class
    :type cross_class_nms: bool
    :param output_polygons: whether the model predicts masks and polygons should be stored in the CSV files
    :type output_polygons: bool
    :param mask_threshold: the threshold to use for determining the contour of a mask
    :type mask_threshold: float
    :param mask_nth: to speed up polygon computation, use only every nth row and column from mask
    :type mask_nth: int
    :param output_minrect: when predicting polygons, whether to output the minimal rectangles around the objects as well
    :type output_minrect: bool
    :param view_margin: the margin in pixels to use around the masks
    :type view_margin: int
    :param fully_connected: whether regions of 'high' or 'low' values should be fully-connected at isthmuses
    :type fully_connected: str
    """

    # counter for keeping track of images that cannot be processed
    incomplete_counter = dict()
    num_imgs = 1

    while True:
        start_time = datetime.now()
        im_list = []
        # Loop to pick up images equal to num_imgs or the remaining images if less
        for image_path in os.listdir(input_dir):
            # Load images only
            ext_lower = os.path.splitext(image_path)[1]
            if ext_lower in SUPPORTED_EXTS:
                full_path = os.path.join(input_dir, image_path)
                if auto.is_image_complete(full_path):
                    im_list.append(full_path)
                else:
                    if not full_path in incomplete_counter:
                        incomplete_counter[full_path] = 1
                    else:
                        incomplete_counter[full_path] = incomplete_counter[full_path] + 1

            # remove images that cannot be processed
            remove_from_blacklist = []
            for k in incomplete_counter:
                if incomplete_counter[k] == MAX_INCOMPLETE:
                    print("%s - %s" % (str(datetime.now()), os.path.basename(k)))
                    remove_from_blacklist.append(k)
                    try:
                        if delete_input:
                            print("  flagged as incomplete {} times, deleting\n".format(MAX_INCOMPLETE))
                            os.remove(k)
                        else:
                            print("  flagged as incomplete {} times, skipping\n".format(MAX_INCOMPLETE))
                            os.rename(k, os.path.join(output_dir, os.path.basename(k)))
                    except:
                        print(traceback.format_exc())

            for k in remove_from_blacklist:
                del incomplete_counter[k]

            if len(im_list) == num_imgs:
                break

        if len(im_list) == 0:
            time.sleep(1)
            break
        else:
            print("%s - %s" % (str(datetime.now()), ", ".join(os.path.basename(x) for x in im_list)))

        try:
            for i in range(len(im_list)):
                roi_path = "{}/{}-rois.csv".format(output_dir, os.path.splitext(os.path.basename(im_list[i]))[0])
                if tmp_dir is not None:
                    roi_path_tmp = "{}/{}-rois.tmp".format(tmp_dir, os.path.splitext(os.path.basename(im_list[i]))[0])
                else:
                    roi_path_tmp = "{}/{}-rois.tmp".format(output_dir, os.path.splitext(os.path.basename(im_list[i]))[0])

                img = cv2.imread(im_list[i])
                height, width, _ = img.shape
                frame = torch.from_numpy(img).cuda().float()
                batch = FastBaseTransform()(frame.unsqueeze(0))
                preds = model(batch)
                roiobjs = predictions_to_rois(preds, width, height, top_k, score_threshold,
                                              output_polygons, mask_threshold, mask_nth, output_minrect,
                                              view_margin, fully_connected)

                info = ImageInfo(os.path.basename(im_list[i]))
                roiext = (info, roiobjs)
                roiwriter = ROIWriter(output=tmp_dir if tmp_dir is not None else output_dir, no_images=True)
                roiwriter.save([roiext])
                if tmp_dir is not None:
                    os.rename(roi_path_tmp, roi_path)
        except:
            print("Failed processing images: {}".format(",".join(im_list)))
            print(traceback.format_exc())

        # Move finished images to output_path or delete it
        for i in range(len(im_list)):
            if delete_input:
                os.remove(im_list[i])
            else:
                os.rename(im_list[i], os.path.join(output_dir, os.path.basename(im_list[i])))

        end_time = datetime.now()
        inference_time = end_time - start_time
        inference_time = int(inference_time.total_seconds() * 1000)
        print("  Inference + I/O time: {} ms\n".format(inference_time))


def main(argv=None):
    """
    Parses the parameters or, if None, sys.argv and starts prediction mode.

    :param argv: the command-line parameters to parse (list of strings)
    :type: argv: list
    """

    parser = argparse.ArgumentParser(description='YOLACT Prediction')
    parser.add_argument('--model', required=True, type=str,
                        help='The trained model to use (.pth file).')
    parser.add_argument('--config', default="external_config",
                        help='The name of the configuration to use.')
    parser.add_argument('--top_k', default=5, type=int,
                        help='Further restrict the number of predictions (eg objects) to parse')
    parser.add_argument('--score_threshold', default=0, type=float,
                        help='Detections with a score under this threshold will not be considered.')
    parser.add_argument('--fast_nms', action="store_false",
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--cross_class_nms', action="store_true",
                        help='Whether compute NMS cross-class or per-class.')
    parser.add_argument('--prediction_in', default=None, type=str, required=True,
                        help='The directory in which to look for images for processing.')
    parser.add_argument('--prediction_out', default=None, type=str, required=True,
                        help='The directory to store the results in.')
    parser.add_argument('--prediction_tmp', default=None, type=str, required=False,
                        help='The directory to store the results in first, before moving them to the actual output directory.')
    parser.add_argument('--delete_input', action="store_true",
                        help='Whether to delete the input images rather than moving them to the output directory.')
    parser.add_argument('--output_polygons', action='store_true',
                        help='Whether to masks are predicted and polygons should be output in the ROIS CSV files', required=False, default=False)
    parser.add_argument('--mask_threshold', type=float,
                        help='The threshold (0-1) to use for determining the contour of a mask', required=False, default=0.1)
    parser.add_argument('--mask_nth', type=int,
                        help='To speed polygon detection up, use every nth row and column only', required=False, default=1)
    parser.add_argument('--output_minrect', action='store_true',
                        help='When outputting polygons whether to store the minimal rectangle around the objects in the CSV files as well', required=False, default=False)
    parser.add_argument('--view_margin', default=2, type=int, required=False,
                        help='The number of pixels to use as margin around the masks when determining the polygon')
    parser.add_argument('--fully_connected', default='high', choices=['high', 'low'], required=False,
                        help='When determining polygons, whether regions of high or low values should be fully-connected at isthmuses')
    parsed = parser.parse_args(args=argv)

    with torch.no_grad():
        # initializing cudnn
        print('Initializing cudnn', end='')
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print(' Done.')

        # load configuration and model
        print('Loading config %s' % parsed.config, end='')
        set_cfg(parsed.config)
        cfg.mask_proto_debug = False
        print(' Done.')

        print('Loading model: %s' % parsed.model, end='')
        net = Yolact()
        net.load_weights(parsed.model)
        net.eval()
        net = net.cuda()
        net.detect.use_fast_nms = parsed.fast_nms
        net.detect.use_cross_class_nms = parsed.cross_class_nms
        print(' Done.')

        predict(model=net, input_dir=parsed.prediction_in, output_dir=parsed.prediction_out, tmp_dir=parsed.prediction_tmp,
                top_k=parsed.top_k, score_threshold=parsed.score_threshold, delete_input=parsed.delete_input,
                output_polygons=parsed.output_polygons, mask_threshold=parsed.mask_threshold, mask_nth=parsed.mask_nth,
                output_minrect=parsed.output_minrect, view_margin=parsed.view_margin, fully_connected=parsed.fully_connected)


if __name__ == '__main__':
    main()