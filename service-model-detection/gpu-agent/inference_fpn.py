from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
from detectron.utils.timer import Timer
from json_dataset import JsonDataset
import detectron.utils.c2 as c2_utils
# from detectron.core.test import im_detect_all
import datetime
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import logging
from detectron.core.config import cfg
import numpy as np
from detectron.core.test_engine import initialize_model_from_cfg
from detectron.core.test import im_detect_bbox
from detectron.core.test import im_detect_bbox_hflip
from detectron.core.test import im_detect_bbox_scale
import detectron.utils.py_cpu_nms as py_nms_utils
import detectron.utils.boxes as box_utils
import detectron.utils.vis as vis_utils
# from detectron.core.test import box_results_with_nms_and_limit
import os
import detectron.utils.cython_bbox as cython_bbox
from detectron.core.config import merge_cfg_from_file
from detectron.core.config import assert_and_infer_cfg
import detectron.datasets.dataset_catalog as dataset_catalog
import json
from detectron.core.config import get_output_dir


import cv2
import logging
import numpy as np
import os
import json
import pprint
from caffe2.python import workspace

from detectron.core.config import cfg
from detectron.core.config import get_output_dir

import argparse

logger = logging.getLogger(__name__)
bbox_overlaps = cython_bbox.bbox_overlaps

def im_detect_all(model, im, box_proposals, timers=None):
    if timers is None:
        timers = defaultdict(Timer)

    timers['im_detect_bbox'].tic()
    scores, boxes, im_scale = im_detect_bbox(model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes=box_proposals)
    timers['im_detect_bbox'].toc()

    timers['misc_bbox'].tic()
    scores, boxes, cls_boxes, entropy_m = box_results_with_nms_and_limit(scores, boxes)
    timers['misc_bbox'].toc()

    total_uc = calc_ent_class(entropy_m)

    if scores.shape[0] > 0:
        total_uc = total_uc  # +scores.shape[0]/10.0
    else:
        total_uc = 1.0
    # print(total_uc)
    return cls_boxes, total_uc


def calc_uc_class(origin_det, new_det):
    uc = 0.0
    num_classes = cfg.MODEL.NUM_CLASSES
    for j in range(1, num_classes):
        class_uc = 0.0

        cls_origin = origin_det[j]
        cls_new = new_det[j]
        top_dets_out = cls_origin.copy()
        top_boxes = cls_origin[:, :4]
        origin_scores = cls_origin[:, 4]
        all_boxes = cls_new[:, :4]
        all_scores = cls_new[:, 4]
        top_to_all_overlaps = bbox_overlaps(top_boxes, all_boxes)
        for k in range(top_dets_out.shape[0]):
            inds_to_vote = np.where(top_to_all_overlaps[k] >= 0.5)[0]
            scores_to_vote = all_scores[inds_to_vote]
            if len(inds_to_vote) > 0:
                iou = top_to_all_overlaps[k][inds_to_vote]
                value = (np.sum(np.abs(origin_scores[k] - scores_to_vote)) + np.sum(1 - iou)) * origin_scores[k]
                # print(scores_to_vote)
                # print(iou)
                # print(value)
                class_uc += value
            else:
                # print('####')
                class_uc += 2.0 * origin_scores[k]
        if top_dets_out.shape[0] > 0:
            uc += class_uc / top_dets_out.shape[0]
    return uc


def calc_uc(origin_det, new_det):
    uc = 0.0
    num_classes = cfg.MODEL.NUM_CLASSES
    for j in range(1, num_classes):
        cls_origin = origin_det[j]
        cls_new = new_det[j]
        top_dets_out = cls_origin.copy()
        top_boxes = cls_origin[:, :4]
        all_boxes = cls_new[:, :4]
        all_scores = cls_new[:, 4]
        top_to_all_overlaps = bbox_overlaps(top_boxes, all_boxes)
        for k in range(top_dets_out.shape[0]):
            inds_to_vote = np.where(top_to_all_overlaps[k] >= 0.5)[0]
            scores_to_vote = all_scores[inds_to_vote]
            if len(inds_to_vote) > 0:
                iou = top_to_all_overlaps[k][inds_to_vote]
                value = (1 - scores_to_vote) + (1 - iou)
                uc += np.sum(value)
            else:
                uc += 2.0
    return uc


def calc_ent_class(entropy_m):
    uc = 0.0
    num_classes = cfg.MODEL.NUM_CLASSES
    for j in range(1, num_classes):
        class_uc = 0.0
        scores = entropy_m[j]
        if scores.shape[0] > 0:
            logp = np.log2(scores)
            ent = np.sum(-np.multiply(scores, logp))  # /scores.shape[0]
            class_uc += ent / scores.shape[0]
        uc += class_uc
    return uc


def calc_ent(scores):
    logger.info("shape = {}".format(scores.shape))
    scores = scores[1:, :]
    logp = np.log2(scores)
    if scores.shape[0] > 0:
        ent = np.sum(-np.multiply(scores, logp))  # /scores.shape[0]
    else:
        ent = 0.5
    # logger.info(ent)
    return ent


def box_results_with_nms_and_limit(scores, boxes, confs=None):
    """Returns bounding-box detection results by thresholding on scores and
    applying non-maximum suppression (NMS).

    `boxes` has shape (#detections, 4 * #classes), where each row represents
    a list of predicted bounding boxes for each of the object classes in the
    dataset (including the background class). The detections in each row
    originate from the same object proposal.

    `scores` has shape (#detection, #classes), where each row represents a list
    of object detection confidence scores for each of the object classes in the
    dataset (including the background class). `scores[i, j]`` corresponds to the
    box at `boxes[i, j * 4:(j + 1) * 4]`.
    """
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_boxes = [[] for _ in range(num_classes)]
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    entropy_m = [[] for _ in range(num_classes)]
    # entropy_m = np.zeros((num_classes-1,), dtype = np.int)
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > cfg.TEST.SCORE_THRESH)[0]
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j * 4:(j + 1) * 4]
        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(
            np.float32, copy=False
        )
        if cfg.STD_NMS:
            confs_j = confs[inds, j * 4:(j + 1) * 4]
            nms_dets, _ = py_nms_utils.soft(dets_j,
                                            confidence=confs_j)
        elif cfg.TEST.SOFT_NMS.ENABLED:
            nms_dets, _ = box_utils.soft_nms(
                dets_j,
                sigma=cfg.TEST.SOFT_NMS.SIGMA,
                overlap_thresh=cfg.TEST.NMS,
                score_thresh=0.0001,
                method=cfg.TEST.SOFT_NMS.METHOD
            )
        else:
            keep = box_utils.nms(dets_j, cfg.TEST.NMS)
            nms_dets = dets_j[keep, :]
            # keep_idx = np.concatenate((keep_idx, inds[keep]))
            inds = inds[keep]
            keep_idx = np.empty((0, 20))
            keep_idx = np.vstack((keep_idx, scores[inds, 1:]))
        entropy_m[j] = keep_idx
        # Refine the post-NMS boxes using bounding-box voting
        if cfg.TEST.BBOX_VOTE.ENABLED:
            nms_dets = box_utils.box_voting(
                nms_dets,
                dets_j,
                cfg.TEST.BBOX_VOTE.VOTE_TH,
                scoring_method=cfg.TEST.BBOX_VOTE.SCORING_METHOD
            )
        cls_boxes[j] = nms_dets

    # Limit to max_per_image detections **over all classes**
    if cfg.TEST.DETECTIONS_PER_IM > 0:
        image_scores = np.hstack(
            [cls_boxes[j][:, -1] for j in range(1, num_classes)]
        )
        if len(image_scores) > cfg.TEST.DETECTIONS_PER_IM:
            image_thresh = np.sort(image_scores)[-cfg.TEST.DETECTIONS_PER_IM]
            for j in range(1, num_classes):
                keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]
                entropy_m[j] = entropy_m[j][keep, :]
    #     im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
    #     boxes = im_results[:, :-1]
    #     scores = im_results[:, -1]
    #     return scores, boxes, cls_boxes
    # entropy = calc_ent(entropy_m)
    im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]
    return scores, boxes, cls_boxes, entropy_m


def compare(x, y):
    if x['entropy'] > y['entropy']:
        return -1
    elif x['entropy'] < y['entropy']:
        return 1
    else:
        return 0


def vis(dataset, roidb):
    tmp = []
    for i, entry in enumerate(roidb):
        tmp.append(entry)
    tmp.sort(cmp=compare)

    file_name = '/home/wushujie/vis/result_scale_mean.txt'
    with open(file_name, 'wb') as file_object:
        for entry in tmp:
            file_object.write(entry['image'])
            file_object.write(str(entry['entropy']))
            file_object.write('\n')

    max_img = tmp[:20]
    min_img = tmp[-20:]
    for i, entry in enumerate(max_img):
        im = cv2.imread(entry['image'])
        im_name = str(i)  # +'_'+entry['file_name']
        output_dir = '/home/wushujie/vis/max_scale_mean'
        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            output_dir,
            entry['result'],
            None,
            None,
            dataset=dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            ext='png',
            out_when_no_box=True
        )
    for i, entry in enumerate(min_img):
        im = cv2.imread(entry['image'])
        im_name = str(i)
        output_dir = '/home/wushujie/vis/min_scale_mean'
        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            output_dir,
            entry['result'],
            None,
            None,
            dataset=dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            ext='png',
            out_when_no_box=True
        )

def get_roidb_and_dataset(dataset_name, ind_range):
    dataset = JsonDataset(dataset_name)
    roidb = dataset.get_roidb()
    if ind_range is not None:
        total_num_images = len(roidb)
        start, end = ind_range
        roidb = roidb[start:end]
    else:
        start = 0
        end = len(roidb)
        total_num_images = end
    return roidb, dataset, start, end, total_num_images

def empty_results(num_classes, num_images):
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    return all_boxes

def extend_results(index, all_res, im_res):
    for cls_idx in range(1, len(im_res)):
        all_res[cls_idx][index] = im_res[cls_idx]

def test_net(weights_file, dataset_name, ind_range=None, gpu_id=0):
    roidb, dataset, start_ind, end_ind, total_num_images = get_roidb_and_dataset(dataset_name, ind_range)
    model = initialize_model_from_cfg(weights_file, gpu_id=gpu_id)
    num_images = len(roidb)
    num_classes = cfg.MODEL.NUM_CLASSES
    all_boxes = empty_results(num_classes, num_images)
    timers = defaultdict(Timer)
    result = {}
    for i, entry in enumerate(roidb):
        box_proposals = None
        im = cv2.imread(entry['image'])
        # print(entry['image'])
        # im_name = os.path.splitext(os.path.basename(entry['image']))[0]
        im_result = {}
        im_result['fileName'] = entry['file_name']
        im_result['imageId'] = str(entry['id'])
        # print(entry['file_name'])
        with c2_utils.NamedCudaScope(gpu_id):
            cls_boxes_i, uc = im_detect_all(model, im, box_proposals, timers)
            entry['entropy'] = uc
            # print(uc)
            # entry['result'] = cls_boxes_i
            # entropy.append(entropy_i)
        ## all_boxes
        cls_box_result = {}
        for cls_idx in range(1, len(cls_boxes_i)):
            if(len(cls_boxes_i[cls_idx]))==0:
                continue
            cls_box_result[str(cls_idx)] = []
            for box_idx in cls_boxes_i[cls_idx]:
                box_i={}
                box_i['x']=int(box_idx[0])
                box_i['y']=int(box_idx[1])
                box_i['w']=int(box_idx[2]-box_idx[0])
                box_i['h']=int(box_idx[3]-box_idx[1])
                box_i['score']=float(box_idx[4])
                cls_box_result[str(cls_idx)].append(box_i)
        extend_results(i, all_boxes, cls_boxes_i)
        im_result['annotation'] = cls_box_result
        result[str(im_result['imageId'])] = im_result
        if i % 10 == 0:  # Reduce log file size
            ave_total_time = np.sum([t.average_time for t in timers.values()])
            eta_seconds = ave_total_time * (num_images - i - 1)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            det_time = (
                    timers['im_detect_bbox'].average_time +
                    timers['im_detect_mask'].average_time +
                    timers['im_detect_keypoints'].average_time
            )
            misc_time = (
                    timers['misc_bbox'].average_time +
                    timers['misc_mask'].average_time +
                    timers['misc_keypoints'].average_time
            )
            logger.info(
                (
                    'im_detect: range [{:d}, {:d}] of {:d}: '
                    '{:d}/{:d} {:.3f}s + {:.3f}s (eta: {})'
                ).format(
                    start_ind + 1, end_ind, total_num_images, start_ind + i + 1,
                    start_ind + num_images, det_time, misc_time, eta
                )
            )

    print(result)
    return roidb, result

def load_data(dataset_name,image_id_list = None):
    imgs = []
    annotation_file = dataset_catalog.get_ann_fn(dataset_name)
    dataset = json.load(open(annotation_file, 'r'))
    for img in dataset['images']:
        imgs.append(img)
    output_dir = get_output_dir(cfg.TRAIN.DATASETS, training=True)
    file_name = os.path.join(output_dir, 'train_init.json')
    select = []
    image_id_set = set()
    for i in image_id_list:
        image_id_set.add(int(i))
    for i in range(0, len(imgs)):
        if(imgs[i]['id'] in image_id_set):
            select.append(imgs[i])
    dataset['images'] = select
    with open(file_name, 'wt') as f:
        json.dump(dataset, f)
    return dataset

def parse_args():
    parser = argparse.ArgumentParser(
        description='Inference Config'
    )
    parser.add_argument(
        '--input',
        dest='input_file',
        help='image id file for training (and optionally testing)',
        type=str
    )
    # parser.add_argument(
    #     '--dataset',
    #     dest='dataset',
    #     help='model file for training (and optionally testing)',
    #     default='voc_2007_test',
    #     type=str
    # )
    # parser.add_argument(
    #     '--task',
    #     dest='task_id',
    #     help='task_id',
    #     type=str
    # )
    parser.add_argument(
        '--model',
        dest='model_file',
        help='model file for training (and optionally testing)',
        default='/home/LAB/wusj/exp/output/test_fpn/train/voc_2007_train/generalized_rcnn/model_final0.7.pkl',
        type=str
    )
    parser.add_argument(
        '--output',
        dest='output_dir',
        help='output_dir',
        default='/home/LAB/wusj/fastwash_tmp/inference/',
        type=str
    )
    return parser.parse_args()

def main():
    c2_utils.import_contrib_ops()
    c2_utils.import_detectron_ops()
    cv2.ocl.setUseOpenCL(False)
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0', '--caffe2_gpu_memory_tracking=1'])
    merge_cfg_from_file('/home/LAB/wusj/exp/KL-Loss/configs/e2e_faster_rcnn_R-50-FPN_2x_entropy.yaml')
    assert_and_infer_cfg(cache_urls=False)
    smi_output, cuda_ver, cudnn_ver = c2_utils.get_nvidia_info()
    logger.info("cuda version : {}".format(cuda_ver))
    logger.info("cudnn version: {}".format(cudnn_ver))
    logger.info("nvidia-smi output:\n{}".format(smi_output))
    logger.info('Training with config:')
    logger.info(pprint.pformat(cfg))
    workspace.ResetWorkspace()

    np.random.seed(cfg.RNG_SEED)
    args = parse_args()
    input_file = args.input_file
    result_output_dir = args.output_dir

    with open(input_file, 'r') as f:
        config = json.load(f)

        dataset = config['dataSetName']
        model_file = args.model_file
        task_id = config['id']
        image_id_list = config['imageIdList']

        load_data(dataset,image_id_list)
        roidb, result = test_net(model_file,dataset)
        config['inferenceResult'] = result
        with open(result_output_dir+'result_'+task_id, 'wt') as f2:
            json.dump(config, f2)

    # weight_file = '/home/LAB/wusj/exp/output/test_base_plus/train/voc_2007_train/generalized_rcnn/model_final0.1.pkl'

    # for dataset in ['voc_2007_train']:
    #     imgs, perm, data = load_data(dataset)
    # ratio = 0.1
    # sum = 0.2
    #
    # weight = computeClassWeight()
    # imgs, data = computeImageWeight(data, imgs, weight)
    #
    # initsize = int(len(imgs) * 0.1)
    # for i in range(0, initsize):
    #     global_train.append(imgs[perm[i]])
    #
    # perm = perm[initsize:]
    # perm = read_pre(sum, perm)

    # while sum <= 1.00001:
    #     select, perm = select_data(imgs, perm, ratio, data, sum - 0.1, type='select', classWeight=weight)
    #     select, perm = select_data(imgs, perm, ratio, data, sum, type='train')
    #     print('sum = {},select = {},remain_size = {} '.format(sum, len(select), len(perm)))
    #     res = commands.getoutput(
    #         'python ' + './exp/train_net_fpn_class.py ' + '--cfg ./configs/e2e_faster_rcnn_R-50-FPN_2x_fpn_no_weight.yaml ' + '--sum ' + str(
    #             sum))
    #     print(res)
    #     print('end' + str(sum))
    #     sum = sum + ratio
if __name__ == '__main__':
    main()
def im_detect_bbox_aug(model, im, box_proposals=None):
    """Performs bbox detection with test-time augmentations.
    Function signature is the same as for im_detect_bbox.
    """
    assert not cfg.TEST.BBOX_AUG.SCALE_SIZE_DEP, \
        'Size dependent scaling not implemented'
    assert not cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION' or \
           cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION', \
        'Coord heuristic must be union whenever score heuristic is union'
    assert not cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION' or \
           cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION', \
        'Score heuristic must be union whenever coord heuristic is union'
    assert not cfg.MODEL.FASTER_RCNN or \
           cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION', \
        'Union heuristic must be used to combine Faster RCNN predictions'

    # Collect detections computed under different transformations
    scores_ts = []
    boxes_ts = []

    ## type:hflip:1,scale:value,scale_flip:bu yong,origin:0
    def add_preds_t(scores_t, boxes_t, type):
        scores_ts.append(scores_t)
        boxes_ts.append(boxes_t)

    # Perform detection on the horizontally flipped image
    if cfg.TEST.BBOX_AUG.H_FLIP:
        scores_hf, boxes_hf, _ = im_detect_bbox_hflip(
            model,
            im,
            cfg.TEST.SCALE,
            cfg.TEST.MAX_SIZE,
            box_proposals=box_proposals
        )
        add_preds_t(scores_hf, boxes_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.BBOX_AUG.SCALES:
        max_size = cfg.TEST.BBOX_AUG.MAX_SIZE
        scores_scl, boxes_scl = im_detect_bbox_scale(
            model, im, scale, max_size, box_proposals
        )
        add_preds_t(scores_scl, boxes_scl)

        if cfg.TEST.BBOX_AUG.SCALE_H_FLIP:
            scores_scl_hf, boxes_scl_hf = im_detect_bbox_scale(
                model, im, scale, max_size, box_proposals, hflip=True
            )
            add_preds_t(scores_scl_hf, boxes_scl_hf)

    # Compute detections for the original image (identity transform) last to
    # ensure that the Caffe2 workspace is populated with blobs corresponding
    # to the original image on return (postcondition of im_detect_bbox)
    scores_i, boxes_i, im_scale_i = im_detect_bbox(
        model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes=box_proposals
    )
    add_preds_t(scores_i, boxes_i)

    # Combine the predicted scores
    if cfg.TEST.BBOX_AUG.SCORE_HEUR == 'ID':
        scores_c = scores_i
    elif cfg.TEST.BBOX_AUG.SCORE_HEUR == 'AVG':
        scores_c = np.mean(scores_ts, axis=0)
    elif cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION':
        scores_c = np.vstack(scores_ts)
    else:
        raise NotImplementedError(
            'Score heur {} not supported'.format(cfg.TEST.BBOX_AUG.SCORE_HEUR)
        )

    # Combine the predicted boxes
    if cfg.TEST.BBOX_AUG.COORD_HEUR == 'ID':
        boxes_c = boxes_i
    elif cfg.TEST.BBOX_AUG.COORD_HEUR == 'AVG':
        boxes_c = np.mean(boxes_ts, axis=0)
    elif cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION':
        boxes_c = np.vstack(boxes_ts)
    else:
        raise NotImplementedError(
            'Coord heur {} not supported'.format(cfg.TEST.BBOX_AUG.COORD_HEUR)
        )

    return scores_c, boxes_c, im_scale_i