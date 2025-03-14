# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np
import math
import sys

from siamese.core.config import cfg
from siamese.tracker.tracker_builder import build_tracker
from siamese.utils.bbox import get_axis_aligned_bbox
from siamese.utils.model_load import load_pretrain
from siamese.models.model_builder import ModelBuilder
from toolkit.utils.region import vot_overlap, vot_float2str

from toolkit.datasets import DatasetFactory

sys.path.append('../')

parser = argparse.ArgumentParser(description='siamfdb tracking')

parser.add_argument('--video', type=str, default='',
                    help='eval one special video')
parser.add_argument('--dataset_dir', type=str, default='',
                    help='datasets')
parser.add_argument('--dataset', type=str, default='',
                    help='datasets')
parser.add_argument('--vis', action='store_true', default=False,
                    help='whether visualzie result')
parser.add_argument('--start_epoch', type=int, default=11,
                    help='snapshot of models to eval')
parser.add_argument('--end_epoch', type=int, default=20,
                    help='snapshot of models to eval')
parser.add_argument('--snapshot_dir', type=str, default='snapshot',
                    help='snapshot of models to eval')
parser.add_argument('--config', type=str, default='',
                    help='config file')
args = parser.parse_args()

torch.set_num_threads(1)



def main():
    # load config
    cfg.merge_from_file(args.config)

    # hp_search
    dataset = args.dataset
    params = getattr(cfg.HP_SEARCH, dataset)
    hp = {'lr': params[0], 'penalty_k': params[1], 'window_lr': params[2]}

    dataset_root = os.path.join(args.dataset_dir, args.dataset)
    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)
    model = ModelBuilder()
    start_epoch = args.start_epoch
    end_epoch = args.end_epoch
    for idx in range(start_epoch, end_epoch + 1):
        print(f"Epoch: {idx}({start_epoch}~{end_epoch}), Left_epoch: {end_epoch - idx}")
        snapshot = os.path.join(args.snapshot_dir, "checkpoint_e{}.pth".format(idx))
        model_name = snapshot.split('/')[-1].split('.')[0] + '_' + str(hp['lr']) + '_' + str(hp['penalty_k']) + '_' + str(hp['window_lr'])
        if not os.path.exists(snapshot):
            if idx is not end_epoch:
                print(snapshot + 'file does not exist')
                continue
            else:
                snapshot = os.path.join(args.snapshot_dir, 'end', "checkpoint_e{}.pth".format(idx))
                if not os.path.exists(snapshot):
                    print(snapshot + 'file does not exist')
                    continue
                model_name = snapshot.split('/')[-1].split('.')[0] + '_' + str(hp['lr']) + '_' + str(hp['penalty_k']) + '_' + str(hp['window_lr'])
        model = load_pretrain(model, snapshot).cuda().eval()
        # build tracker
        tracker = build_tracker(model)

        model_path = os.path.join('results', args.dataset, model_name)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        # OPE tracking
        total_lost = 0
        if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
            for v_idx, video in enumerate(dataset):
                if args.video != '':
                    # test one special video
                    if video.name != args.video:
                        continue
                frame_counter = 0
                lost_number = 0
                toc = 0
                pred_bboxes = []
                for idx, (img, gt_bbox) in enumerate(video):
                    if len(gt_bbox) == 4:
                        gt_bbox = [gt_bbox[0], gt_bbox[1],
                                   gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                                   gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1,
                                   gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]
                    tic = cv2.getTickCount()
                    if idx == frame_counter:
                        cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                        gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                        tracker.init(img, gt_bbox_)
                        pred_bbox = gt_bbox_
                        pred_bboxes.append(1)
                    elif idx > frame_counter:
                        outputs = tracker.track(img, hp)
                        pred_bbox = outputs['bbox']
                        overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                        if overlap > 0:
                            # not lost
                            pred_bboxes.append(pred_bbox)
                        else:
                            # lost object
                            pred_bboxes.append(2)
                            frame_counter = idx + 5  # skip 5 frames
                            lost_number += 1
                    else:
                        pred_bboxes.append(0)
                    toc += cv2.getTickCount() - tic
                    if idx == 0:
                        cv2.destroyAllWindows()
                    if args.vis and idx > frame_counter:
                        cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                                      True, (0, 255, 0), 3)
                        bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (bbox[0], bbox[1]),
                                      (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 255), 3)
                        cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow(video.name, img)
                        cv2.waitKey(1)
                toc /= cv2.getTickFrequency()
                # save results
                video_path = os.path.join(model_path,
                                          'baseline', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        if isinstance(x, int):
                            f.write("{:d}\n".format(x))
                        else:
                            f.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')
                print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                    v_idx + 1, video.name, toc, idx / toc, lost_number))
                total_lost += lost_number
            print("{:s} total lost: {:d}".format(model_name, total_lost))
        else:
            for v_idx, video in enumerate(dataset):
                if args.video != '':
                    # test one special video
                    if video.name != args.video:
                        continue
                toc = 0
                pred_bboxes = []
                track_times = []
                scores = []
                for idx, (img, gt_bbox) in enumerate(video):
                    tic = cv2.getTickCount()
                    if idx == 0:
                        tracker.init(img, gt_bbox)
                        pred_bbox = gt_bbox
                        scores.append(None)
                        if 'VOT2018-LT' == args.dataset:
                            pred_bboxes.append([1])
                        else:
                            pred_bboxes.append(pred_bbox)
                    else:
                        outputs = tracker.track(img, hp)
                        pred_bbox = outputs['bbox']
                        pred_bboxes.append(pred_bbox)
                        scores.append(outputs['best_score'])
                    toc += cv2.getTickCount() - tic
                    track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
                    if idx == 0:
                        cv2.destroyAllWindows()
                    if args.vis and idx > 0:
                        if not any(map(math.isnan, gt_bbox)):
                            gt_bbox = list(map(int, gt_bbox))
                            pred_bbox = list(map(int, pred_bbox))
                            cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                          (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 3)
                            cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                          (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 255), 3)
                            cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            cv2.imshow(video.name, img)
                            cv2.waitKey(1)
                toc /= cv2.getTickFrequency()
                # save results
                if 'VOT2018-LT' == args.dataset:
                    video_path = os.path.join('results', args.dataset, model_name,
                                              'longterm', video.name)
                    if not os.path.isdir(video_path):
                        os.makedirs(video_path)
                    result_path = os.path.join(video_path,
                                               '{}_001.txt'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in pred_bboxes:
                            f.write(','.join([str(i) for i in x]) + '\n')
                    result_path = os.path.join(video_path,
                                               '{}_001_confidence.value'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in scores:
                            f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                    result_path = os.path.join(video_path,
                                               '{}_time.txt'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in track_times:
                            f.write("{:.6f}\n".format(x))
                elif "GOT" in args.dataset:
                    video_path = os.path.join(model_path, video.name)
                    if not os.path.isdir(video_path):
                        os.makedirs(video_path)
                    result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in pred_bboxes:
                            f.write(','.join([str(i) for i in x]) + '\n')
                    result_path = os.path.join(video_path,
                                               '{}_time.txt'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in track_times:
                            f.write("{:.6f}\n".format(x))
                else:
                    result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in pred_bboxes:
                            f.write(','.join([str(i) for i in x]) + '\n')
                print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                    v_idx + 1, video.name, toc, idx / toc))


if __name__ == '__main__':
    main()
