from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2

from siamese.core.config import cfg
from siamese.tracker.base_tracker import SiameseBaseTracker
from siamese.utils.misc import bbox_clip
from siamese.utils.bbox import corner2center


class SiameseTracker(SiameseBaseTracker):
    def __init__(self, model, cfg=cfg.TRACK):
        super(SiameseTracker, self).__init__()
        hanning = np.hanning(cfg.SCORE_SIZE)
        self.window = np.outer(hanning, hanning)
        self.model = model
        self.model.eval()
        if cfg.INTERPOLATION is False:
            def generate_points(stride, size):
                ori = - (size // 2) * stride
                x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                                   [ori + stride * dy for dy in np.arange(0, size)])
                points = np.zeros((size * size, 2), dtype=np.float32)
                points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()

                return points
            self.points = generate_points(cfg.STRIDE, cfg.SCORE_SIZE)

    def _convert_cls(self, cls):
        cls = cls[:, :, :, :].data[:, 1, :, :].cpu().numpy()
        return cls

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                                    bbox[1] + (bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.model.template(z_crop)

    def change(self, r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        return np.sqrt((w + pad) * (h + pad))

    def cal_penalty(self, ltrbs, penalty_lk):
        bboxes_w = ltrbs[0, :, :] + ltrbs[2, :, :]
        bboxes_h = ltrbs[1, :, :] + ltrbs[3, :, :]
        s_c = self.change(
            self.sz(bboxes_w, bboxes_h) / self.sz(self.size[0] * self.scale_z, self.size[1] * self.scale_z))
        r_c = self.change((self.size[0] / self.size[1]) / (bboxes_w / bboxes_h))
        penalty = np.exp(-(r_c * s_c - 1) * penalty_lk)
        return penalty

    def accurate_location(self, max_r_up, max_c_up):
        dist = int((cfg.TRACK.INSTANCE_SIZE - (cfg.TRACK.SCORE_SIZE - 1) * 8) / 2)
        max_r_up += dist
        max_c_up += dist
        p_cool_s = np.array([max_r_up, max_c_up])
        disp = p_cool_s - (np.array([cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE]) - 1.) / 2.
        return disp

    def coarse_location(self, hp_score_up, p_score_up, scale_score, lrtbs):
        upsize = (cfg.TRACK.SCORE_SIZE - 1) * cfg.TRACK.STRIDE + 1
        max_r_up_hp, max_c_up_hp = np.unravel_index(hp_score_up.argmax(), hp_score_up.shape)
        max_r = int(round(max_r_up_hp / scale_score))
        max_c = int(round(max_c_up_hp / scale_score))
        max_r = bbox_clip(max_r, 0, cfg.TRACK.SCORE_SIZE)
        max_c = bbox_clip(max_c, 0, cfg.TRACK.SCORE_SIZE)
        bbox_region = lrtbs[max_r, max_c, :]
        min_bbox = int(cfg.TRACK.REGION_S * cfg.TRACK.EXEMPLAR_SIZE)
        max_bbox = int(cfg.TRACK.REGION_L * cfg.TRACK.EXEMPLAR_SIZE)
        l_region = int(min(max_c_up_hp, bbox_clip(bbox_region[0], min_bbox, max_bbox)) / 2.0)
        t_region = int(min(max_r_up_hp, bbox_clip(bbox_region[1], min_bbox, max_bbox)) / 2.0)

        r_region = int(min(upsize - max_c_up_hp, bbox_clip(bbox_region[2], min_bbox, max_bbox)) / 2.0)
        b_region = int(min(upsize - max_r_up_hp, bbox_clip(bbox_region[3], min_bbox, max_bbox)) / 2.0)
        mask = np.zeros_like(p_score_up)
        mask[max_r_up_hp - t_region:max_r_up_hp + b_region + 1, max_c_up_hp - l_region:max_c_up_hp + r_region + 1] = 1
        p_score_up = p_score_up * mask
        return p_score_up

    def getCenter(self, hp_score_up, p_score_up, scale_score, lrtbs):
        # corse location
        score_up = self.coarse_location(hp_score_up, p_score_up, scale_score, lrtbs)
        # accurate location
        max_r_up, max_c_up = np.unravel_index(score_up.argmax(), score_up.shape)
        disp = self.accurate_location(max_r_up, max_c_up)
        disp_ori = disp / self.scale_z
        new_cx = disp_ori[1] + self.center_pos[0]
        new_cy = disp_ori[0] + self.center_pos[1]
        return max_r_up, max_c_up, new_cx, new_cy

    def track(self, img, hp):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        self.scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        outputs = self.model.track(x_crop)
        cls = outputs['cls'][:, :, :, :].data[:, 1, :, :].cpu().numpy().squeeze()
        ltrbs = outputs['loc'].data.cpu().numpy().squeeze()

        if cfg.TRACK.INTERPOLATION:
            upsize = (cfg.TRACK.SCORE_SIZE - 1) * cfg.TRACK.STRIDE + 1
            penalty = self.cal_penalty(ltrbs, hp['penalty_k'])
            p_score = penalty * cls
            if cfg.TRACK.hanming:
                hp_score = p_score * (1 - hp['window_lr']) + self.window * hp['window_lr']
            else:
                hp_score = p_score
            hp_score_up = cv2.resize(hp_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
            p_score_up = cv2.resize(p_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
            cls_up = cv2.resize(cls, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
            ltrbs = np.transpose(ltrbs, (1, 2, 0))
            ltrbs_up = cv2.resize(ltrbs, (upsize, upsize), interpolation=cv2.INTER_CUBIC)

            scale_score = upsize / cfg.TRACK.SCORE_SIZE
            # get center
            max_r_up, max_c_up, new_cx, new_cy = self.getCenter(hp_score_up, p_score_up,
                                                                scale_score, ltrbs)
            # get w h
            ave_w = (ltrbs_up[max_r_up, max_c_up, 0] + ltrbs_up[max_r_up, max_c_up, 2]) / self.scale_z
            ave_h = (ltrbs_up[max_r_up, max_c_up, 1] + ltrbs_up[max_r_up, max_c_up, 3]) / self.scale_z
            s_c = self.change(self.sz(ave_w, ave_h) / self.sz(self.size[0] * self.scale_z,
                                                              self.size[1] * self.scale_z))
            r_c = self.change((self.size[0] / self.size[1]) / (ave_w / ave_h))
            penalty = np.exp(-(r_c * s_c - 1) * hp['penalty_k'])
            lr = penalty * cls_up[max_r_up, max_c_up] * hp['lr']
            new_width = lr * ave_w + (1 - lr) * self.size[0]
            new_height = lr * ave_h + (1 - lr) * self.size[1]
        else:
            score = cls.reshape(-1)
            ltrbs = ltrbs.reshape(4, -1)
            ltrbs[0, :] = self.points[:, 0] - ltrbs[0, :]
            ltrbs[1, :] = self.points[:, 1] - ltrbs[1, :]
            ltrbs[2, :] = self.points[:, 0] + ltrbs[2, :]
            ltrbs[3, :] = self.points[:, 1] + ltrbs[3, :]
            ltrbs[0, :], ltrbs[1, :], ltrbs[2, :], ltrbs[3, :] = corner2center(ltrbs)
            s_c = self.change(self.sz(ltrbs[2, :], ltrbs[3, :]) /
                              (self.sz(self.size[0] * self.scale_z, self.size[1] * self.scale_z)))

            r_c = self.change((self.size[0] / self.size[1]) /
                              (ltrbs[2, :] / ltrbs[3, :]))
            penalty = np.exp(-(r_c * s_c - 1) * hp['penalty_k'])
            p_score = penalty * score
            if cfg.TRACK.hanming:
                hp_score = p_score * (1 - hp['window_lr']) + self.window.flatten() * hp['window_lr']
            else:
                hp_score = p_score
            best_idx = np.argmax(hp_score)
            bbox = ltrbs[:, best_idx] / self.scale_z
            lr = penalty[best_idx] * score[best_idx] * hp['lr']

            new_cx = bbox[0] + self.center_pos[0]
            new_cy = bbox[1] + self.center_pos[1]
            new_width = self.size[0] * (1 - lr) + bbox[2] * lr
            new_height = self.size[1] * (1 - lr) + bbox[3] * lr
        cx = bbox_clip(new_cx, 0, img.shape[1])
        cy = bbox_clip(new_cy, 0, img.shape[0])
        width = bbox_clip(new_width, 10, img.shape[1])
        height = bbox_clip(new_height, 10, img.shape[0])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
            'bbox': bbox,
            'best_score': best_score
        }
