from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters.gaussian import gaussian_blur2d
import torchvision.transforms as transforms

def blur(x):
    return gaussian_blur2d(x, kernel_size=(51, 51), sigma=(50., 50.))

class UnNormalize(nn.Module):
    def __init__(self, mean, std):
        super(UnNormalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        """
        Args:
        :param tensor: tensor image of size (B,C,H,W) to be un-normalized
        :return: UnNormalized image
        """
        mean = torch.as_tensor(self.mean, dtype=tensor.dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=tensor.dtype, device=tensor.device)
        tensor = tensor
        
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)
        
        if tensor.ndim > 3:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        mean = mean.expand(tensor.shape)
        std = std.expand(tensor.shape)
        tensor = tensor * std + mean
        tensor = tensor.clamp(min=0, max=1)
        return tensor


class GroupCAM(object):
    def __init__(self, model, target_layer="layer3.2", groups=32):
        super(GroupCAM, self).__init__()
        self.model = model
        self.groups = groups
        self.gradients = dict()
        self.activations = dict()
        self.transform_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.Nutransform = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.target_layer = target_layer
        if "backbone" in target_layer:
            for module in self.model.model.model.backbone.named_modules():
                if module[0] == '.'.join(target_layer.split('.')[1:]):
                    module[1].register_forward_hook(self.forward_hook)
                    module[1].register_backward_hook(self.backward_hook)
        if 'neck' in target_layer:
            for module in self.model.model.model.neck.named_modules():
                module[1].register_forward_hook(self.forward_hook)
                module[1].register_backward_hook(self.backward_hook)

        if 'head' in target_layer:
            for module in self.model.model.model.head.named_modules():
                if module[0] == '.'.join(target_layer.split('.')[1:]):
                    module[1].register_forward_hook(self.forward_hook)
                    module[1].register_backward_hook(self.backward_hook)
        if 'Model.relu' in target_layer:
            for module in self.model.model.named_modules():
                print(module)
                if module[0] == '.'.join(target_layer.split('.')[1:]):
                    module[1].register_forward_hook(self.forward_hook)
                    module[1].register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):

        self.gradients['value'] = grad_output[0]

    def forward_hook(self, module, input, output):
        self.activations['value'] = output

    def forward(self, x, hp, retain_graph=False):
        output = self.model.track_cam(x, hp)
        cls = output["cls"]
        x_crop = output["x_crop"]
        b, c, h, w = x_crop.size()
        self.model.model.zero_grad()
        idx = torch.argmax(cls)
        score = cls.reshape(-1)[idx]
        score.backward(retain_graph=retain_graph)

        gradients = self.gradients['value'].data
        activations = self.activations['value'].data
        b, k, u, v = activations.size()

        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
        activations = weights * activations

        score_saliency_map = torch.zeros((1, 1, h, w))

        if torch.cuda.is_available():
            activations = activations.cuda()
            score_saliency_map = score_saliency_map.cuda()

        masks = activations.chunk(self.groups, 1)
        with torch.no_grad():
            x_crop = x_crop / 255.0
            x_crop = torch.cat([x_crop[:, 2, :, :][:, None, :, :],x_crop[:, 1, :, :][:, None, :, :], x_crop[:, 0, :, :][:, None, :, :]],dim=1)
            norm_img = self.transform_norm(x_crop)
            blur_img = blur(norm_img)
            # img = self.Nutransform(blur_img)
            img = blur_img
            img = torch.cat([img[:, 2, :, :][:, None, :, :], img[:, 1, :, :][:, None, :, :], img[:, 0, :, :][:, None, :, :]], dim=1) * 255
            base_line = self.model.model.track(img)["cls"].reshape(-1)[idx]
            for saliency_map in masks:
                saliency_map = saliency_map.sum(1, keepdims=True)
                saliency_map = F.relu(saliency_map)
                threshold = np.percentile(saliency_map.cpu().numpy(), 70)
                saliency_map = torch.where(
                    saliency_map > threshold, saliency_map, torch.full_like(saliency_map, 0))
                saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

                if saliency_map.max() == saliency_map.min():
                    continue

                # normalize to 0-1
                norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

                # how much increase if keeping the highlighted region
                # predication on masked input
                blur_input = norm_img * norm_saliency_map + blur_img * (1 - norm_saliency_map)
                norm_img = self.transform_norm(blur_input)
                blur_img = blur(norm_img)
                img = blur_img
                img = torch.cat([img[:, 2, :, :][:, None, :, :], img[:, 1, :, :][:, None, :, :],
                                 img[:, 0, :, :][:, None, :, :]], dim=1) * 255
                outcls = self.model.model.track(img)["cls"].reshape(-1)[idx]
                score = outcls - base_line

                # score_saliency_map += score * saliency_map
                score_saliency_map += score * norm_saliency_map

        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None, None

        score_saliency_map = (score_saliency_map - score_saliency_map_min) / (
                score_saliency_map_max - score_saliency_map_min).data
        return score_saliency_map.cpu().data, x_crop.cpu().numpy()

    def __call__(self, input, class_idx=None, retain_graph=True):
        return self.forward(input, class_idx, retain_graph)
