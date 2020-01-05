# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth, OIMLoss
from .center_loss import CenterLoss


def make_loss(cfg, num_classes):    # modified by gu
    if cfg.MODEL.NAME == 'resnet18' or cfg.MODEL.NAME == 'resnet34':
        feat_dim = 512
    if cfg.MODEL.NAME == 'densenet169_ibn_a':
        feat_dim = 1664
    elif cfg.MODEL.NAME == 'densenet121_ibn_a':
        feat_dim = 1024
    else:
        feat_dim = 2048
    sampler = cfg.DATALOADER.SAMPLER
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_oim' or cfg.MODEL.METRIC_LOSS_TYPE == 'oim':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        if cfg.MODEL.IF_LABELSMOOTH == 'on':
            oim = OIMLoss(feat_dim=feat_dim, num_classes=num_classes, scalar=30.0, momentum=0.4,
                 label_smooth=True, epsilon=0.1, margin=cfg.SOLVER.OIM_MARGIN, weight=None)  # oim loss
            print("oim loss, label smooth on, numclasses:", num_classes)
        else:
            oim = OIMLoss(feat_dim=feat_dim, num_classes=num_classes, scalar=30.0, momentum=0.4,
                 label_smooth=False, margin=cfg.SOLVER.OIM_MARGIN, weight=None)  # oim loss
            print("oim loss, label smooth off, numclasses:", num_classes)
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif cfg.DATALOADER.SAMPLER == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, epoch=80):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, target) + triplet(feat, target)[0]
                else:
                    return F.cross_entropy(score, target) + triplet(feat, target)[0]
            elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_oim':
                oim_loss, oim_outputs = oim(score, target)
                return oim_loss + triplet(feat, target)[0], oim_outputs
            elif cfg.MODEL.METRIC_LOSS_TYPE == 'oim':
                return oim(score, target, epoch=epoch)
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func


def make_loss_with_center(cfg, num_classes):    # modified by gu
    if cfg.MODEL.NAME == 'resnet18' or cfg.MODEL.NAME == 'resnet34':
        feat_dim = 512
    else:
        feat_dim = 2048

    if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center_oim':
        oim = OIMLoss(feat_dim=feat_dim, num_classes=num_classes, scalar=30.0, momentum=0.5,
                 weight=None)  # oim loss
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True) # center loss

    else:
        print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    def loss_func(score, feat, target):
        if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return F.cross_entropy(score, target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, target) + \
                        triplet(feat, target)[0] + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return F.cross_entropy(score, target) + \
                        triplet(feat, target)[0] + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center_oim':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                print('Not implement\n')
            else:
                oim_loss, oim_outputs = oim(score, target)
                return oim_loss + \
                        triplet(feat, target)[0] + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target), oim_outputs
        
        else:
            print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
                  'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    return loss_func, center_criterion