# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
from torch import nn

import torch.nn.functional as F
from torch import nn, autograd

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        # margin=None
        self.margin = margin
        #print(margin)
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        # normalize_feature=True
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class OIM(autograd.Function):
    def __init__(self, lut, momentum=0.5):
        super(OIM, self).__init__()
        self.lut = lut
        self.momentum = momentum
    
    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs = inputs.mm(self.lut.t())
        return outputs
    
    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(self.lut)
        for x, y in zip(inputs, targets):
            self.lut[y] = self.momentum * self.lut[y] + (1. - self.momentum) * x
            self.lut[y] /= self.lut[y].norm()
        # return grad_inputs, None
        return grad_inputs.float(), None

def oim(inputs, targets, lut, momentum=0.5):
    return OIM(lut, momentum=momentum)(inputs, targets)


class OIMLoss(nn.Module):
    def __init__(self, feat_dim, num_classes, scalar=1.0, momentum=0.5, label_smooth=False,
                 epsilon=0.1, margin=0.1, weight=None, reduction='mean'):
        super(OIMLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.margin = margin
        self.momentum = momentum
        print('oim momentum:',self.momentum)
        self.scalar = scalar
        self.weight = weight
        self.reduction = reduction
        self.label_smooth = label_smooth
        if self.label_smooth:
            self.epsilon = epsilon
            self.logsoftmax = nn.LogSoftmax(dim=1)

        self.register_buffer('lut', torch.zeros(num_classes, feat_dim))
        self.lut = self.lut.cuda()
        
    def forward(self, inputs, targets, normalize_feature=True, epoch=80): 
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        # import pdb
        # pdb.set_trace()

        # inputs = oim(inputs, targets, self.lut, momentum=self.momentum)
        # momentum = min(self.momentum*epoch/60, self.momentum)
        momentum = self.momentum
        # print(momentum)
        inputs  = oim(inputs, targets, self.lut, momentum=momentum)

        ### add margin (11/1)
        phi = inputs - self.margin
        one_hot = torch.zeros(inputs.size(), device='cuda')
        one_hot.scatter_(1, targets.view(-1,1).long(), 1)
        inputs = (one_hot*phi) + ((1-one_hot)*inputs)

        inputs *= self.scalar
        ### add label smooth (11/4)
        if self.label_smooth:
            log_probs = self.logsoftmax(inputs)
            targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
            targets = targets.cuda()
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = F.cross_entropy(inputs, targets, weight=self.weight,
                                    reduction=self.reduction)
        # inputs *= self.scalar
        # loss = F.cross_entropy(inputs, targets, weight=self.weight,
        #                        reduction=self.reduction)
        return loss, inputs


'''
class OIM(autograd.Function):
    def __init__(self, lut, queue, index, momentum=0.5):
        super(OIM, self).__init__()
        self.lut = lut
        self.queue = queue
        self.momentum = momentum
        self.index = index

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs_labeled = inputs.mm(self.lut.t())
        outputs_unlabeled = inputs.mm(self.queue.t())
        return  torch.cat((outputs_labeled, outputs_unlabeled), 1)

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        # used=[]
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(torch.cat((self.lut, self.queue), 0))
        for x, y in zip(inputs, targets):
            if y<0 or y>=self.lut.size(0):
                self.queue[self.index, :] = x.view(1,-1)
                self.index = (self.index+1) % self.queue.size(0)
            else:
                # if y in used:
                #     continue
                # used.append(y)
                self.lut[y] = self.momentum * self.lut[y] + (1. - self.momentum) * x
                self.lut[y] /= self.lut[y].norm()
        return grad_inputs.float(), None


def oim(inputs, targets, lut, queue, index, momentum=0.5):
    return OIM(lut, queue, index, momentum=momentum)(inputs, targets)

class OIMLoss(nn.Module):
    def __init__(self, feat_dim, num_classes, queue_size=2, scalar=1.0, momentum=0.5,
                 label_smooth=False, epsilon=0.1, weight=None, reduction='mean', loss_weight=1.0):
        super(OIMLoss, self).__init__()
        self.feat_dim = feat_dim
        # num_classes = num_classes-1
        print(num_classes)
        self.num_classes = num_classes
        self.queue_size = queue_size
        self.momentum = momentum
        self.scalar = scalar
        self.index = 0
        self.loss_weight = loss_weight
        # if weight is None:
        #     self.weight = torch.cat([torch.ones(num_classes).cuda(), torch.zeros(queue_size).cuda()])
        # else:
        #     self.weight = weight
        self.reduction = reduction
        self.label_smooth = label_smooth
        if self.label_smooth:
            self.epsilon = epsilon
            self.logsoftmax = nn.LogSoftmax(dim=1)

        self.register_buffer('lut', torch.zeros(num_classes, feat_dim))
        self.register_buffer('queue', torch.zeros(queue_size, feat_dim))
        self.lut = self.lut.cuda()
        self.queue = self.queue.cuda()

    def forward(self, inputs, targets, normalize_feature=True, margin=0.0, epoch=80):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        # import pdb
        # pdb.set_trace()
        inputs = oim(inputs, targets, self.lut, self.queue, self.index, momentum=self.momentum)
        # targets[targets>=self.num_classes] = self.num_classes
        ### add margin (11/1)
        phi = inputs - margin
        one_hot = torch.zeros(inputs.size(), device='cuda')
        one_hot.scatter_(1, targets.view(-1,1).long(), 1)
        inputs = (one_hot*phi) + ((1-one_hot)*inputs)

        inputs *= self.scalar
        ### add label smooth (11/4)
        if self.label_smooth:
            log_probs = self.logsoftmax(inputs)
            targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
            targets = targets.cuda()
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
            loss = (- targets * log_probs).mean(0).sum()
        else:
            # import pdb
            # pdb.set_trace()
            # weight = torch.cat([torch.ones(self.num_classes), torch.zeros(self.queue_size)])
            # weight = torch.cat([torch.ones(4768), torch.zeros(self.num_classes+self.queue_size-4768)])
            # weight = torch.cat([torch.ones(9968), torch.zeros(self.num_classes+self.queue_size-9968)])
            weight = torch.cat([torch.ones(6987), torch.zeros(self.num_classes+self.queue_size-6987)])
            weight = weight.cuda() 
            loss = F.cross_entropy(inputs, targets, weight=weight, reduction=self.reduction, ignore_index=-1)
        #self.index = (self.index + torch.nonzero(targets<0) % self.queue_size
        self.index = (self.index + torch.nonzero(targets>=self.num_classes).size(0)) % self.queue_size
        return loss, inputs
'''

'''
#### 12.27 AE
class OIM(autograd.Function):
    def __init__(self, lut, momentum=0.5):
        super(OIM, self).__init__()
        self.lut = lut
        self.momentum = momentum
    
    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs = inputs.mm(self.lut.t())
        return outputs
    
    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(self.lut)
        for x, y in zip(inputs, targets):
            self.lut[y] = self.momentum * self.lut[y] + (1. - self.momentum) * x
            self.lut[y] /= self.lut[y].norm()
        return grad_inputs, None
        # return grad_inputs.float(), None

def oim(inputs, targets, lut, momentum=0.5):
    return OIM(lut, momentum=momentum)(inputs, targets)


class OIMLoss(nn.Module):
    def __init__(self, feat_dim, num_classes, num_classes_extra=0, scalar=1.0, momentum=0.5, 
                delta = 3.5, xi = 0.2, lambda0=0.55, label_smooth=False, epsilon=0.1, weight=None, 
                reduction='mean'):
        super(OIMLoss, self).__init__()
        num_classes = 6987
        # num_classes_extra = 24905
        num_classes_extra = 1511
        print('num_class_extra:',num_classes_extra)
        print('num_class:',num_classes)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.momentum = momentum
        self.scalar = scalar
        self.weight = weight
        self.reduction = reduction
        self.label_smooth = label_smooth
        if self.label_smooth:
            self.epsilon = epsilon
            self.logsoftmax = nn.LogSoftmax(dim=1)
        
        self.num_classes_extra = num_classes_extra
        self.delta = delta
        self.xi = xi
        self.lambda0 = lambda0
        if self.num_classes_extra>0:
            self.register_buffer('em', torch.zeros(num_classes_extra, feat_dim))
            self.em = self.em.cuda()

        self.register_buffer('lut', torch.zeros(num_classes, feat_dim))
        self.lut = self.lut.cuda()
    
    def adaptive_selection(self, inputs_extra, targets_extra):
        ###minimize distances between all person images
        targets_onehot = (inputs_extra > self.lambda0*self.scalar).float()
        ks = (targets_onehot.sum(dim=1)).float()
        ks1 = ks.cpu()
        ks_mask = (ks > 1).float()
        ks = ks * ks_mask + (1 - ks_mask) * 2
        ks = self.delta / (ks * torch.log(ks))
        ks = (ks * ks_mask).view(-1,1)
        targets_onehot = targets_onehot * ks
       
        ###maximize distances between similar person images
        targets_extra = torch.unsqueeze(targets_extra, 1)
        targets_onehot.scatter_(1, targets_extra, float(1))
        
        return targets_onehot, ks1

    def target_loss(self, inputs_extra, targets_extra):
        targets_line, ks = self.adaptive_selection(inputs_extra.detach().clone(),targets_extra.detach().clone())
        outputs = F.log_softmax(inputs_extra, dim=1)
        loss = -(targets_line*outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)
        return loss, ks

    def forward(self, inputs, targets, normalize_feature=True, margin=0.2, epoch=80): 
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)

        # inputs = oim(inputs, targets, self.lut, momentum=self.momentum)
        # momentum = min(self.momentum*epoch/60, self.momentum)
        momentum = self.momentum
        # print(momentum)
        # import pdb
        # pdb.set_trace()

        index_extra = torch.nonzero(targets>=self.num_classes).view(-1)
        inputs_extra = inputs[index_extra,:]
        targets_extra = targets[index_extra] - self.num_classes

        index_train = torch.nonzero(targets<self.num_classes).view(-1)
        inputs = inputs[index_train,:]
        targets = targets[index_train]

        #### input train:
        inputs  = oim(inputs, targets, self.lut, momentum=momentum)

        ### add margin (11/1)
        phi = inputs - margin
        one_hot = torch.zeros(inputs.size(), device='cuda')
        one_hot.scatter_(1, targets.view(-1,1).long(), 1)
        inputs = (one_hot*phi) + ((1-one_hot)*inputs)

        inputs *= self.scalar
        ### add label smooth (11/4)
        if self.label_smooth:
            log_probs = self.logsoftmax(inputs)
            targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
            targets = targets.cuda()
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = F.cross_entropy(inputs, targets, weight=self.weight,
                                    reduction=self.reduction)
        
        if len(targets_extra)==0:
            return loss, inputs
        #### input extra:
        inputs_extra = oim(inputs_extra, targets_extra, self.em, momentum=momentum)
        inputs_extra *= self.scalar
        if epoch>4:
            loss_extra, ks = self.target_loss(inputs_extra, targets_extra)
        else:
            loss_extra = F.cross_entropy(inputs_extra, targets_extra)
            ks = torch.FloatTensor(128).zero_()
        
        loss = (1-self.xi)*loss + self.xi*loss_extra

        return loss, inputs
'''


'''
#### 12.28 ME
class OIM(autograd.Function):
    def __init__(self, lut, em, k, num_classes, momentum=0.5):
        super(OIM, self).__init__()
        self.lut = lut
        self.momentum = momentum

        self.em = em
        self.k = k
        self.num_classes = num_classes
    
    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs = inputs.mm(self.lut.t())

        # import pdb
        # pdb.set_trace()
        temp = inputs.mm(self.em.t())
        value,_ = torch.topk(temp, self.k, dim=1)
        # outputs = torch.cat((outputs,value), dim=1)

        return outputs, value
    
    def backward(self, grad_outputs, grad_value):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(self.lut)
        for x, y in zip(inputs, targets):
            if y<self.num_classes:
                self.lut[y] = self.momentum * self.lut[y] + (1. - self.momentum) * x
                self.lut[y] /= self.lut[y].norm()
            else:
                self.em[y-self.num_classes, :] = x.view(1,-1)
        return grad_inputs, None
        # return grad_inputs.float(), None

def oim(inputs, targets, lut, em, k, num_classes, momentum=0.5):
    return OIM(lut, em, k, num_classes, momentum=momentum)(inputs, targets)


class OIMLoss(nn.Module):
    def __init__(self, feat_dim, num_classes, num_classes_extra=0, scalar=1.0, momentum=0.5, 
                k=15, lambda0=0.55, label_smooth=False, epsilon=0.1, weight=None, 
                reduction='mean'):
        super(OIMLoss, self).__init__()
        num_classes = 6987
        num_classes_extra = 24905
        print('num_class_extra:',num_classes_extra)
        print('num_class:',num_classes)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.momentum = momentum
        self.scalar = scalar
        self.weight = weight
        self.reduction = reduction
        self.label_smooth = label_smooth
        if self.label_smooth:
            self.epsilon = epsilon
            self.logsoftmax = nn.LogSoftmax(dim=1)
        
        self.num_classes_extra = num_classes_extra
        self.k = k
        self.lambda0 = lambda0
        if self.num_classes_extra>0:
            self.register_buffer('em', torch.zeros(num_classes_extra, feat_dim))
            self.em = self.em.cuda()

        self.register_buffer('lut', torch.zeros(num_classes, feat_dim))
        self.lut = self.lut.cuda()

    def forward(self, inputs, targets, normalize_feature=True, margin=0.2, epoch=80): 
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)

        # inputs = oim(inputs, targets, self.lut, momentum=self.momentum)
        # momentum = min(self.momentum*epoch/60, self.momentum)
        momentum = self.momentum
        # print(momentum)
        # import pdb
        # pdb.set_trace()

        #### input train:
        inputs, value  = oim(inputs, targets, self.lut, self.em, self.k, self.num_classes, momentum=momentum)
        inputs = torch.cat((inputs, value), dim=1)

        index_train = torch.nonzero(targets<self.num_classes).view(-1)
        inputs = inputs[index_train,:]
        targets = targets[index_train]

        ### add margin (11/1)
        phi = inputs - margin
        one_hot = torch.zeros(inputs.size(), device='cuda')
        one_hot.scatter_(1, targets.view(-1,1).long(), 1)
        inputs = (one_hot*phi) + ((1-one_hot)*inputs)

        inputs *= self.scalar
        ### add label smooth (11/4)
        if self.label_smooth:
            log_probs = self.logsoftmax(inputs)
            targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
            targets = targets.cuda()
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = F.cross_entropy(inputs, targets, weight=self.weight,
                                    reduction=self.reduction)
        
        return loss, inputs
'''