import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        # can only binary target
        # input: model output (batch_size, target_size (2), image_size, image_size)
        # target: teacher label (batch_size, image_size, image_size)
        # 
        # output: loss
        if len(target.size()) <= 3: target = F.one_hot(target, 2).permute(0, 3, 1, 2).float()
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = input.softmax(1).view(-1, )
        target = target.contiguous().view(-1, )
        intersection = (input * target).sum() # 各ピクセルの正解ラベルに対する予測確率
        dice = 1 - (2. * intersection.float() + smooth) / (input.sum() + target.sum() + smooth)
        return bce + dice

class WeightedBCELoss(nn.Module):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()

    def forward(self, input, target, weights=[0.01, 0.99]):
        # can only binary target
        # input: model output (batch_size, target_size (2), image_size, image_size)
        # target: teacher label (batch_size, image_size, image_size)
        # 
        # output: loss
        weights = torch.Tensor(weights).unsqueeze(0).unsqueeze(2).unsqueeze(3) # hard coded weight
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        weights = weights.to(device)

        if len(target.size()) <= 3: target = F.one_hot(target, 2).permute(0, 3, 1, 2).float()
        bce = F.binary_cross_entropy_with_logits(input, target, weights)
        return bce
        
class WeightedBCEDiceLoss(nn.Module):
    def __init__(self):
        super(WeightedBCEDiceLoss, self).__init__()

    def forward(self, input, target, weights=[0.1, 0.9]):
        # can only binary target
        # input: model output (batch_size, target_size (2), image_size, image_size)
        # target: teacher label (batch_size, image_size, image_size)
        # 
        # output: loss
        weights = torch.Tensor(weights).unsqueeze(0).unsqueeze(2).unsqueeze(3) # hard coded weight
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        weights = weights.to(device)

        if len(target.size()) <= 3: target = F.one_hot(target.long(), 2).permute(0, 3, 1, 2).float()
        bce = F.binary_cross_entropy_with_logits(input, target, weights)

        smooth = 1e-5
        input = input.softmax(1).view(-1, )
        target = target.contiguous().view(-1, )
        intersection = (input * target).sum() # 各ピクセルの正解ラベルに対する予測確率
        dice = 1 - (2. * intersection.float() + smooth) / (input.sum() + target.sum() + smooth)
        return bce + dice
        

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=1, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
    def forward(self, inputs, targets):
        targets = F.one_hot(targets, 2).permute(0, 3, 1, 2).float()
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        #print(F_loss)
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class FocalCosineLoss(nn.Module):
    def __init__(self, alpha=1, gamma=1, xent=.1):
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.xent = xent
        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        self.y = torch.Tensor([1]).to(device)

    def forward(self, input, target, reduction="mean"):
        target = F.one_hot(target, 2).permute(0, 3, 1, 2).float()
        cosine_loss = F.cosine_embedding_loss(input, target, self.y, reduction=reduction)
        cent_loss = F.binary_cross_entropy_with_logits(F.normalize(input), target, reduce=False)
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * cent_loss

        if reduction == "mean":
            focal_loss = torch.mean(focal_loss)

        return cosine_loss + self.xent * focal_loss


class LovaszHingeLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, inputs, targets):
        # inputs : (B, C, H, W)
        # targets : (B, H, W)
        inputs = inputs.softmax(1)
        if len(targets.size()) <= 3: targets = F.one_hot(targets.long(), 2).permute(0, 3, 1, 2).float()
        #targets = F.one_hot(targets, 2).permute(0, 3, 1, 2).float()   
        Lovasz = lovasz_softmax(inputs, targets, per_image=False)                       
        return Lovasz

def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.contiguous().view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss

def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss




def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        #fg = (labels == c).float() # foreground for class c, binary
        fg = labels[:,c].float() # soft label
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    # B, H, W, C -> P, C
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B, H, W, C -> P, C
    if labels.dim() == 3:
        # ここに来ちゃダメ
        labels = labels.contiguous().view(-1)
    else:
        # ターゲットだけに絞る？
        labels = labels.permute(0, 2, 3, 1).contiguous().view(-1, C)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)



def isnan(x):
    return x != x
    
    
def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n