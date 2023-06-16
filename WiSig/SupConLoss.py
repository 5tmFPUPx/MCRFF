"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
https://github.com/HobbitLong/SupContrast/blob/master/losses.py
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        features = features.view(features.shape[0], 1, features.shape[1]) # feature.shape=(batch_size,1,128).

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            # torch.eq 比较tensor的元素，相等就置1
            mask = torch.eq(labels, labels.T).float().to(device) #mask: 用于对比学习的mask，尺寸为 [batch_size, batch_size], 如果样本i和j属于同一个label，那么mask_{i,j}=1 
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1] # = number of views =1
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) #contrast_feature.shape=(batch_size,128)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count # =1
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature) # 计算所有两两样本间点乘相似度 anchor_dot_contrast.shape = batch_sizes * batch_sizes.
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # logits是anchor_dot_contrast减去每一行的最大值得到的最终相似度. 
        #每一行最大值就是对角线上（因为是自己和自己的相似度），所以logits对角线为0，其他值为负数。

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        ) # logits_mask就是对角线为0，其他全为1的矩阵，大小是batch_size * batch_size。
        mask = mask * logits_mask  # 对于计算Loss而言，(i,i)位置表示样本本身的相似度，对Loss是没用的，所以要mask掉
        # mask * logits_mask得到positive_mask（相比原来的mask，对角线置0，即去掉自己）

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask # exp(logits)是所有样本的相似度乘上所有样本的mask（去除自己本身）
        
        # exp_logits.sum(1, keepdim=True) 对exp_logits按行求和，即shape是（batchsize，1）。对应paper公式三的分母。
        # logits.shape =（batchsize，batchsize）的每一行元素分别减exp_logits.sum(1, keepdim=True)
        # log_prob.shape = (batch_size, batch_size)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6) 

        pos_per_sample = mask.sum(1) # sum: Returns the sum of all elements in the input tensor. 1表示方向，按行求和. 每类除自己外（每类的正样本）的个数。
        pos_per_sample[pos_per_sample < 1e-6] = 1.0

        # compute mean of log-likelihood over positive
        #mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_per_sample

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss