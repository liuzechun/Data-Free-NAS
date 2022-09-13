'''This code is modified from https://github.com/liuzechun/ReActNet/blob/master/utils/KD_loss.py'''

import torch
from torch.nn import functional as F
from torch.nn.modules import loss


class DistributionLoss(loss._Loss):
    """The KL-Divergence loss for the SuperNet (student) model and teacher model output."""

    def forward(self, model_output, real_output):

        self.size_average = True

        # Target is ignored at training time. Loss is defined as KL divergence
        # between the model output and the refined labels.
        if real_output.requires_grad:
            raise ValueError("real network output should not require gradients.")

        model_output_log_prob = F.log_softmax(model_output/3, dim=1)
        real_output_soft = F.softmax(real_output/3, dim=1)
        del model_output, real_output

        # Loss is -dot(model_output_log_prob, real_output). Prepare tensors
        # for batch matrix multiplicatio
        real_output_soft = real_output_soft.unsqueeze(1)
        model_output_log_prob = model_output_log_prob.unsqueeze(2)

        # Compute the loss, and average/sum for the batch.
        cross_entropy_loss = -torch.bmm(real_output_soft, model_output_log_prob)
        if self.size_average:
             cross_entropy_loss = cross_entropy_loss.mean()
        else:
             cross_entropy_loss = cross_entropy_loss.sum()

        return cross_entropy_loss
