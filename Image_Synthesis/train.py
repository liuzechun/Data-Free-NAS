'''This code is modified from https://github.com/NVlabs/DeepInversion/blob/master/deepinversion.py'''

import os
import sys
import shutil
import numpy as np
import time, datetime
import torch
import random
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
import torch.optim as optim
import torchvision.models as models
import collections
import random


from utils import *
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
from resnet import build_resnet

class BNFeatureHook():

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)


    def hook_fn(self, module, input, output):

        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().reshape([nch, -1]).var(1, unbiased=False)
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)
        self.r_feature = r_feature

    def close(self):
        self.hook.remove()

def get_image_prior_losses(inputs_jit):

    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0

    return loss_var_l1, loss_var_l2


def get_images(gpu_id, num_generations, args, model_student, model_teacher, criterion, criterion_recur, hook_for_display, jitter=32, targets_in=None):
    print("get_images call")

    model_student.eval()
    save_every = 100
    batch_size = args.batch_size

    kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()
    best_cost = 1e4

    loss_r_feature_layers = []
    for module in model_teacher.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(BNFeatureHook(module))

    # setup target labels
    if targets_in == None:
        targets_all = torch.LongTensor(np.random.permutation(1000))
    else:
        targets_all = targets_in

    targets_new = None

    for kk in range(0, 1000, batch_size):

        targets = targets_all[kk:min(kk+batch_size,1000)].to('cuda')

        img_resize = 256
        img_crop = 224

        data_type = torch.float
        inputs = torch.randn((targets.shape[0], 3, img_resize, img_resize), requires_grad=True, device='cuda',
                             dtype=data_type)
        pooling_function = nn.modules.pooling.AvgPool2d(kernel_size=2)

        skipfirst = False

        iteration = 0
        for lr_it, lower_res in enumerate([2, 1]):
            if lr_it==0:
                iterations_per_layer = 2000
            else:
                iterations_per_layer = 3000

            lim_0, lim_1 = jitter // lower_res, jitter // lower_res
            img_size = img_crop // lower_res

            optimizer = optim.Adam([inputs], lr=args.lr, betas=[0.5, 0.9], eps = 1e-8)

            lr_scheduler = lr_cosine_policy(args.lr, 100, iterations_per_layer)

            for iteration_loc in range(iterations_per_layer):
                iteration += 1
                # learning rate scheduling
                lr_scheduler(optimizer, iteration_loc, iteration_loc)

                # perform downsampling if needed
                if lower_res != 1:
                    inputs_jit0 = pooling_function(inputs)
                else:
                    inputs_jit0 = inputs

                # apply random jitter offsets
                off1 = random.randint(0, lim_0)
                off2 = random.randint(0, lim_1)
                inputs_jit = inputs_jit0[:, :, off1 : off1+img_size, off2 : off2+img_size]


                # Flipping
                flip = random.random() > 0.5
                inputs_jit = torch.flip(inputs_jit, dims=(3,))

                # forward pass
                optimizer.zero_grad()
                model_teacher.zero_grad()

                outputs = model_teacher(inputs_jit)

                # R_cross classification loss
                if targets_in == None:
                    loss = criterion(outputs, targets)
                else:
                    loss = criterion_recur(outputs, targets)

                # R_prior losses
                loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)

                # R_feature loss

                rescale = [args.first_bn_multiplier] + [1. for _ in range(len(loss_r_feature_layers)-1)]
                loss_r_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])


                # l2 loss on images
                loss_l2 = torch.norm(inputs_jit.reshape(batch_size, -1), dim=1).mean()

                # combining losses
                loss_aux = args.tv_l2 * loss_var_l2 + \
                           args.tv_l1 * loss_var_l1 + \
                           args.r_feature * loss_r_feature + \
                           args.l2_scale * loss_l2

                loss = args.main_loss_multiplier * loss + loss_aux

                if iteration % save_every==0:
                    print("------------iteration {}----------".format(iteration))
                    print("total loss", loss.item())
                    print("loss_r_feature", loss_r_feature.item())
                    if targets_in == None:
                        print("main criterion", criterion(outputs, targets).item())
                        if hook_for_display is not None:
                            hook_for_display(inputs, targets)
                    else:
                        print("main criterion", criterion_recur(outputs, targets).item())
                        targets_max, indices = targets.max(1)
                        if hook_for_display is not None:
                            hook_for_display(inputs, indices)

                # do image update
                loss.backward()

                optimizer.step()

                # clip color outlayers
                inputs.data = clip(inputs.data)

                if best_cost > loss.item() or iteration == 1:
                    best_inputs = inputs.data.clone()

        if args.store_best_images:
            best_inputs = denormalize(best_inputs)
            save_images(args, gpu_id, num_generations, best_inputs, targets)

        logits = model_teacher(inputs_jit)
        if targets_new == None:
            targets_new = logits.data
        else:
            targets_new = torch.cat((targets_new, logits.data), 0)

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)

    torch.cuda.empty_cache()
    return targets_new

def save_images(args, gpu_id, num_generations, images, targets):
    # method to store generated images locally
    local_rank = torch.cuda.current_device()
    for id in range(images.shape[0]):
        if targets.ndimension() == 1:
            class_id = targets[id].item()
        else:
            class_id = targets[id].argmax().item()
        #if 0:
        if not os.path.exists(args.final_data_path):
            os.mkdir(args.final_data_path)
        store_path = args.final_data_path + '/new_class_' + str(class_id) + '/'
        if not os.path.exists(store_path):
            os.mkdir(store_path)

        place_to_store = store_path + 'ei_' + str(gpu_id) + str(num_generations) + '_' + str(id) + '.jpg'

        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)

def validate(input, target, model):

    def accuracy(output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    with torch.no_grad():
        output = model(input)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

    print("Verifier accuracy: ", prec1.item())

def main_syn(targets_new=None):

    parser = argparse.ArgumentParser("resnet18")
    parser.add_argument('-s', '--worldsize', type=int, default=1, help='Number of processes participating in the job.')
    parser.add_argument('--local_rank', '--rank', type=int, default=0, help='Rank of the current process.')
    parser.add_argument('--jitter', default=32, type=int, help='random shift')
    parser.add_argument('--comment', default='', type=str, help='batch size')
    parser.add_argument('--arch_name', default='resnet50', type=str, help='model name from torchvision or resnet50v15')

    parser.add_argument('--fp16', action='store_true', help='use FP16 for optimization')
    parser.add_argument('--exp_name', type=str, default='test', help='where to store experimental data')

    parser.add_argument('--verifier', action='store_true', help='evaluate batch with another model')
    parser.add_argument('--verifier_arch', type=str, default='mobilenet_v2', help = "arch name from torchvision models to act as a verifier")

    parser.add_argument('--r_feature', type=float, default=0.05, help='coefficient for feature distribution regularization')
    parser.add_argument('--first_bn_multiplier', type=float, default=10., help='additional multiplier on first bn layer of R_feature')
    parser.add_argument('--tv_l1', type=float, default=0.0, help='coefficient for total variation L1 loss')
    parser.add_argument('--tv_l2', type=float, default=0.0001, help='coefficient for total variation L2 loss')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate for optimization')
    parser.add_argument('--l2_scale', type=float, default=0.00001, help='l2 loss on the image')
    parser.add_argument('--main_loss_multiplier', type=float, default=1.0, help='coefficient for the main loss in optimization')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=256, help='num of training epochs')
    parser.add_argument('--final_data_path', type=str, default='../final_images', help='num of training epochs')
    parser.add_argument('--store_best_images', action='store_true', help='save best images as separate files')
    args = parser.parse_args()


    model_teacher = build_resnet("resnet50", "classic", 1000)
    #model_teacher = models.__dict__[args.arch_name](pretrained=True)
    #model_teacher = model_teacher.cuda()
    model_teacher = nn.DataParallel(model_teacher).cuda()
    model_teacher.eval()

    checkpoint_tar = './models/checkpoint.pth.tar'
    checkpoint = torch.load(checkpoint_tar)
    model_teacher.load_state_dict(checkpoint['state_dict'])

    model_student = models.__dict__['resnet18'](pretrained=True)
    model_student = model_student.cuda()
    model_student.eval()

    model_verifier = models.__dict__[args.verifier_arch](pretrained=True)
    model_verifier = model_verifier.cuda()
    model_verifier.eval()

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    #criterion_smooth = CrossEntropyLabelSmooth(1000, 0.1)
    #criterion_smooth = criterion_smooth.cuda()
    criterion_recur = CrossEntropyRecursiveLabel()
    criterion_recur = criterion_recur.cuda()

    exp_name = "generations/%s"%args.exp_name
    hook_for_display = lambda x,y: validate(x, y, model_verifier)

    targets_new = get_images(0, i, args, model_student, model_teacher, criterion, criterion_recur, hook_for_display, targets_in=targets_new)

    return targets_new

if __name__ == '__main__':
    i = 0
    targets_new = main_syn()
    for i in range(1,10):
        print('epoch:', i)
        targets_new = main_syn(targets_new)



