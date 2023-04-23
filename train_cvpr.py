import utils.crl_utils
from utils import utils
import torch.nn as nn
import time
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import random


class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        q = torch.softmax(target/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss
kdloss = KDLoss(2.0)


def classAug(args, x, y, base_numclass, alpha=10.0, mix_times=2):
    batch_size = x.size()[0]
    mix_data = []
    mix_target = []
    new_classnum = args.newclassnum
    for _ in range(mix_times):
        index = torch.randperm(batch_size).cuda()
        for i in range(batch_size):
            if y[i] != y[index][i]:
                new_label = generate_label(y[i].item(), y[index][i].item(), base_numclass, new_classnum)
                lam = np.random.beta(alpha, alpha)
                if lam < 0.4 or lam > 0.6:
                    lam = 0.5
                mix_data.append(lam * x[i] + (1 - lam) * x[index, :][i])
                mix_target.append(new_label)

    new_target = torch.Tensor(mix_target)
    y = torch.cat((y, new_target.cuda().long()), 0)
    for item in mix_data:
        x = torch.cat((x, item.unsqueeze(0)), 0)
    return x, y


def generate_label(y_a, y_b, base_numclass=10, new_classnum=1):
    y_a, y_b = y_a, y_b
    assert y_a != y_b
    if y_a > y_b:
        tmp = y_a
        y_a = y_b
        y_b = tmp
    label_index = ((2 * base_numclass - y_a - 1) * y_a) / 2 + (y_b - y_a) - 1
    return (label_index % new_classnum) + base_numclass


def mixup_data(x, y, alpha=0.3, use_cuda=True):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def RegMixup_data(x, y, alpha=10, use_cuda=True):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def OE_mixup(x_in, x_out, alpha=10.0):
    if x_in.size()[0] != x_out.size()[0]:
        length = min(x_in.size()[0], x_out.size()[0])
        x_in = x_in[:length]
        x_out = x_out[:length]
    lam = np.random.beta(alpha, alpha)
    x_oe = lam * x_in + (1 - lam) * x_out
    return x_oe, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(loader, loader_out, model, criterion, criterion_ranking, optimizer, epoch, history, logger, args):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    total_losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    end = time.time()
    model.train()

    if args.method == 'OpenMix':
        for in_set, out_set in zip(loader, loader_out):

            in_data, out_data, target = in_set[0].cuda(), out_set[0].cuda(), in_set[1].cuda()
            in_oe, lam = OE_mixup(in_data, out_data)
            inputs = torch.cat([in_data, in_oe], dim=0)
            logits = model(inputs)
            target_oe = torch.LongTensor(in_oe.shape[0]).random_(args.classnumber, args.classnumber + 1).cuda()
            loss_in = F.cross_entropy(logits[:in_data.shape[0]], target)
            loss_oe = lam * F.cross_entropy(logits[in_data.shape[0]:], target[:target_oe.shape[0]]) \
                      + (1 - lam) * F.cross_entropy(logits[in_data.shape[0]:], target_oe)
            loss = loss_in + args.lambda_o * loss_oe
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec, correct = utils.accuracy(logits[:in_data.shape[0]], target)
            total_losses.update(loss.item(), logits[:in_data.shape[0]].size(0))
            top1.update(prec.item(), logits[:in_data.shape[0]].size(0))

            batch_time.update(time.time() - end)
            end = time.time()
        logger.write([epoch, total_losses.avg, top1.avg])

    elif args.method == 'OpenMix_manifold':
        for in_set, out_set in zip(loader, loader_out):
            in_data, out_data, target = in_set[0].cuda(), out_set[0].cuda(), in_set[1].cuda()
            inputs = torch.cat([in_data, out_data], dim=0)
            logits, lam = model.forward_manifold(inputs, length_size=in_data.shape[0])
            logits_in = logits[:in_data.shape[0]]
            logits_oe = logits[in_data.shape[0]:]

            target_oe = torch.LongTensor(logits_oe.shape[0]).random_(args.classnumber, args.classnumber + 1).cuda()
            loss_in = F.cross_entropy(logits_in, target)
            loss_oe = lam * F.cross_entropy(logits_oe, target[:target_oe.shape[0]]) \
                      + (1 - lam) * F.cross_entropy(logits_oe, target_oe)
            loss = loss_in + args.lambda_o * loss_oe
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec, correct = utils.accuracy(logits[:in_data.shape[0]], target)
            total_losses.update(loss.item(), logits[:in_data.shape[0]].size(0))
            top1.update(prec.item(), logits[:in_data.shape[0]].size(0))

            batch_time.update(time.time() - end)
            end = time.time()
        logger.write([epoch, total_losses.avg, top1.avg])

    elif args.method == 'OpenMix-CRL':
        for in_set, out_set in zip(loader, loader_out):
            in_data, out_data, target = in_set[0].cuda(), out_set[0].cuda(), in_set[1].cuda()
            in_oe, lam = OE_mixup(in_data, out_data)
            inputs = torch.cat([in_data, in_oe], dim=0)
            logits = model(inputs)
            output = logits[:in_data.shape[0]]
            conf = F.softmax(output, dim=1)
            confidence, _ = conf.max(dim=1)
            idx = in_set[2]
            rank_input1 = confidence
            rank_input2 = torch.roll(confidence, -1)
            idx2 = torch.roll(idx, -1)
            rank_target, rank_margin = history.get_target_margin(idx, idx2)
            rank_target_nonzero = rank_target.clone()
            rank_target_nonzero[rank_target_nonzero == 0] = 1
            rank_input2 = rank_input2 + rank_margin / rank_target_nonzero
            ranking_loss = criterion_ranking(rank_input1, rank_input2, rank_target)
            ranking_loss = args.rank_weight * ranking_loss
            target_oe = torch.LongTensor(in_oe.shape[0]).random_(args.classnumber, args.classnumber + 1).cuda()
            loss_in = F.cross_entropy(logits[:in_data.shape[0]], target)
            loss_oe = lam * F.cross_entropy(logits[in_data.shape[0]:], target[:target_oe.shape[0]]) \
                      + (1 - lam) * F.cross_entropy(logits[in_data.shape[0]:], target_oe)
            loss = loss_in + args.lambda_o * loss_oe + ranking_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec, correct = utils.accuracy(logits[:in_data.shape[0]], target)
            total_losses.update(loss.item(), logits[:in_data.shape[0]].size(0))
            top1.update(prec.item(), logits[:in_data.shape[0]].size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            history.correctness_update(idx, correct, output)
        history.max_correctness_update(epoch)
        logger.write([epoch, total_losses.avg, top1.avg])

    elif args.method == 'OE':
        for in_set, out_set in zip(loader, loader_out):
            in_data, out_data, target = in_set[0].cuda(), out_set[0].cuda(), in_set[1].cuda()
            inputs = torch.cat([in_data, out_data], dim=0)
            logits = model(inputs)
            loss_in = F.cross_entropy(logits[:in_data.shape[0]], target)
            loss_oe = 0.5 * (-(logits[in_data.shape[0]:].mean(1) - torch.logsumexp(logits[in_data.shape[0]:], dim=1)).mean())

            loss = loss_in + loss_oe
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec, correct = utils.accuracy(logits[:in_data.shape[0]], target)
            total_losses.update(loss.item(), logits[:in_data.shape[0]].size(0))
            top1.update(prec.item(), logits[:in_data.shape[0]].size(0))

            batch_time.update(time.time() - end)
            end = time.time()
        logger.write([epoch, total_losses.avg, top1.avg])
    else:
        for i, (input, target, idx) in enumerate(loader):
            data_time.update(time.time() - end)
            if args.method == 'Baseline':
                input, target = input.cuda(), target.long().cuda()
                output = model(input)
                loss = criterion(output, target)
            elif args.method == 'classAug':
                input, target = input.cuda(), target.long().cuda()
                input, target = classAug(args, input, target, base_numclass=args.classnumber)
                output = model(input)
                loss = criterion(output, target)
            elif args.method == 'Mixup':
                input, target = input.cuda(), target.long().cuda()
                input, target_a, target_b, lam = mixup_data(input, target)
                output = model(input)
                loss = mixup_criterion(criterion, output, target_a, target_b, lam)
            elif args.method == 'Manifold':
                input, target = input.cuda(), target.long().cuda()
                output, target_b, lam = model.forward_manimix(input, target)
                loss = mixup_criterion(criterion, output, target, target_b, lam)
            elif args.method == 'RegMixup':
                input, target = input.cuda(), target.long().cuda()
                input_mix, target_a, target_b, lam = RegMixup_data(input, target)
                inputs_all = torch.cat([input, input_mix], dim=0)
                output = model(inputs_all)
                loss_original = criterion(output[:input.shape[0]], target)
                loss_Mixup = mixup_criterion(criterion, output[input.shape[0]:], target_a, target_b, lam)
                loss = loss_original + loss_Mixup
                output = output[:input.shape[0]]
            elif args.method == 'CRL':
                input, target = input.cuda(), target.long().cuda()
                output = model(input)
                conf = F.softmax(output, dim=1)
                confidence, _ = conf.max(dim=1)

                rank_input1 = confidence
                rank_input2 = torch.roll(confidence, -1)
                idx2 = torch.roll(idx, -1)

                rank_target, rank_margin = history.get_target_margin(idx, idx2)
                rank_target_nonzero = rank_target.clone()
                rank_target_nonzero[rank_target_nonzero == 0] = 1
                rank_input2 = rank_input2 + rank_margin / rank_target_nonzero

                ranking_loss = criterion_ranking(rank_input1,rank_input2,rank_target)
                cls_loss = criterion(output, target)
                ranking_loss = args.rank_weight * ranking_loss
                loss = cls_loss + ranking_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prec, correct = utils.accuracy(output, target)
            total_losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            if args.method == 'CRL':
                history.correctness_update(idx, correct, output)
        if args.method == 'CRL':
            history.max_correctness_update(epoch)
        logger.write([epoch, total_losses.avg, top1.avg])
