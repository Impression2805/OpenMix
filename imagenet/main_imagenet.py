import argparse
import os
import shutil
import time
import csv
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim

import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import resource
from collections import OrderedDict
import random
import metrics_imagenet

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
from ImageFolder import *

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


def csv_writter(path, dic, start):
    if os.path.isdir(path) == False: os.makedirs(path)
    os.chdir(path)
    # Write dic
    if start == 1:
        mode = 'w'
    else:
        mode = 'a'
    with open('logs.csv', mode) as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        if start == 1:
            writer.writerow(dic.keys())
        writer.writerow([elem["string"] for elem in dic.values()])


def one_hot_encoding(label):
    print("one_hot_encoding process")
    cls = set(label)
    class_dict = {c: np.identity(len(cls))[i, :] for i, c in enumerate(cls)}
    one_hot = np.array(list(map(class_dict.get, label)))
    return one_hot


def OE_mixup(x_in, x_out, alpha=10.0):
    if x_in.size()[0] != x_out.size()[0]:
        length = min(x_in.size()[0], x_out.size()[0])
        x_in = x_in[:length]
        x_out = x_out[:length]
    lam = np.random.beta(alpha, alpha)
    x_oe = lam * x_in + (1 - lam) * x_out
    return x_oe, lam


class Counter(dict):
    def __missing__(self, key):
        return None


def fast_collate(batch, memory_format):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if (nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets


def parse():
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                        metavar='N', help='mini-batch size per process (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR',
                        help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=60, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')

    parser.add_argument('--prof', default=-1, type=int,
                        help='Only run 10 iterations for profiling.')
    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', 0), type=int)
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling apex sync BN.')

    parser.add_argument('--opt-level', type=str)
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--channels-last', type=bool, default=False)

    ############################### our modification ##############################
    parser.add_argument('--save_path', default='./imagenet_cvpr/', type=str, help='Savefiles directory')
    parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')
    parser.add_argument('--method', default='OpenMix', type=str,
                        help='[OpenMix, Baseline, Mixup]')
    parser.add_argument('--plot', default=1, type=int, help='')
    parser.add_argument('-lambda_o', type=float, default=1, help='[0.1, 0.5, 1.0, 1.5, 2] dnl loss weight')
    parser.add_argument('--train_dir', default='/lustre/home/fzhu/imagenet_classaug/train', type=str, help='train_dir')
    parser.add_argument('--test_dir', default='/lustre/home/fzhu/imagenet_classaug/val', type=str, help='test_dir')
    parser.add_argument('--num_class', default=500, type=int, metavar='n',
                        help='number of training classes')
    parser.add_argument('--new_class', default=1, type=int, metavar='n',
                        help='number of training classes')
    args = parser.parse_args()
    return args


def save_checkpoint(state, is_best, save_path, epoch):
    filename = save_path + '/checkpoint' + '.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, save_path + '/model_best.pth.tar')


def main():
    global best_prec1, args
    args = parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
    print(args.method)
    print("opt_level = {}".format(args.opt_level))
    print("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))
    print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))

    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    cudnn.benchmark = True
    save_path = args.save_path + 'imagenet' + '_' + args.arch + '_' + args.method + '_' + str(args.num_class)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    best_prec1 = 0
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    # args.gpu = 4
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    if args.channels_last:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format

    # create model
    if args.method == 'OpenMix':
        classnum = args.num_class + 1
    else:
        classnum = args.num_class
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=classnum)

    if args.sync_bn:
        import apex
        print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    model = model.cuda().to(memory_format=memory_format)

    # Scale learning rate based on global batch size
    args.lr = args.lr * float(args.batch_size * args.world_size) / 256.
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale
                                      )
    if args.distributed:
        model = DDP(model, delay_allreduce=True)

    criterion = nn.CrossEntropyLoss().cuda()

    # if args.resume:
    path_reuse = save_path + '/checkpoint.pth.tar'

    def resume():
        if os.path.isfile(path_reuse):
            print("=> loading checkpoint '{}'".format(path_reuse))
            # checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
            checkpoint = torch.load(path_reuse, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            global best_prec1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(path_reuse, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(path_reuse))

    resume()

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    crop_size = 224
    val_size = 256
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        # normalize,
    ])

    transform_test = transforms.Compose([
        transforms.Resize(val_size),
        transforms.CenterCrop(crop_size),
        # transforms.ToTensor(),
        # normalize,
    ])
    class_index = list(range(args.num_class))
    train_dataset = ImageFolder(args.train_dir, transform_train, index=class_index)
    val_dataset = ImageFolder(args.test_dir, transform_test, index=class_index)
    class_index_ood = list(range(args.num_class, 2 * args.num_class))
    train_dataset_ood = ImageFolder(args.train_dir, transform_train, index=class_index_ood)

    train_sampler = None
    val_sampler = None
    train_sampler_ood = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        train_sampler_ood = torch.utils.data.distributed.DistributedSampler(train_dataset_ood)

    collate_fn = lambda b: fast_collate(b, memory_format)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                               collate_fn=collate_fn)

    train_loader_ood = torch.utils.data.DataLoader(train_dataset_ood,
                                                   batch_size=args.batch_size,
                                                   shuffle=(train_sampler_ood is None),
                                                   num_workers=args.workers, pin_memory=True, sampler=train_sampler_ood,
                                                   collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers, pin_memory=True,
                                             sampler=val_sampler,
                                             collate_fn=collate_fn)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, train_loader_ood, model, criterion, optimizer, epoch)

        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, save_path, epoch)


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target


def train(train_loader, train_loader_ood, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    end = time.time()

    if args.method == 'OpenMix':
        prefetcher = data_prefetcher(train_loader)
        prefetcher_ood = data_prefetcher(train_loader_ood)
        input_in, target = prefetcher.next()
        input_ood, _ = prefetcher_ood.next()
        i = 0
        while input_in is not None:
            if input_ood is not None:
                in_ood_mix, lam = OE_mixup(input_in, input_ood)
                i += 1
                if args.prof >= 0 and i == args.prof:
                    print("Profiling begun at iteration {}".format(i))
                    torch.cuda.cudart().cudaProfilerStart()
                if args.prof >= 0: torch.cuda.nvtx.range_push("Body of iteration {}".format(i))

                adjust_learning_rate(optimizer, epoch, i, len(train_loader))
                if args.prof >= 0: torch.cuda.nvtx.range_push("forward")

                inputs = torch.cat([input_in, in_ood_mix], dim=0)
                logits = model(inputs)
                output = logits[:input_in.shape[0]]
                target_oe = torch.LongTensor(in_ood_mix.shape[0]).random_(args.num_class, args.num_class + 1).cuda()
                if args.prof >= 0: torch.cuda.nvtx.range_pop()
                loss_in = F.cross_entropy(output, target)
                loss_oe = lam * F.cross_entropy(logits[input_in.shape[0]:], target[:in_ood_mix.shape[0]]) \
                          + (1 - lam) * F.cross_entropy(logits[input_in.shape[0]:], target_oe)
                loss = loss_in + args.lambda_o * loss_oe
            else:
                i += 1
                if args.prof >= 0 and i == args.prof:
                    print("Profiling begun at iteration {}".format(i))
                    torch.cuda.cudart().cudaProfilerStart()
                if args.prof >= 0: torch.cuda.nvtx.range_push("Body of iteration {}".format(i))

                adjust_learning_rate(optimizer, epoch, i, len(train_loader))
                if args.prof >= 0: torch.cuda.nvtx.range_push("forward")
                logits = model(input_in)
                output = logits
                if args.prof >= 0: torch.cuda.nvtx.range_pop()
                loss = F.cross_entropy(output, target)
            optimizer.zero_grad()

            if args.prof >= 0: torch.cuda.nvtx.range_push("backward")
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if args.prof >= 0: torch.cuda.nvtx.range_pop()

            if args.prof >= 0: torch.cuda.nvtx.range_push("optimizer.step()")
            optimizer.step()
            if args.prof >= 0: torch.cuda.nvtx.range_pop()

            if i % args.print_freq == 0:
                # Measure accuracy
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

                # Average loss and accuracy across processes for logging
                if args.distributed:
                    reduced_loss = reduce_tensor(loss.data)
                    prec1 = reduce_tensor(prec1)
                    prec5 = reduce_tensor(prec5)
                else:
                    reduced_loss = loss.data

                # to_python_float incurs a host<->device sync
                losses.update(to_python_float(reduced_loss), input_in.size(0))
                top1.update(to_python_float(prec1), input_in.size(0))
                top5.update(to_python_float(prec5), input_in.size(0))

                torch.cuda.synchronize()
                batch_time.update((time.time() - end) / args.print_freq)
                end = time.time()

                if args.local_rank == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Speed {3:.3f} ({4:.3f})\t'
                          'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, i, len(train_loader),
                        args.world_size * args.batch_size / batch_time.val,
                        args.world_size * args.batch_size / batch_time.avg,
                        batch_time=batch_time,
                        loss=losses, top1=top1, top5=top5))
            if args.prof >= 0: torch.cuda.nvtx.range_push("prefetcher.next()")
            input_in, target = prefetcher.next()
            input_ood, _ = prefetcher_ood.next()
            if args.prof >= 0: torch.cuda.nvtx.range_pop()

            # Pop range "Body of iteration {}".format(i)
            if args.prof >= 0: torch.cuda.nvtx.range_pop()

            if args.prof >= 0 and i == args.prof + 10:
                print("Profiling ended at iteration {}".format(i))
                torch.cuda.cudart().cudaProfilerStop()
                quit()

    else:  ## args.method == 'Baseline
        prefetcher = data_prefetcher(train_loader)
        input, target = prefetcher.next()
        i = 0
        while input is not None:
            i += 1
            if args.prof >= 0 and i == args.prof:
                print("Profiling begun at iteration {}".format(i))
                torch.cuda.cudart().cudaProfilerStart()
            if args.prof >= 0: torch.cuda.nvtx.range_push("Body of iteration {}".format(i))

            adjust_learning_rate(optimizer, epoch, i, len(train_loader))
            if args.prof >= 0: torch.cuda.nvtx.range_push("forward")
            output = model(input)
            if args.prof >= 0: torch.cuda.nvtx.range_pop()
            loss = criterion(output, target)
            optimizer.zero_grad()

            if args.prof >= 0: torch.cuda.nvtx.range_push("backward")
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if args.prof >= 0: torch.cuda.nvtx.range_pop()

            if args.prof >= 0: torch.cuda.nvtx.range_push("optimizer.step()")
            optimizer.step()
            if args.prof >= 0: torch.cuda.nvtx.range_pop()

            if i % args.print_freq == 0:
                # Measure accuracy
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

                # Average loss and accuracy across processes for logging
                if args.distributed:
                    reduced_loss = reduce_tensor(loss.data)
                    prec1 = reduce_tensor(prec1)
                    prec5 = reduce_tensor(prec5)
                else:
                    reduced_loss = loss.data

                # to_python_float incurs a host<->device sync
                losses.update(to_python_float(reduced_loss), input.size(0))
                top1.update(to_python_float(prec1), input.size(0))
                top5.update(to_python_float(prec5), input.size(0))

                torch.cuda.synchronize()
                batch_time.update((time.time() - end) / args.print_freq)
                end = time.time()

                if args.local_rank == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Speed {3:.3f} ({4:.3f})\t'
                          'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, i, len(train_loader),
                        args.world_size * args.batch_size / batch_time.val,
                        args.world_size * args.batch_size / batch_time.avg,
                        batch_time=batch_time,
                        loss=losses, top1=top1, top5=top5))
            if args.prof >= 0: torch.cuda.nvtx.range_push("prefetcher.next()")
            input, target = prefetcher.next()
            if args.prof >= 0: torch.cuda.nvtx.range_pop()

            # Pop range "Body of iteration {}".format(i)
            if args.prof >= 0: torch.cuda.nvtx.range_pop()

            if args.prof >= 0 and i == args.prof + 10:
                print("Profiling ended at iteration {}".format(i))
                torch.cuda.cudart().cudaProfilerStop()
                quit()


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    prefetcher = data_prefetcher(val_loader)
    input, target = prefetcher.next()
    i = 0
    while input is not None:
        i += 1

        # compute output
        with torch.no_grad():
            output = model(input)[:, :args.num_class]
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        if args.local_rank == 0 and i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader),
                args.world_size * args.batch_size / batch_time.val,
                args.world_size * args.batch_size / batch_time.avg,
                batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

        input, target = prefetcher.next()

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30
    if epoch >= 80:
        factor = factor + 1
    lr = args.lr * (0.1 ** factor)
    """Warmup"""
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt


if __name__ == '__main__':
    main()