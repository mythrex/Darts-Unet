import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
from tqdm import tqdm

# import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from dataloader import DataLoader
import utils

from torch.autograd import Variable
from model_search import Network
from architect import Architect

CLASSES = 2

def parse_args():
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--data', type=str, default='./data',
                        help='location of the data corpus')
    parser.add_argument('--batch_size', type=int,
                        default=1, help='batch size')
    parser.add_argument('--learning_rate', type=float,
                        default=0.025, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float,
                        default=0.001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float,
                        default=3e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float,
                        default=10, help='report frequency')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--epochs', type=int, default=50,
                        help='num of training epochs')
    parser.add_argument('--init_channels', type=int,
                        default=3, help='num of init channels')
    parser.add_argument('--layers', type=int, default=5,
                        help='total number of layers')
    parser.add_argument('--model_path', type=str,
                        default='saved_models', help='path to save the model')
    parser.add_argument('--cutout', action='store_true',
                        default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int,
                        default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float,
                        default=0.3, help='drop path probability')
    parser.add_argument('--save', type=str, default='EXP',
                        help='experiment name')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--grad_clip', type=float,
                        default=5, help='gradient clipping')
    parser.add_argument('--train_portion', type=float,
                        default=1.0, help='portion of training data')
    parser.add_argument('--unrolled', action='store_true',
                        default=False, help='use one-step unrolled validation loss')
    parser.add_argument('--arch_learning_rate', type=float,
                        default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float,
                        default=1e-3, help='weight decay for arch encoding')
    args = parser.parse_args()

    return args


def main(args):
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.BCELoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CLASSES, args.layers, criterion)
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    # * Data handling here
    train_data = DataLoader(
        x_path="E:/URBAN_DATASET_BGH/train_x.npy",
        y_path="E:/URBAN_DATASET_BGH/train_y.npy",
        batch_size=args.batch_size,
        shuffle=True
    )
    train_queue = train_data.make_queue()
    train_queue = train_queue[:int(len(train_queue) * args.train_portion)]

    val_data = DataLoader(
        x_path="E:/URBAN_DATASET_BGH/val_x.npy",
        y_path="E:/URBAN_DATASET_BGH/val_y.npy",
        batch_size=args.batch_size,
        shuffle=True
    )
    valid_queue = val_data.make_queue()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    mIoUs = [0]
    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        # training
        train_acc, train_iou = train(
            train_queue, valid_queue, model, architect, criterion, optimizer, lr)
        
        # here should be
        logging.info('Final Train Acc %f', train_acc)
        logging.info('Final Train mIoU %f', train_iou)

        # validation
        valid_acc, valid_iou = infer(valid_queue, model, criterion)
        logging.info('Final Valid Acc %f', valid_acc)
        logging.info('Final Valid mIoU %f', valid_iou)

        # save the final genotype
        if(max(mIoUs) < valid_iou):
            print("Writing the computed genotype tp ./cnn/final_models/final_genotype.py")
            utils.write_genotype(genotype)
        
        mIoUs.append(valid_iou)
        np.save(os.path.join(args.save,"mIoUs.npy"), mIoUs)

def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
    """Trains the network. Gradient step is performed here

    Args:
        train_queue (array): Train queue
        valid_queue (array): Validation queue
        model (Network): Network
        architect (Architect): the architechture of network
        criterion (fn): Loss function
        optimizer (Optimiser): Adam / SGD
        lr (float): Learning Rate

    Returns:
        (float, float): returns top1 avg and overall avg
    """
    # objs = utils.AvgrageMeter()
    # top1 = utils.AvgrageMeter()
    # top5 = utils.AvgrageMeter()
    tq = tqdm(train_queue)
    step = 0
    intersections = []
    unions = []
    model.train()
    for (input, target) in tq:
        input = torch.tensor(input).float()
        target = torch.tensor(target)
        n = input.size(0)
        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda().float()

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = torch.tensor(input_search).float()
        target_search = torch.tensor(target_search).long()
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(
            target_search, requires_grad=False).cuda().float()
    
        architect.step(input, target, input_search, target_search,
                       lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        acc = utils.accuracy(logits, target)
        iou, intersection, union = utils.iou(logits,target)

        intersections.append(intersection.item())
        unions.append(union.item())

        if step % args.report_freq == 0:
            tq.set_postfix({
                "Acc": acc.item(),
                "IoU": iou.item(),
                "Loss": loss.item()
            })
        step += 1

    # for removing all unions where union = 0
    unions = np.array(unions)
    intersections = np.array(intersections)
    non_zero_mask = unions != 0
    mIoU = np.mean(intersections[non_zero_mask])/(np.mean(unions[non_zero_mask]) + 1e-6)
    # return here mean iou
    return acc, mIoU


def infer(valid_queue, model, criterion):
    # objs = utils.AvgrageMeter()
    # top1 = utils.AvgrageMeter()
    # top5 = utils.AvgrageMeter()
    model.eval()
    tq = tqdm(valid_queue)
    step = 0
    intersections = []
    unions = []
    for (input, target) in tq:
        input = torch.tensor(input).float()
        target = torch.tensor(target)
        input = Variable(input).cuda()
        target = Variable(target).cuda().float()

        logits = model(input)
        loss = criterion(logits, target)

        acc = utils.accuracy(logits, target)
        iou, intersection, union = utils.iou(logits,target)

        intersections.append(intersection.item())
        unions.append(union.item())

        if step % args.report_freq == 0:
            tq.set_postfix({
                "Acc": acc.item(),
                "IoU": iou.item(),
                "Loss": loss.item()
            })
        step += 1

    # for removing all unions where union = 0
    unions = np.array(unions)
    intersections = np.array(intersections)
    non_zero_mask = unions != 0
    mIoU = np.mean(intersections[non_zero_mask]) / (np.mean(unions[non_zero_mask]) + 1e-6)
    return acc, mIoU


if __name__ == '__main__':
    args = parse_args()

    args.save = 'search-{}-{}'.format(args.save,
                                      time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    main(args)
