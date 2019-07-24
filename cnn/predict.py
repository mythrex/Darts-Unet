import os
import sys
import time
import numpy as np
import torch
import utils
import argparse
import tifffile as t
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable

from model import Network as Network
from final_model.final_genotype import genotype as GENOTYPE

ROOT_PATH = os.getcwd()


def parse_args():
    parser = argparse.ArgumentParser("segmentation")
    parser.add_argument('--image', type = str,
                        default = './pred/input.tif', help = 'Location of input image')
    parser.add_argument('--weights', type = str,
                        default = './cnn/final_model/weights.pt', help = 'Path to weight file')
    parser.add_argument('--output_path', type = str,
                        default = './pred', help = 'Path to prediction folder')
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
    parser.add_argument('--layers', type=int, default=4,
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
                        default=0.7, help='portion of training data')
    parser.add_argument('--unrolled', action='store_true',
                        default=False, help='use one-step unrolled validation loss')
    parser.add_argument('--arch_learning_rate', type=float,
                        default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float,
                        default=1e-3, help='weight decay for arch encoding')
    args=parser.parse_args()

    return args

CLASSES=2

def predict(args, num_classes, genotype):
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark=True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)


    model=Network(args.init_channels, num_classes,
                    args.layers, False, genotype)
    model = model.cuda()
    model.load_state_dict(torch.load(args.weights))


    criterion = nn.BCELoss()
    criterion = criterion.cuda()
    if(not os.path.exists('./pred')):
        os.mkdir('./pred')

    input = t.imread(args.image).astype("float32")

    input = input.reshape((1, input.shape[0], input.shape[1], input.shape[2]))
    input = input.swapaxes(2, 3).swapaxes(1, 2)
    input = torch.tensor(input)
    input = Variable(input).cuda().float()

    logits = model(input)
    mask = (logits > 0.5).int() * 255

    # convert to numpy
    mask = mask.cpu().numpy()
    mask = mask.swapaxes(1, 2).swapaxes(2, 3)
    mask = mask.reshape((mask.shape[1], mask.shape[2], mask.shape[3]))
    
    t.imsave("{}/output.tif".format(args.output_path), mask.astype(np.uint8))

    print("Prediction saved")

if __name__ == "__main__":
    args = parse_args()
    predict(args, CLASSES, GENOTYPE)
