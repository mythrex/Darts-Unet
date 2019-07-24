import os
import sys
import time
import numpy as np
import torch
import utils
import argparse
import tifffile as t

from model import Network as Network
from final_model.final_genotype import genotype as GENOTYPE

ROOT_PATH = os.getcwd()


def parse_args():
    parser = argparse.ArgumentParser("cifar")
                        help = 'location of the data corpus')
    parser.add_argument('--image', type = str,
                        default = './cnn/pred/input.tif', help = 'Location of input image')
    parser.add_argument('--weights', type = str,
                        default = './cnn/final_model/weights.pt', help = 'Path to weight file')
    parser.add_argument('--output_path', type = str,
                        default = './cnn/pred', help = 'Path to prediction folder')
    parser.add_argument('--gpu', type = int, default = 0,
                        help = 'gpu device id')
    parser.add_argument('--init_channels', type = int,
                        default = 3, help = 'num of init channels')
    parser.add_argument('--layers', type = int, default = 4,
                        help = 'total number of layers')
    parser.add_argument('--seed', type = int,
                        default = 2, help = 'random seed')
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
    utils.load(model, args.weights)


    criterion = nn.BCELoss()
    criterion = criterion.cuda()
    if(not os.path.exists('./cnn/pred')):
        os.mkdir('./cnn/pred')

    input = t.imread(args.image)
    input = torch.tensor(input)
    input = Variable(input).cuda().float()

    logits = model(input)
    mask = (logits > 0.5).int() * 255
    # convert to numpy
    mask = mask.numpy()
    t.imsave("{}/output.tif".format(args.output_path), mask.astype(np.uint8))

    print("Prediction saved")

if __name__ == "__main__":
    args = parse_args()
    predict(args, CLASSES, GENOTYPE)
