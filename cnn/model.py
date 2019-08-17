import torch
import torch.nn as nn
from operations import *
from torch.autograd import Variable
from utils import drop_path


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, upsample_prev):
        super(Cell, self).__init__()
        self.reduction = reduction
        reduction_prev = reduction_prev

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        elif upsample_prev:
            self.preprocess0 = FactorizedUp(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            h1 = op1(h1)
            h2 = op2(h2)

            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class UpsampleCell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C):
        super(UpsampleCell, self).__init__()
        self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        concat = genotype.normal_concat
        self.multiplier = len(concat)
        self.UpConv = nn.ConvTranspose2d(C,
                                         C*self.multiplier,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         output_padding=1,
                                         dilation=1)
        self.reduction = False

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        s0 = self.UpConv(s0)
        s1 = self.UpConv(s1)

        return s0 + s1


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, genotype):
        super(Network, self).__init__()
        assert layers % 2 == 1

        self._layers = layers

        C_curr = 12
        self.stem = nn.Sequential(
            nn.Conv2d(C, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C_curr
        self.cells = nn.ModuleList()
        reduction_prev = False

        self.skip_ops = nn.ModuleList()
        # Downsample
        for i in range(layers):
            if i % 2 == 1:
                C_curr *= 2
                reduction = True

                self.skip_ops += [SkipConnection(C_prev)]
            else:
                reduction = False

            cell = Cell(genotype,
                        C_prev_prev,
                        C_prev,
                        C_curr,
                        reduction,
                        reduction_prev,
                        upsample_prev=False)

            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr

        for i in range(layers-1):
            # Skip operations
            if i % 2 == 0:
                C_curr = C_curr // 2
                cell = UpsampleCell(genotype, C_prev_prev, C_prev, C_curr)
            else:
                cell = Cell(genotype,
                            C_prev_prev,
                            C_prev,
                            C_curr,
                            reduction=False,
                            reduction_prev=False,
                            upsample_prev=True)

            self.cells += [cell]

            C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr

        self.sigmoidConv = nn.Sequential(
            nn.Conv2d(C_prev,
                      1,
                      kernel_size=1,
                      stride=1,
                      padding=0
                      ),
            nn.Sigmoid()
        )

    def forward(self, input):
        s0 = s1 = self.stem(input)
        pos = -1
        middle = self._layers - 1
        skip_cells = []
        skip_ops = list(self.skip_ops)

        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            # skip operation here
            if (i < middle and i % 2 == 0):
                skip_cells.append(s1)

            if (i > middle and i % 2 == 1):
                s1 = skip_ops[pos](skip_cells[pos], s1)
                pos -= 1

        return self.sigmoidConv(s1)
