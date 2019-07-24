import torch
import torch.nn as nn
from operations import *
from torch.autograd import Variable
from utils import drop_path


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction=False, reduction_prev=False, cell_type="cell1"):
        super(Cell, self).__init__()
        self.cell_type = cell_type
        self.reduction = False
        reduction_prev = False

        self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if self.cell_type == "cell1":
            op_names, indices = zip(*genotype.cell1)
            concat = genotype.cell1_concat
        elif self.cell_type == "cell2":
            op_names, indices = zip(*genotype.cell2)
            concat = genotype.cell2_concat
        elif self.cell_type == "cell3":
            op_names, indices = zip(*genotype.cell3)
            concat = genotype.cell3_concat
        elif self.cell_type == "cell4":
            op_names, indices = zip(*genotype.cell4)
            concat = genotype.cell4_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
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
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(Network, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        stem_multiplier = 3
        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            reduction = False
            if(i % 4 == 0):
                cell = Cell(genotype, C_prev_prev, C_prev,
                            C_curr, reduction, reduction_prev, cell_type="cell1")
            elif(i % 4 == 1):
                cell = Cell(genotype, C_prev_prev, C_prev,
                            C_curr, reduction, reduction_prev, cell_type="cell2")
            elif(i % 4 == 2):
                cell = Cell(genotype, C_prev_prev, C_prev,
                            C_curr, reduction, reduction_prev, cell_type="cell3")
            else:
                cell = Cell(genotype, C_prev_prev, C_prev,
                            C_curr, reduction, reduction_prev, cell_type="cell4")

            reduction_prev = reduction
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

        for _, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, drop_prob=0.)

        return self.sigmoidConv(s1)
