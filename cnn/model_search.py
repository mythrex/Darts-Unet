import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


class MixedOp(nn.Module):
    """Makes mixed operations object
    """

    def __init__(self, C, stride):
        """Makes ops array of all the arrays

        Args:
            C (int): the current state
        """
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        """Converts the discrete set of operation into conitnuous mixed operation

        Args:
            x (tensor): can be tensor or array, (e.g. image-array)
            weights (tensor): tensor or array of Softmax probability of alphas

        Returns:
            tensor: sum of product(weights, operation(x))
        """
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):
    """Makes a cell unit

    Parameters:
        reduction (boolean): if this cell is reduction cell
        preprocess0 (function): preprocessing funtion for prev to prev layer
        preprocess1 (function): preprocessing funtion for prev cell
    """

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        """Makes an array of operations for the blocks

            Args:
                    steps (int): no of blocks in cell
                    multiplier (int): multiplier
                    C_prev_prev (int): hidden state from prev prev layer
                    C_prev (int): hidden state from prev layer
                    C (int): current state
                    reduction (bool): if this is reduction cell or not
                    reduction_prev (bool): if prev layer was reduction
            """

        super(Cell, self).__init__()
        self.reduction = False
        reduction_prev = False

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(
                C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        """Returns the cell output

        Args:
            s0 (tensor): the state from prev to prev cell
            s1 (tensor): the state from prev cell
            weights (tensor): tensor or array of Softmax probability of alphas

        Returns:
            tensor: concaternation of all the blocks in a cell
        """
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset+j](h, weights[offset+j])
                    for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):
    """Constructs the network

    Parameters:
        stem (Sequential): Sequential layer, conv and BatchNorm
        cells(ModuleList): contains all the cell of a network
        global_pooling(nn.AdaptiveAvgPool2d): applies a 2D adaptive average pooling to output 1
        classifier(nn.Linear): classifies into classes
    """

    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        """Intialise the cell list, and functions
        The array constructed is [N(16), N(16), R(32), N(32), N(32), R(64), N(64), N(64)]
        Args:
            C (int): current state(initial no of channels)
            num_classes (int): no of classes in dataset
            layers (int): total cell layers in network
            criterion (loss_fn): the criteria for update of weights
            steps (int, optional): no of blocks. Defaults to 4.
            multiplier (int, optional): factor by which . Defaults to 4.
            stem_multiplier (int, optional): factor for stem cell output. Defaults to 3.
        """
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            # if i in [layers//3, 2*layers//3]:
            #     C_curr *= 2
            #     reduction = True
            # else:
            reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev,
                        C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def new(self):

        model_new = Network(self._C, self._num_classes,
                            self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        """Passes the input through stem, network cell
        input -> S0 -> S1 -> N -> N -> R -> N -> N -> R -> N -> N -> pooling -> logits 

        Args:
            input (tensor): array or tensor (can be image)

        Returns:
            tensor: logits computed by passing it through network 
        """
        s0 = s1 = self.stem(input)
        for _, cell in enumerate(self.cells):
            # if cell.reduction:
            #     weights = F.softmax(self.alphas_reduce, dim=-1)
            # else:
            weights = F.softmax(self.alphas_normal, dim=-1)
            # * forward function for Cell
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))

        return logits

    def _loss(self, input, target):
        """[]

        Args:
            input (tensor): logits, array or tensor (can be image)
            target (tensor): labels, array or tensor (can be image)

        Returns:
            tensor: loss, or creterion fn that is passed
        """
        logits = self(input)
        return self._criterion(logits, target.long())

    def _initialize_alphas(self):
        """initialize the alphas for network
        """
        k = sum(1 for i in range(self._steps) for n in range(2+i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(
            1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        # self.alphas_reduce = Variable(
        #     1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            # self.alphas_reduce
        ]

    def arch_parameters(self):
        """returns the arch params

        Returns:
            array: array containing alphas for normal and reduce cell
        """
        return self._arch_parameters

    def genotype(self):
        """Finds the genotype of normal and reduction cell

        Returns:
            dict: a dict containing normal, reduce genotype
        """
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(
                    W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(
            F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        # gene_reduce = _parse(
        #     F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2+self._steps-self._multiplier, self._steps+2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat
            # reduce=gene_reduce, reduce_concat=concat
        )
        return genotype
