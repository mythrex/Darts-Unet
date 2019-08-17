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

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, upsample_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    elif upsample_prev:
      self.preprocess0 = FactorizedUp(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
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
    
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)
    
    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)

class UpsampleCell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C):
        super(UpsampleCell, self).__init__()
        self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier
        self.UpConv = nn.ConvTranspose2d(C, 
                                        C*self._multiplier, 
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        output_padding=1,
                                        dilation=1)
        self.reduction = False

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        s0 = self.UpConv(s0)
        s1 = self.UpConv(s1)

        return s0 + s1

class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self._layers = layers
    
    C_curr = C*stem_multiplier
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
    
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i % 2 == 1:
        C_curr *= 2
        reduction = True
        
      else:
        reduction = False
        
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr,
                  reduction,
                  reduction_prev,
                  upsample_prev=False)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr
    
    for i in range(layers-1):
      if i % 2 == 0:
        C_curr = C_curr // 2
        
        cell = UpsampleCell(steps, multiplier, C_prev_prev, C_prev, C_curr)
      else:
        
        cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr,
                    reduction=False,
                    reduction_prev=False,
                    upsample_prev=True)
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr
    
    self.sigmoidConv = nn.Sequential(
            nn.Conv2d(C_prev,
                      1,
                      kernel_size=1,
                      stride=1,
                      padding=0
                      ),
            nn.Sigmoid()
    )
    
    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = s1 = self.stem(input)
    self.arr = []
    ids = []
    pos = -1
    
    middle = self._layers - 1
    
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      
      s0, s1 = s1, cell(s0, s1, weights)
    #   print("Cell no: {}, s1: {}".format(i, s1.shape))
      
      if (i < middle and i % 2 == 0):
        self.arr.append(s1)
        ids.append(i)
      
      if (i > middle and i % 2 == 1):
        # print("Skip Connection on Cell no {} and {}".format(ids[pos], i))
        C_curr = s1.shape[1]
        op = SkipConnection(C_curr).cuda()
        s1 = op(self.arr[pos], s1)
        pos -= 1

    return self.sigmoidConv(s1)

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
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

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype
