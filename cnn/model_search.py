import tensorflow as tf
from tensorflow.keras import Model
from genotypes import PRIMITIVES
from genotypes import Genotype
from operations import *
import numpy as np
from utils import get_tensor_at


class MixedOp(Model):
    """Makes mixed operations object
    """

    def __init__(self, C, stride):
        """Makes ops array of all the arrays

        Args:
            C (int): the current state
        """
        super(MixedOp, self).__init__()
        self._ops = []
        for primitive in PRIMITIVES:
                op = OPS[primitive](C, stride)
                if 'pool' in primitive:
                    if('avg' in primitive):
                        op = AvgPool3x3(C, stride)
                    elif('max' in primitive):
                        op = MaxPool3x3(C, stride)
                self._ops.append(op)
        
    def call(self, x, weights):
        """Converts the discrete set of operation into conitnuous mixed operation

        Args:
            x (tensor): can be tensor or array, (e.g. image-array)
            weights (tensor): tensor or array of Softmax probability of alphas

        Returns:
            tensor: sum of product(weights, operation(x))
        """
        op_on_x = self._ops[0](x)
        mask = tf.eye(int(weights.shape[0]), dtype=tf.bool)
        s = tf.zeros(op_on_x.shape, dtype=op_on_x.dtype)
        for i in range(len(self._ops)):
            s += get_tensor_at(weights, mask, i) * self._ops[i](x)
        return s
    
class Cell(Model):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, upsample_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        elif upsample_prev:
            self.preprocess0 = FactorizedUp(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 'same')
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 'same')
        self._steps = steps
        self._multiplier = multiplier

        self._ops = []
        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def call(self, s0, s1, weights):
    
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
    
        for i in range(self._steps):
            s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return tf.concat(states[-self._multiplier:], axis=-1)
    
class UpsampleCell(Model):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C):
        super(UpsampleCell, self).__init__()
        self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 'same')
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 'same')
        self._steps = steps
        self._multiplier = multiplier
        self.UpConv = layers.Conv2DTranspose(C*self._multiplier, 
                                        kernel_size=3,
                                        strides=2,
                                        padding='same')
        self.reduction = False

    def call(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        s0 = self.UpConv(s0)
        s1 = self.UpConv(s1)

        return s0 + s1

class Network(Model):
    
    def __init__(self, C, net_layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.net_layers = net_layers

        C_curr = C*stem_multiplier

        # stem operation
        self.stem_op = tf.keras.Sequential()
        self.stem_op.add(tf.keras.layers.Conv2D(C_curr, kernel_size=3, padding='same', use_bias=False))
        self.stem_op.add(tf.keras.layers.BatchNormalization())

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = []

        reduction_prev = False
        # For reduction
        for i in range(self.net_layers):
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

        for i in range(self.net_layers-1):
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

        self.softmaxConv = tf.keras.Sequential()
        self.softmaxConv.add(tf.keras.layers.Conv2D(1, kernel_size=1, strides=1, padding='same'))
        self.softmaxConv.add(Softmax())

        self._initialize_alphas()
    
    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2+i))
        num_ops = len(PRIMITIVES)
        
        self.alphas_normal = tf.Variable(1e-3*tf.random.uniform([k, num_ops]), name='alphas_normal')
        self.alphas_reduce = tf.Variable(1e-3*tf.random.uniform([k, num_ops]), name='alphas_reduce')
        self._arch_parameters = [
          self.alphas_normal,
          self.alphas_reduce,
        ]
    
    def arch_parameters(self):
        return self._arch_parameters
    
    def new(self):
        model_new = Network(self._C, self.net_layers, self._criterion)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.assign(y)
        return model_new
    
    def _loss(self, logits, target):
        return self._criterion(logits, tf.dtypes.cast(target, tf.float32))
    
    def call(self, inp):
        s0 = s1 = self.stem_op(inp)
        self.arr = []
        ids = []
        pos = -1

        middle = self.net_layers - 1
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = tf.nn.softmax(self.alphas_reduce, axis=-1)
            else:
                weights = tf.nn.softmax(self.alphas_normal, axis=-1)

            s0, s1 = s1, cell(s0, s1, weights)

            if (i < middle and i % 2 == 0):
                self.arr.append(s1)
                ids.append(i)

            if (i > middle and i % 2 == 1):
                C_curr = s1.shape[1]
                op = SkipConnection(C_curr)
                s1 = op(self.arr[pos], s1)
                pos -= 1

        return self.softmaxConv(s1)

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

            gene_normal = _parse(tf.nn.softmax(self.alphas_normal, axis=-1).numpy())
            gene_reduce = _parse(tf.nn.softmax(self.alphas_reduce, dim=-1).numpy())

            concat = range(2+self._steps-self._multiplier, self._steps+2)
            genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
            )
            return genotype