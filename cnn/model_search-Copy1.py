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
        s = tf.zeros_like(op_on_x)
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
            s = sum(self._ops[offset+j](h, weights[offset+j])
                    for j, h in enumerate(states))
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

    def __init__(self, C, net_layers, criterion, steps=4, multiplier=4, stem_multiplier=3, num_classes=1):
        super(Network, self).__init__()
        self._C = C
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.net_layers = net_layers
        self.num_classes = num_classes
        
        C_curr = C*stem_multiplier

        # stem operation
        self.stem_op = tf.keras.Sequential()
        self.stem_op.add(tf.keras.layers.Conv2D(
            C_curr, kernel_size=3, padding='same', use_bias=False))
        self.stem_op.add(tf.keras.layers.BatchNormalization())

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = []
        self.skip_ops = []

        reduction_prev = False

        # For reduction
        for i in range(self.net_layers):
            if i % 2 == 1:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
                self.skip_ops += [SkipConnection(C_curr)]
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

                cell = UpsampleCell(steps, multiplier,
                                    C_prev_prev, C_prev, C_curr)
            else:
                cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr,
                            reduction=False,
                            reduction_prev=False,
                            upsample_prev=True)
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr

        self.softmaxConv = tf.keras.Sequential(name="softmaxConv")
        self.softmaxConv.add(tf.keras.layers.Conv2D(
            self.num_classes, kernel_size=1, strides=1, padding='same'))
        self.softmaxConv.add(Softmax())

        self._initialize_alphas()

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2+i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = tf.Variable(
            1e-3*tf.random.uniform([k, num_ops]), name='alphas_normal')
        self.alphas_reduce = tf.Variable(
            1e-3*tf.random.uniform([k, num_ops]), name='alphas_reduce')
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
        return self._criterion(logits, tf.to_float(target))

    def call(self, inp):
        s0 = s1 = self.stem_op(inp)
        self.arr = []
        ids = []
        pos = -1

        middle = self.net_layers - 1
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = tf.nn.softmax(self.alphas_reduce, axis=0)
            else:
                weights = tf.nn.softmax(self.alphas_normal, axis=0)

            s0, s1 = s1, cell(s0, s1, weights)

            if (i < middle and i % 2 == 0):
                self.arr.append(s1)
                ids.append(i)

            if (i > middle and i % 2 == 1):
                C_curr = s1.shape[1]
                s1 = self.skip_ops[-pos-1](self.arr[pos], s1)
                pos -= 1
        logits = tf.argmax(self.softmaxConv(s1), axis=-1, name="output")
        logits = tf.cast(logits, inp.dtype)
        return tf.reshape(logits, (logits.shape[0], logits.shape[1], logits.shape[2], 1), name="output_reshaped")

    def genotype(self):
        def _parse(weights):
            primitives = tf.convert_to_tensor(PRIMITIVES)
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = tf.identity(weights[start:end])
                none_idx = PRIMITIVES.index('none')
                # make none_weight to be -inf
                mask = np.ones(W.shape)
                mask[:, none_idx] = -1
                W = W * tf.convert_to_tensor(mask, dtype=W.dtype)        
                # calc of edges
                W_sorted = tf.sort(W, axis=-1, direction='DESCENDING', name='sorted_weights')
                edges = tf.argsort(W_sorted[:,0], axis=-1, direction='DESCENDING', name='edges')[:2]

                for idx in range(edges.shape[0]):
                    j = edges[idx] 
                    k_best = tf.argsort(W, axis=-1, direction='DESCENDING', name='k_best')[j][0]

                    gene.append((primitives[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(tf.nn.softmax(
            self.alphas_normal, axis=-1))
        gene_reduce = _parse(tf.nn.softmax(
            self.alphas_reduce, axis=-1))

        concat = range(2+self._steps-self._multiplier, self._steps+2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype

    def get_thetas(self):
        specific_tensor = []
        specific_tensor_name = []
        for var in self.trainable_weights:
            if not 'alphas' in var.name:
                specific_tensor.append(var)
                specific_tensor_name.append(var.name)
        return specific_tensor
