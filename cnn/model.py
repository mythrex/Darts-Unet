import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from operations import *
from genotypes import UNET_NAS as genotype
tf.enable_eager_execution()


class Cell(Model):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, upsample_prev):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.reduction_prev = reduction_prev

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, False)
        elif upsample_prev:
            self.preprocess0 = FactorizedUp(C_prev_prev, C, False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, False)

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

        self._ops = []
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, False)
            self._ops += [op]
        self._indices = indices

    def call(self, s0, s1):
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

        return tf.concat([states[i] for i in self._concat], axis=-1)


class UpsampleCell(Model):

    def __init__(self, genotype, C_prev_prev, C_prev, C):
        super(UpsampleCell, self).__init__()
        self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, unrolled=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, unrolled=False)
        concat = genotype.normal_concat
        self.multiplier = len(concat)
        self.UpConv = tf.keras.layers.Conv2DTranspose(C*self.multiplier,
                                                      kernel_size=3,
                                                      strides=2,
                                                      padding='same',
                                                      output_padding=1)
        self.reduction = False

    def call(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        s0 = self.UpConv(s0)
        s1 = self.UpConv(s1)

        return s0 + s1


class Network(Model):
    def __init__(self, C, num_classes, layers, genotype):
        super(Network, self).__init__()
        assert layers % 2 == 1
        stem_multiplier = 3
        C_curr = C * stem_multiplier
        self._layers_ = layers
        self.num_classes = num_classes
        self.stem_op = tf.keras.Sequential()
        self.stem_op.add(tf.keras.layers.Conv2D(
            C_curr, 3, use_bias=False, padding="same"))
        self.stem_op.add(tf.keras.layers.BatchNormalization())

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = []
        reduction_prev = False

        self.skip_ops = []
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

        self.softmaxConv = tf.keras.Sequential(name="softmaxConv")
        self.softmaxConv.add(tf.keras.layers.Conv2D(
            self.num_classes, kernel_size=1, strides=1, padding='same'))
        self.softmaxConv.add(tf.keras.layers.Softmax())

    def call(self, input):
        s0 = s1 = self.stem_op(input)
        pos = -1
        middle = self._layers_ - 1
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

        return self.softmaxConv(s1)
