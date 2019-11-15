import tensorflow as tf
from tensorflow import layers

OPS = {
    'none': lambda C, stride: Zero(stride),
    'avg_pool_3x3': lambda C, stride: layers.AveragePooling2D(3, strides=stride, padding='same'),
    'max_pool_3x3': lambda C, stride: layers.MaxPooling2D(3, strides=stride, padding='same'),
    'skip_connect': lambda C, stride: Identity() if stride == 1 else FactorizedReduce(C, C),
    'sep_conv_3x3': lambda C, stride: SepConv(C, C, 3, stride, 'same'),
    'sep_conv_5x5': lambda C, stride: SepConv(C, C, 5, stride, 'same'),
    'sep_conv_7x7': lambda C, stride: SepConv(C, C, 7, stride, 'same'),
    'dil_conv_3x3': lambda C, stride: DilConv(C, C, 3, stride, 'same', 2),
    'dil_conv_5x5': lambda C, stride: DilConv(C, C, 5, stride, 'same', 2),
    'conv_7x1_1x7': lambda C, stride: Conv_7x1_1x7(C, stride)
}


class MaxPool3x3(tf.keras.layers.Layer):

    def __init__(self, C, stride):
        super(MaxPool3x3, self).__init__()
        self.pool = OPS['max_pool_3x3'](C, stride)
        self.bn = Identity()

    def call(self, x):
        x = self.pool(x)
        x = self.bn(x)
        return x


class AvgPool3x3(tf.keras.layers.Layer):

    def __init__(self, C, stride):
        super(AvgPool3x3, self).__init__()
        self.pool = OPS['avg_pool_3x3'](C, stride)
        self.bn = Identity()

    def call(self, x):
        x = self.pool(x)
        x = self.bn(x)
        return x


class ReLUConvBN(tf.keras.layers.Layer):
    """Applies ReLU, Conv and BatchNormalisation operation
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding='same'):
        """Initializes the operation

        Args:
            C_in (int): no of kernels in
            C_out (int): no of kernels out
            kernel_size (int): size of kernel
            stride (int): stride
            padding (int): padding
            affine (bool), optional): Defaults to True.
        """
        super(ReLUConvBN, self).__init__()
        self.relu = tf.nn.relu
        self.conv = layers.Conv2D(filters=C_out,
                                  kernel_size=kernel_size,
                                  strides=stride,
                                  padding='same',
                                  use_bias=False
                                  )
        self.bn = Identity()

    def call(self, x):
        """Applies the ReLU, Conv, BN to input

        Args:
            x (tensor): array or tensor (can be image)

        Returns:
            tensor: array or tensor with operations applied on it
        """
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class DilConv(tf.keras.layers.Layer):
    """Applies ReLU, Conv with dilation and BatchNormalisation operation
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation):
        super(DilConv, self).__init__()
        self.relu = tf.nn.relu
        # ! Since tensorflow does not allow stride > 1 with dilation > 1
        self.dil_conv = layers.Conv2D(filters=C_out,
                                      kernel_size=kernel_size,
                                      strides=1,
                                      padding=padding,
                                      dilation_rate=dilation,
                                      use_bias=False
                                      )
        self.conv = layers.Conv2D(filters=C_out,
                                  kernel_size=1,
                                  strides=stride,
                                  padding='same',
                                  use_bias=False
                                  )
        self.bn = Identity()

    def call(self, x):
        """Applies the ReLU, Conv, BN to input

        Args:
            x (tensor): array or tensor (can be image)

        Returns:
            tensor: array or tensor with operations applied on it
        """
        x = self.relu(x)
        x = self.dil_conv(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class SepConv(tf.keras.layers.Layer):
    """Applies ReLU, Sep Conv with dilation and BatchNormalisation operation
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(SepConv, self).__init__()
        self.relu = tf.nn.relu
        self.conv1 = layers.Conv2D(filters=C_in,
                                   kernel_size=kernel_size,
                                   strides=stride,
                                   padding='same',
                                   use_bias=False
                                   )
        self.conv2 = layers.Conv2D(filters=C_in,
                                   kernel_size=1,
                                   padding='same',
                                   use_bias=False
                                   )
        self.bn1 = Identity()
        self.conv3 = layers.Conv2D(filters=C_in,
                                   kernel_size=kernel_size,
                                   strides=1,
                                   padding='same',
                                   use_bias=False
                                   )
        self.conv4 = layers.Conv2D(filters=C_out,
                                   kernel_size=1,
                                   padding='same',
                                   use_bias=False
                                   )
        self.bn2 = Identity()

    def call(self, x):
        """Applies the ReLU, Conv, BN to input

        Args:
            x (tensor): array or tensor (can be image)

        Returns:
            tensor: array or tensor with operations applied on it
        """
        x = self.relu(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn2(x)
        return x


class Identity(tf.keras.layers.Layer):
    """Apply the identity operation
    """

    def __init__(self):
        super(Identity, self).__init__()

    def call(self, x):
        return x


class Zero(tf.keras.layers.Layer):
    """Makes array element zero with given stride
    """

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def call(self, x):
        if self.stride == 1:
            return tf.multiply(x, 0)
        return tf.multiply(x[:, ::self.stride, ::self.stride, :], 0)


class FactorizedUp(tf.keras.layers.Layer):

    def __init__(self, C_in, C_out):
        super(FactorizedUp, self).__init__()
        self.relu = tf.nn.relu
        self.trans_conv1 = layers.Conv2DTranspose(filters=C_out,
                                                  kernel_size=3,
                                                  strides=2,
                                                  padding='same',
                                                  )
        self.trans_conv2 = layers.Conv2DTranspose(filters=C_out,
                                                  kernel_size=3,
                                                  strides=2,
                                                  padding='same',
                                                  )

        self.bn = Identity()

    def call(self, x):
        x = self.relu(x)
        out = (self.trans_conv1(x) + self.trans_conv2(x)) * 0.5
        out = self.bn(out)
        return out


class Conv_7x1_1x7(tf.keras.layers.Layer):

    def __init__(self, C, stride):
        super(Conv_7x1_1x7, self).__init__()
        self.relu = tf.nn.relu
        self.conv1 = layers.Conv2D(filters=C,
                                   kernel_size=(1, 7),
                                   strides=(1, stride),
                                   padding='same',
                                   use_bias=False)
        self.conv2 = layers.Conv2D(filters=C,
                                   kernel_size=(7, 1),
                                   strides=(stride, 1),
                                   padding='same',
                                   use_bias=False)
        self.bn = Identity()

    def call(self, x):
        x = self.relu(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        return x


class FactorizedReduce(tf.keras.layers.Layer):
    """Applies ReLU, conv with stride=2 and c_out/2 
    """

    def __init__(self, C_in, C_out):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = tf.nn.relu
        self.conv_1 = layers.Conv2D(filters=C_out//2,
                                    kernel_size=1,
                                    strides=2,
                                    padding='same',
                                    use_bias=False)
        self.conv_2 = layers.Conv2D(filters=C_out//2,
                                    kernel_size=1,
                                    strides=2,
                                    padding='same',
                                    use_bias=False)
        self.bn = Identity()

    def call(self, x):
        """concats conv and Batch normalise them

        Args:
            x (tensor): array or tensor (can be image)

        Returns:
            tensor: tensor of operations on input
        """
        x = self.relu(x)
        out = tf.concat([self.conv_1(x), self.conv_2(x[:, 1:, 1:, :])], axis=3)
        out = self.bn(out)
        return out


class SkipConnection(tf.keras.layers.Layer):

    def __init__(self, C):
        super(SkipConnection, self).__init__()
        self.relu = tf.nn.relu
        self.conv = layers.Conv2D(filters=C,
                                  kernel_size=3,
                                  strides=1,
                                  padding='same',
                                  use_bias=False)

    def call(self, s0, s1):
        s0 = self.relu(s0)
        s1 = self.relu(s1)
        x = tf.concat([s1, s0], axis=3)
        out = self.conv(x)
        return out


class Softmax(tf.keras.layers.Layer):

    def __init__(self):
        super(Softmax, self).__init__()
        self.op = tf.nn.softmax

    def call(self, x):
        return self.op(x)
#         return x
