# This file is auto-generated. This file contains the model searched through train_search

from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

genotype=Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 4), ('dil_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 2), ('skip_connect', 4), ('dil_conv_3x3', 0)], reduce_concat=range(2, 6))