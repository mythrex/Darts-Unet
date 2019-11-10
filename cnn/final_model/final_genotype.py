# This file is auto-generated. This file contains the model searched through train_search

from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

genotype=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3), ('sep_conv_5x5', 3), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 4), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])