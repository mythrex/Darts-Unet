from collections import namedtuple

Genotype = namedtuple(
    'Genotype', 'cell1 cell1_concat cell2 cell2_concat cell3 cell3_concat cell4 cell4_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]
