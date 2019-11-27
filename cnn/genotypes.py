from collections import namedtuple

Genotype = namedtuple(
    'Genotype', 'normal normal_concat reduce reduce_concat')

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

# PRIMITIVES = [
#     'max_pool_3x3',
#     'none'
# ]

UNET_NAS = Genotype(
    normal=[
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 0),
        ('max_pool_3x3', 1), 
        ('dil_conv_3x3', 1),
        ('sep_conv_3x3', 3),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0)
    ], 
    normal_concat=range(2, 6),
    reduce=[
        ('skip_connect', 0),
        ('max_pool_3x3', 1),
        ('skip_connect', 1),
        ('max_pool_3x3', 0),
        ('dil_conv_5x5', 3),
        ('dil_conv_3x3', 0),
        ('skip_connect', 0),
        ('dil_conv_3x3', 2)
    ], 
    reduce_concat=range(2, 6)
)