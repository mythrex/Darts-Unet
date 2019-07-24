from collections import namedtuple

Genotype = namedtuple(
    'Genotype', 'cell1 cell1_concat cell2 cell2_concat cell3 cell3_concat cell4 cell4_concat')

genotype = Genotype(
    cell1=[
        ('sep_conv_5x5', 1),
        ('max_pool_3x3', 0),
        ('skip_connect', 2),
        ('dil_conv_3x3', 0),
        ('max_pool_3x3', 3),
        ('skip_connect', 2),
        ('dil_conv_3x3', 4),
        ('skip_connect', 3)
    ],
    cell1_concat=range(2, 6),
    cell2=[
        ('dil_conv_3x3', 0),
        ('skip_connect', 1),
        ('dil_conv_3x3', 2),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 3),
        ('avg_pool_3x3', 0),
        ('dil_conv_3x3', 1),
        ('max_pool_3x3', 2)
    ],
    cell2_concat=range(2, 6),
    cell3=[
        ('skip_connect', 1),
        ('dil_conv_5x5', 0),
        ('max_pool_3x3', 0),
        ('dil_conv_3x3', 2),
        ('dil_conv_5x5', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0),
        ('skip_connect', 4)
    ],
    cell3_concat=range(2, 6),
    cell4=[
        ('sep_conv_3x3', 1),
        ('max_pool_3x3', 0),
        ('sep_conv_5x5', 2),
        ('sep_conv_5x5', 1),
        ('sep_conv_5x5', 3),
        ('max_pool_3x3', 1),
        ('avg_pool_3x3', 4),
        ('skip_connect', 3)
    ], cell4_concat=range(2, 6))
