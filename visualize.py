import sys
from final_model.final_genotype_1000 import genotype
from graphviz import Digraph
import os
import shutil
import glob

ROOT_PATH = os.getcwd()


def plot(genotype, filename):
    g = Digraph(
        format='pdf',
        edge_attr=dict(fontsize='20', fontname="Calibri"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20',
                       height='0.5', width='0.5', penwidth='2', fontname="Calibri"),
        engine='dot')
    g.body.extend(['rankdir=LR'])

    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')
    assert len(genotype) % 2 == 0
    steps = len(genotype) // 2
    print(steps)

    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')

    for i in range(steps):
        for k in [2*i, 2*i + 1]:
            op, j = genotype[k]
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j-2)
            v = str(i)
            g.edge(u, v, label=op, fillcolor="gray")

    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(steps):
        g.edge(str(i), "c_{k}", fillcolor="gray")

    if(not os.path.exists(os.path.join(ROOT_PATH, "final_model", "cells"))):
        os.mkdir(os.path.join(ROOT_PATH, "final_model", "cells"))

    filename = os.path.join(ROOT_PATH, "final_model", "cells", filename)
    g.render(filename, view=False)


if __name__ == '__main__':
    files = glob.glob
    print("Normal Genotype")
    plot(genotype.normal, "normal")
    print("Reduce Genotype")
    plot(genotype.reduce, "reduce")
