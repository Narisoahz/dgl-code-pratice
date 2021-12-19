import dgl
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib

dataset = dgl.data.CoraGraphDataset()
g = dataset[0]

a=np.array(g.in_degrees())
weights = np.ones_like(a)/float(len(a))
n, bins, patches = plt.hist(a,bins=50,facecolor='g', alpha=0.75)
plt.xlabel("degree")
plt.ylabel("distribution")
plt.savefig("./2.jpg")