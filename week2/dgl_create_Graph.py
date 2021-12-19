import dgl
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
#创建图结构
g = dgl.graph(([0, 0, 0, 0, 0,1], [1, 2, 3, 4, 5,2]), num_nodes=6)
# Print the source and destination nodes of every edge.
# print(g.edges())


#将节点和边特征分配给图形
# Assign a 3-dimensional node feature vector for each node.
# g.ndata['x'] = torch.randn(6, 3)
# # Assign a 4-dimensional edge feature vector for each edge.
# g.edata['a'] = torch.randn(5, 4)
# # Assign a 5x4 node feature matrix for each node.  Node and edge features in DGL can be multi-dimensional.
# g.ndata['y'] = torch.randn(6, 5, 4)
# print(g.edata['a'])

#查询图结构
# print(g.num_nodes())
# print(g.num_edges())
# # Out degrees of the center node
# print(g.out_degrees(0))
# # In degrees of the center node - note that the graph is directed so the in degree should be 0.
# print(g.in_degrees(0))

#图转换
#如提取子图
# Induce a subgraph from node 0, node 1 and node 3 from the original graph.
sg1=g.subgraph([0,1,3])
# Induce a subgraph from edge 0, edge 1 and edge 3 from the original graph.
sg2 = g.edge_subgraph([0, 1, 3])

# # The original IDs of each node in sg1
# print(sg1.ndata[dgl.NID])
# # The original IDs of each edge in sg1
# print(sg1.edata[dgl.EID])
# # The original IDs of each node in sg2
# print(sg2.ndata[dgl.NID])
# # The original IDs of each edge in sg2
# print(sg2.edata[dgl.EID])

#为原始图中的每一条边添加一条反向边(无向图)
newg = dgl.add_reverse_edges(g)
print(newg.in_degrees())
# print(newg.edges())

#加载和保存图形
dgl.save_graphs('graph.dgl', g)
dgl.save_graphs('graphs.dgl', [g, sg1, sg2])

# Load graphs
(g,), _ = dgl.load_graphs('graph.dgl')

(g, sg1, sg2), _ = dgl.load_graphs('graphs.dgl')

# a=np.array(newg.in_degrees())
# weights = np.ones_like(a)/float(len(a))
# n, bins, patches = plt.hist(a,bins=30,weights=weights,facecolor='g', alpha=0.75)
# # plt.hist(a,50,density=1, facecolor="blue", edgecolor="black", alpha=0.75)
# # plt.axis([1,max(a)])
# plt.savefig("./1.jpg")

