import networkx as nx
import igraph as ig
import numpy as np
from graph_tools.gwrapper import GraphWrapper
from graph_tools.gwrapper import create_ig_graph, create_nx_graph
import pandas as pd

edges = [('x0', 'x1'), ('x1', 'x2'), ('x2', 'x3'), ('x3', 'x0'), ('x1', 'y'), ('x2', 'y'), ('x3', 'y')]
edge_weights = np.arange(1, len(edges)+1)[::-1]

df = pd.DataFrame(edges, columns=['va', 'vb'])

g_ig = create_ig_graph(edges, edge_weights)

gw_ig_dir = GraphWrapper(g_ig, mode='ig', directed=True, edge_attr_count='weight', edge_attr_dist='dist')
gw_ig_undir = GraphWrapper(g_ig, mode='ig', directed=False, edge_attr_count='weight', edge_attr_dist='dist')
ig_weights = [((e.source, e.target), e['weight']) for e in gw_ig_dir.g.es()]

gw_ig_dir.components('STRONG')
agg_dir = gw_ig_dir.compute_curv_components()
df_dir = pd.DataFrame(agg_dir, columns=['va', 'vb', 'curv_dir'])

gw_ig_undir.components('WEAK')
agg_undir = gw_ig_undir.compute_curv_components()
df_undir = pd.DataFrame(agg_undir, columns=['va', 'vb', 'curv_undir'])

agg_vertices = []
weight_cols = ['vweight', 'vweight_IN', 'vweight_OUT']
for v in gw_ig_dir.g.vs():
    agg_vertices += [[v['name'], *[v[c] for c in weight_cols]]]
df_vert = pd.DataFrame(agg_vertices, columns=['v', *weight_cols])

dft = pd.merge(df, df_undir, on=['va', 'vb'], how='left')
dft = pd.merge(dft, df_dir, on=['va', 'vb'], how='left')
dft = pd.merge(dft, df_vert, left_on=['va'], right_on=['v'], how='left')
dft = pd.merge(dft, df_vert, left_on=['vb'], right_on=['v'], how='left', suffixes=('_a', '_b'))
dft.drop(['v_a', 'v_b'], axis=1, inplace=True)

