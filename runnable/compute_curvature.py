import numpy as np
from graph_tools.gwrapper import GraphWrapper
from graph_tools.gwrapper import create_ig_graph, create_nx_graph
import pandas as pd
from os.path import expanduser

va = 'up'
vb = 'dn'

test = True
test = False
n_tasks = 24
# n_tasks = None

if test:
    edges = [('x0', 'x1'), ('x1', 'x2'), ('x2', 'x3'), ('x3', 'x0'), ('x1', 'y'), ('x2', 'y'), ('x3', 'y')]
    edge_weights = np.arange(1, len(edges)+1)[::-1]
else:
    df0 = pd.read_csv('~/data/kl/final/literature_edges.csv.gz', index_col=0)
    mask = (df0[va] == df0[vb])
    print(sum(mask))
    df0 = df0.loc[~mask].copy()
    print(df0.head())

    # df0 = df0.head(4000)
    edges = df0[[va, vb]].values
    edge_weights = df0['count'].values

df = pd.DataFrame(edges, columns=[va, vb])

g_ig_wei = create_ig_graph(edges, edge_weights)

gw_ig_dir_wei = GraphWrapper(g_ig_wei, mode='ig', directed=True, edge_attr_count='weight', edge_attr_dist='dist')
gw_ig_undir_wei = GraphWrapper(g_ig_wei, mode='ig', directed=False, edge_attr_count='weight', edge_attr_dist='dist')

gw_ig_dir_wei.components('STRONG', verbose=True)
# agg_dir = gw_ig_dir.compute_curv_components(verbose=True)
agg_dir_wei = gw_ig_dir_wei.compute_curv_components(direction='OUT', n_tasks=n_tasks,
                                                    dist_dir='OUT',
                                                    weighted_distance=True,
                                                    verbose=False)
df_dir_wei = pd.DataFrame(agg_dir_wei, columns=[va, vb, 'curv_dir_wei'])

gw_ig_undir_wei.components('WEAK', verbose=True)
agg_undir_wei = gw_ig_undir_wei.compute_curv_components(n_tasks=n_tasks,
                                                        dist_dir=None,
                                                        weighted_distance=True,
                                                        verbose=False)
agg_undir_wei = [(va_, vb_, value) if (va_, vb_) in edges else (vb_, va_, value)
             for va_, vb_, value in agg_undir_wei]
df_undir_wei = pd.DataFrame(agg_undir_wei, columns=[va, vb, 'curv_undir_wei'])

# weightless
g_ig_unwei = create_ig_graph(edges, [1]*len(edges))

gw_ig_dir_unwei = GraphWrapper(g_ig_unwei, mode='ig', directed=True, edge_attr_count=None, edge_attr_dist=None)
gw_ig_undir_unwei = GraphWrapper(g_ig_unwei, mode='ig', directed=False, edge_attr_count=None, edge_attr_dist=None)

gw_ig_dir_unwei.components('STRONG', verbose=True)
agg_dir_unwei = gw_ig_dir_unwei.compute_curv_components(direction='OUT', n_tasks=n_tasks,
                                                        dist_dir='OUT',
                                                        weighted_distance=False,
                                                        verbose=False)
df_dir_unwei = pd.DataFrame(agg_dir_unwei, columns=[va, vb, 'curv_dir_unwei'])

gw_ig_undir_unwei.components('WEAK', verbose=True)
agg_undir_unwei = gw_ig_undir_unwei.compute_curv_components(n_tasks=n_tasks,
                                                            dist_dir=None,
                                                            weighted_distance=False,
                                                            verbose=False)
agg_undir_unwei = [(va_, vb_, value) if (va_, vb_) in edges else (vb_, va_, value)
             for va_, vb_, value in agg_undir_unwei]
df_undir_unwei = pd.DataFrame(agg_undir_unwei, columns=[va, vb, 'curv_undir_unwei'])
print(df_undir_unwei.head())
print(df_undir_wei.head())

agg_vertices = []
weight_cols = ['vweight', 'vweight_IN', 'vweight_OUT']
for v in gw_ig_dir_wei.g.vs():
    agg_vertices += [[v['name'], *[v[c] for c in weight_cols]]]
df_vert = pd.DataFrame(agg_vertices, columns=['v', *weight_cols])

dft = pd.merge(df, df_undir_wei, on=[va, vb], how='left')
dft = pd.merge(dft, df_dir_wei, on=[va, vb], how='left')
dft = pd.merge(dft, df_undir_unwei, on=[va, vb], how='left')
dft = pd.merge(dft, df_dir_unwei, on=[va, vb], how='left')

dft = pd.merge(dft, df_vert, left_on=[va], right_on=['v'], how='left')
dft = pd.merge(dft, df_vert, left_on=[vb], right_on=['v'], how='left', suffixes=('_a', '_b'))
dft.drop(['v_a', 'v_b'], axis=1, inplace=True)

dft.to_csv(expanduser('~/data/gene_curv_feats_all.csv.gz'))
