from numpy import random as rn
import numpy as np
import pandas as pd
import networkx as nx
from numpy.random import RandomState


def transform_cite_dict(cite_dict):
    """

    :param cite_dict: dict {wA : [wBs]}
    :return: 2xN array, a[0] contains wA, a[1] contains wBs
    """

    lens = list(map(lambda z: len(z), cite_dict.values()))
    d1 = sum(lens)
    arr = np.ndarray((2, d1), dtype=int)
    cs1 = np.cumsum(lens)
    cs0 = [0] + list(cs1[:-1])
    for k, val_list, x, y in zip(cite_dict.keys(), cite_dict.values(), cs0, cs1):
        arr[0, x:y] = [k]*(y-x)
        arr[1, x:y] = val_list
    return arr


def create_dummy_journals(df_cite, df_wj, seed=13):
    """
    test only
    df_cite: df with 'wA' and 'wB' columns
    df_wj: df with 'w' and 'j' columns
    """

    rns = rn.RandomState(seed)

    wBs = df_cite['wB'].unique()
    ws = df_wj['w'].unique()
    outstanding_wBs = np.array(list(set(wBs) - set(ws)))
    print('unique journals:', len(df_wj['j'].unique()))
    js = df_wj['j'].unique()

    n = 8
    j_inds = [rns.randint(n) for i in range(len(outstanding_wBs))]

    extra_js = js[j_inds]
    print(len(extra_js), len(outstanding_wBs))
    extra_df2 = pd.DataFrame({'w': outstanding_wBs, 'j': extra_js})
    df_wj_full = pd.concat([df_wj, extra_df2])
    print('unique journals at the end:', len(df_wj_full['j'].unique()))
    return df_wj_full


def create_dummy_cdatas(cdata, k=5, jm=8, seed=13):
    """
    test only

    :param k: number of batches
    :param cdata:  [(w, j, w_refs_list)]
    :return: list of cdatas

    from cdata list : [(w, j, w_refs_list)]
    creates a list of k cdata-format dummy lists [[(w_, j_, w_refs_list_)]]
    where w_ are taken from w_refs_list and j_ are taken from j at random
    """

    from numpy.random import RandomState
    wids_lists = list(map(lambda x: x[2], cdata))
    js_list = list(set(map(lambda x: x[1], cdata)))[:jm]
    wids_set = set([x for sublist in wids_lists for x in sublist])
    wids_list = list(wids_set)
    n = len(wids_set)
    delta = int(n / k)
    inds = [(i * delta, (i + 1) * delta) for i in range(k)]

    dummy_wids_lists = [wids_list[ind[0]:ind[1]] for ind in inds]
    print(list(map(lambda x: len(x), dummy_wids_lists)))

    rns = RandomState(seed)
    rns.randint(len(js_list))
    dummy_js_lists = [[js_list[rns.randint(len(js_list))] for i in range(ind[1] - ind[0])] for ind in inds]
    emptys = [[[] for i in range(ind[1] - ind[0])] for ind in inds]

    print(list(map(lambda x: len(x), dummy_js_lists)))

    cdata_list = [list(zip(x[0], x[1], x[2])) for x in zip(dummy_wids_lists, dummy_js_lists, emptys)]
    return cdata_list


def create_dummy_ajs(cdata, aj_data, k=5, jm=None, seed=13):
    """
    test only

    :param k: number of batches
    :param cdata:  [(w, j, w_refs_list)]
    :return: list of cdatas

    from cdata list : [(w, j, w_refs_list)]
    creates a list of k cdata-format dummy lists [[(w_, j_, w_refs_list_)]]
    where w_ are taken from w_refs_list and j_ are taken from j at random
    """

    from numpy.random import RandomState
    wids_lists = list(map(lambda x: x[1], cdata))
    js_list = list(set(map(lambda x: x[1], aj_data)))[:jm]
    wids_set = set([x for sublist in wids_lists for x in sublist])
    wids_list = list(wids_set)
    n = len(wids_set)
    delta = int(n / k)
    inds = [(i * delta, (i + 1) * delta) for i in range(k)]

    dummy_wids_lists = [wids_list[ind[0]:ind[1]] for ind in inds]
    print(list(map(lambda x: len(x), dummy_wids_lists)))

    rns = RandomState(seed)
    rns.randint(len(js_list))
    dummy_js_lists = [[js_list[rns.randint(len(js_list))] for i in range(ind[1] - ind[0])] for ind in inds]

    print(list(map(lambda x: len(x), dummy_js_lists)))

    aj_list = [list(zip(x[0], x[1])) for x in zip(dummy_wids_lists, dummy_js_lists)]
    return aj_list


def create_adj_df(df_cite, df_wj):
    """
    df_cite: df with 'wA' and 'wB' columns
    df_wj: df with 'w' and 'j' columns
    """

    print('unique journals in adj:', len(df_wj['j'].unique()))
    ws_unique = df_wj['w'].unique()
    df_cite_cut = df_cite.loc[df_cite['wB'].isin(ws_unique)]
    # log if the shapes of df_cite and df_cite_cut are different
    print(df_cite.shape[0], df_cite_cut.shape[0])
    df_agg = pd.merge(df_wj.rename(columns={'w': 'wA'}), df_cite_cut, how='right', on='wA')
    df_agg.rename(columns={'j': 'jA'}, inplace=True)
    print(df_agg.shape)
    df_agg = pd.merge(df_agg, df_wj.rename(columns={'w': 'wB'}), how='left', on='wB')
    df_agg.rename(columns={'j': 'jB'}, inplace=True)
    print(df_agg.shape)
    return df_agg


def extend_jj_df(sorted_js, df_agg):
    """
    extend jA to jB matrix
    (some journals may be not cited
    or cited and haven't published)
    the latter should not be possible

    :param sorted_js:
    :param df_agg:
    :return:
    """

    df_tot = pd.DataFrame(np.nan, columns=sorted_js, index=sorted_js)
    df_agg2 = df_agg[['jA', 'jB']].groupby(['jA', 'jB']).apply(lambda x: x.shape[0])
    df_adj = df_agg2.reset_index().pivot('jA', 'jB', 0)
    # no need to reindex df_adj, update() takes care of that
    df_tot.update(df_adj)
    df_tot = df_tot.fillna(0)
    return df_adj, df_tot


def generate_bigraph_inv_foo_injection(types_pair, nodes_pair, seed=13):
    rns = RandomState(seed)

    sizes = list(map(lambda x: len(x), nodes_pair))
    g = nx.Graph()
    for t, nodes in zip(types_pair, nodes_pair):
        g.add_nodes_from(map(lambda x: (t, x), nodes))

    indices = rns.choice(sizes[0], sizes[1])
    for k in range(sizes[1]):
        g.add_edge((types_pair[0], nodes_pair[0][indices[k]]),
                   (types_pair[1], nodes_pair[1][k]), {'weight': 1.0})
    return g


def create_bigraph(properties_a, properties_b, l=None, seed=13):
    from numpy.random import RandomState
    rns = RandomState(seed)
    name_a, n = properties_a
    name_b, m = properties_b
    print(n, m)
    g = nx.Graph()
    for i in range(n):
        g.add_edge((name_a, i), (name_b, rns.randint(m)), {'weight': 0.5})
    for i in range(m):
        g.add_edge((name_a, rns.randint(n)), (name_b, i), {'weight': 5.0})

    if l:
        if l > n * m:
            raise ValueError('l value greater than n * m')

        extra_edges = l - g.number_of_edges()
        print(extra_edges)
        if extra_edges > 0:
            for i in range(extra_edges):
                g.add_edge((name_a, rns.randint(n)), (name_b, rns.randint(m)), {'weight': 2.0})
    return g

