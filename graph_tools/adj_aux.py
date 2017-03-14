from numpy import array, sort, argsort, zeros
import logging


def create_adj_matrix(in_list):
    """

    :param in_list: [id_a, [id_bs]] or [id_a, [id_bs], [weights]]
    :return: arr : adjacency matirix

    ida_unique :
    idb_unique :
    """

    if len(in_list[0]) == 2:
        weighted_flag = False
    elif len(in_list[0]) == 3:
        weighted_flag = True
    else:
        raise ValueError('in create_adj_matrix() : wrong shape of in_list')

    logging.info(' in create_adj_matrix(): len of in_list = {0}'.format(len(in_list)))

    # unique id_a
    ida_unique = set(map(lambda x: x[0], in_list))
    logging.info(' in create_adj_matrix(): len of ida_unique = {0}'.format(len(ida_unique)))

    # order of unique id_a in in_list
    counter_set = set(ida_unique)
    in_list_uni_a = []
    for x in in_list:
        if x[0] in counter_set:
            in_list_uni_a.append(x)
            counter_set -= {x[0]}

    logging.info(' in create_adj_matrix(): len of in_list_uni_a = {0}'.format(len(in_list_uni_a)))

    # sorted unique id_a
    ida_unique = sort(list(ida_unique))

    # indices that sort in_list_uni_a
    inds_a = argsort(list(map(lambda x: x[0], in_list_uni_a)))
    logging.info(' in create_adj_matrix(): max of inds_a = {0}'.format(max(inds_a)))

    # indices that turn ida_unique in in_list_uni_a first arg
    inds_aa = argsort(inds_a)
    logging.info(' in create_adj_matrix(): max of inds_aa = {0}'.format(max(inds_aa)))

    # list of unique id_bs
    idb_unique = sort(list(set([item for sublist in in_list for item in sublist[1]])))
    # map from id_b from unique_id_bs to index
    inds_b_dict = dict((value, idx) for idx, value in enumerate(idb_unique))

    arr = zeros((len(ida_unique), len(idb_unique)))
    logging.info(' in create_adj_matrix(): arr shape {0}'.format(arr.shape))

    for j, item in enumerate(in_list_uni_a):
        if weighted_flag:
            ida, idbs, weights = item
        else:
            ida, idbs = item
        tmp_arr = zeros(len(idb_unique))
        idb_present_indices = [inds_b_dict[x] for x in idbs]
        tmp_arr[idb_present_indices] = 1.0 if not weighted_flag else array(weights)
        arr[inds_aa[j]] = tmp_arr
    return arr, ida_unique, idb_unique

