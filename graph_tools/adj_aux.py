from numpy import array, sort, argsort, zeros


def create_adj_matrix(in_list):
    if len(in_list[0]) == 2:
        weighted_flag = False
    elif len(in_list[0]) == 3:
        weighted_flag = True
    else:
        raise ValueError('in create_adj_matrix() ')

    ida_unique = set(map(lambda x: x[0], in_list))

    in_list_uni_a = list(filter(lambda x: x[0] in ida_unique, in_list))
    ida_unique = sort(list(ida_unique))
    inds_a = argsort(list(map(lambda x: x[0], in_list_uni_a)))
    inds_aa = argsort(inds_a)
    idb_unique = sort(list(set([item for sublist in in_list for item in sublist[1]])))
    inds_b_dict = dict((value, idx) for idx, value in enumerate(idb_unique))
    arr = zeros((len(ida_unique), len(idb_unique)))
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

