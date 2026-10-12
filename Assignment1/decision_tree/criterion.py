"""
criterion
"""
import math

def get_criterion_function(criterion):
    if criterion == "info_gain":
        return __info_gain
    elif criterion == "info_gain_ratio":
        return __info_gain_ratio
    elif criterion == "gini":
        return __gini_index
    elif criterion == "error_rate":
        return __error_rate


def __label_stat(y, l_y, r_y):
    """Count the number of labels of nodes"""
    left_labels = {}
    right_labels = {}
    all_labels = {}
    for t in y.reshape(-1):
        if t not in all_labels:
            all_labels[t] = 0
        all_labels[t] += 1
    for t in l_y.reshape(-1):
        if t not in left_labels:
            left_labels[t] = 0
        left_labels[t] += 1
    for t in r_y.reshape(-1):
        if t not in right_labels:
            right_labels[t] = 0
        right_labels[t] += 1

    return all_labels, left_labels, right_labels


def _shannon_entropy(label_stat):
    """Shannon entropy of a {label: count} dict."""
    total = sum(label_stat.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for cnt in label_stat.values():
        p = cnt / total
        if p > 0:
            ent -= p * math.log2(p)
    return ent


def _gini_impurity(label_stat):
    """Gini impurity of a {label: count} dict."""
    total = sum(label_stat.values())
    if total == 0:
        return 0.0
    gini = 1.0
    for cnt in label_stat.values():
        p = cnt / total
        gini -= p * p
    return gini


def _classification_error(label_stat):
    """Classification error (1 - max_k p_k) of a {label: count} dict."""
    total = sum(label_stat.values())
    if total == 0:
        return 0.0
    return 1.0 - max(label_stat.values()) / total


def __info_gain(y, l_y, r_y):
    """
    Calculate the info gain

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
    info_gain = 0.0
    # =============== TODO (students) ===============
    # 信息增益 = 父节点熵 - 子节点熵的加权平均
    n = sum(all_labels.values())
    n_l = sum(left_labels.values())
    n_r = sum(right_labels.values())
    info_gain = (_shannon_entropy(all_labels)
                 - (n_l / n) * _shannon_entropy(left_labels)
                 - (n_r / n) * _shannon_entropy(right_labels))
    # ===============================================
    return info_gain


def __info_gain_ratio(y, l_y, r_y):
    """
    Calculate the info gain ratio

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    info_gain = __info_gain(y, l_y, r_y)
    # =============== TODO (students) ===============
    # 分裂信息：按左右子节点样本占比计算的熵
    n = float(len(y.reshape(-1)))
    n_l = float(len(l_y.reshape(-1)))
    n_r = float(len(r_y.reshape(-1)))
    split_info = 0.0
    if n_l > 0:
        p_l = n_l / n
        split_info -= p_l * math.log2(p_l)
    if n_r > 0:
        p_r = n_r / n
        split_info -= p_r * math.log2(p_r)
    # 某一侧为空时分裂信息为 0，为避免除零返回 0
    if split_info == 0.0:
        return 0.0
    return info_gain / split_info
    # ===============================================


def __gini_index(y, l_y, r_y):
    """
    Calculate the gini index

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
    before = 0.0
    after = 0.0

    # =============== TODO (students) ===============
    # 基尼指数 = 分裂前 Gini - 分裂后子节点 Gini 的加权平均
    n = sum(all_labels.values())
    n_l = sum(left_labels.values())
    n_r = sum(right_labels.values())
    before = _gini_impurity(all_labels)
    after = ((n_l / n) * _gini_impurity(left_labels)
             + (n_r / n) * _gini_impurity(right_labels))
    # ===============================================
    return before - after


def __error_rate(y, l_y, r_y):
    """Calculate the error rate"""
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
    before = 0.0
    after = 0.0

    # =============== TODO (students) ===============
    # 分类误差率 = 分裂前误差率 - 分裂后子节点误差率的加权平均
    n = sum(all_labels.values())
    n_l = sum(left_labels.values())
    n_r = sum(right_labels.values())
    before = _classification_error(all_labels)
    after = ((n_l / n) * _classification_error(left_labels)
             + (n_r / n) * _classification_error(right_labels))
    # ===============================================
    return before - after
