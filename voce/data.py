"""Anomaly injection following ECMOD (Chen et al. 2023, github.com/scu-kdde/OAM-ECMOD-2023), the procedure
the SCoNE paper says it adheres to. The function data_perturbation_according_percentage below is copied
VERBATIM from ECMOD's methods.py (data_gen.py); only the wrapper (loading, min-max normalisation, view split,
seeding) is ours.

    from voce.data import prepare
    views, y = prepare('wdbc', '258', seed=1)          # views: list of N x d_v arrays, y: 1 = anomaly

The raw files are read from the folder DATA_DIR (default: datasets/ under the current folder); the README
of the repository lists the files and where to get them.
"""
import random
import numpy as np
import os
import scipy.io as sio

DATA_DIR = os.environ.get('VOCE_DATA', 'datasets')

RATIOS = {'258': (0.02, 0.05, 0.08), '555': (0.05, 0.05, 0.05), '852': (0.08, 0.05, 0.02)}
VIEW_NUM = 3   # UCI-style datasets are split into 3 views


# ---------------- copied verbatim from ECMOD methods.py ----------------
def data_perturbation_according_percentage(data_each_view, feature_num_each_view, label, attribute_percentage, attr_class_percentage, class_percentage):
    view_num = len(data_each_view)
    data_num = len(data_each_view[0]) # w：每个视图中的数据个数
    attribute_outlier_num = int(attribute_percentage * data_num)

    attr_class_outlier_num = int(attr_class_percentage * data_num)
    class_outlier_num = int(class_percentage * data_num)

    # outlier_num_each_type = int(data_num*gl.outlier_percent)
    # outlier_num_each_type_tmp = outlier_num_each_type

    attr_class_outlier_num = attr_class_outlier_num + (attr_class_outlier_num % 2)
    class_outlier_num = class_outlier_num + (class_outlier_num % 2)

    attribute_outlier = []

    new_label = []
    for i in range(0, data_num):
        new_label.append(label[i])

    # w：相当于产生了 attribute异常
    for i in range(0, view_num):
        attribute_outlier.append([])
        for j in range(0, attribute_outlier_num):
            attribute_outlier[i].append([]) # w: 第i个视图下加入 该视图的异常数目个新列表
            for k in range(0, feature_num_each_view[i]): # 每个新列表中加入 第i个视图下的特征数目个随机数字
                attribute_outlier[i][j].append(random.uniform(0, 1)) # 在第i个视图下的

    swap_view_num = int(view_num/2)

    # 产生 类异常
    swap_cnt = 0
    while swap_cnt < int(class_outlier_num/2): # 类异常
        random_index1 = random.randint(0, data_num-1)
        random_index2 = random.randint(0, data_num-1)
        if new_label[random_index1] == new_label[random_index2] or new_label[random_index1] == -1 or new_label[random_index2] == -1:
            continue
        # 随机选择一个视图，作为交换的视图
        random_swap_view_index = random.randint(0, view_num - 1)

        # 调试用
        before_1_tag = new_label[random_index1]
        before_2_tag = new_label[random_index2]
        before_1 = data_each_view[random_swap_view_index][random_index1]
        before_2 = data_each_view[random_swap_view_index][random_index2]

        data_each_view[random_swap_view_index][random_index1], data_each_view[random_swap_view_index][random_index2] = data_each_view[random_swap_view_index][random_index2], data_each_view[random_swap_view_index][random_index1]

        # 调试用
        after_1 = data_each_view[random_swap_view_index][random_index1]
        after_2 = data_each_view[random_swap_view_index][random_index2]
        new_label[random_index1] = -1
        new_label[random_index2] = -1
        after_1_tag = new_label[random_index1]
        after_2_tag = new_label[random_index2]
        swap_cnt += 1

    # 产生 类-属性异常
    swap_cnt = 0
    while swap_cnt < int(attr_class_outlier_num/2):
        random_index1 = random.randint(0, data_num - 1)
        random_index2 = random.randint(0, data_num - 1)
        if new_label[random_index1] == new_label[random_index2] or new_label[random_index1] == -1 or new_label[random_index2] == -1:
            continue
        # 随机选择一个视图，作为交换的视图
        random_swap_view_index2 = random.randint(0, view_num - 1)

        # 调试用
        before_attr_class_1_tag = new_label[random_index1]
        before_attr_class_2_tag = new_label[random_index2]
        before_attr_class_1 = data_each_view[random_swap_view_index2][random_index1]
        before_attr_class_2 = data_each_view[random_swap_view_index2][random_index2]

        data_each_view[random_swap_view_index2][random_index1], data_each_view[random_swap_view_index2][random_index2] = \
            data_each_view[random_swap_view_index2][random_index2], data_each_view[random_swap_view_index2][random_index1]
        new_label[random_index1] = -1
        new_label[random_index2] = -1

        # 调试用
        after_attr_class_1_tag = new_label[random_index1]
        after_attr_class_2_tag = new_label[random_index2]
        after_attr_class_1 = data_each_view[random_swap_view_index2][random_index1]
        after_attr_class_2 = data_each_view[random_swap_view_index2][random_index2]

        swap_cnt += 1
        for i in range(0, view_num): # w:另一半 是 后一半视图中的随机行的 所有列值随机赋值；相当于 attribute异常
            if i == random_swap_view_index2:
                continue
            for j in range(0, len(data_each_view[i][0])):
                data_each_view[i][random_index1][j] = random.uniform(0, 1)
                data_each_view[i][random_index2][j] = random.uniform(0, 1)

    # w: 以上截止： 完成了第三类异常的生成：class-attribute outliers

    # w：下面开始生成 attribute异常
    swap_cnt = 0
    while swap_cnt < attribute_outlier_num:
        random_index3 = random.randint(0, data_num - 1)
        if  new_label[random_index3] == -1:
            continue
        new_label[random_index3] = -1
        for i in range(0, view_num):
            data_each_view[i][random_index3] = attribute_outlier[i][swap_cnt]# 意思就是在每个视图下插入 参数异常的数据
        swap_cnt += 1

    for i in range(0, data_num):
        if new_label[i] == -1:
            new_label[i] = 1  # w：设置为1 means positive
        else:
            new_label[i] = 0

    return data_each_view, label, new_label
# ---------------- end of ECMOD code ----------------


def _load_raw(name):
    """Returns (single feature matrix or list of natural views, labels, natural_views flag)."""
    if name == 'zoo':      # animal name, 16 attributes, type
        rows = [l.strip().split(',') for l in open(DATA_DIR + '/zoo.data') if l.strip()]
        X = np.array([[float(v) for v in r[1:-1]] for r in rows]); y = np.array([int(r[-1]) for r in rows])
        return X, y, False
    if name == 'wdbc':     # ID, diagnosis (M/B), 30 features
        rows = [l.strip().split(',') for l in open(DATA_DIR + '/wdbc.data') if l.strip()]
        X = np.array([[float(v) for v in r[2:]] for r in rows]); y = np.array([1 if r[1] == 'M' else 0 for r in rows])
        return X, y, False
    if name == 'wine':     # class (1-3), 13 features
        a = np.loadtxt(DATA_DIR + '/wine.data', delimiter=',')
        return a[:, 1:], a[:, 0].astype(int), False
    if name == 'pima':     # 8 features, class (0/1); no longer on UCI, file from the ECMOD repo
        a = np.loadtxt(DATA_DIR + '/pima.csv', delimiter=',')
        return a[:, :-1], a[:, -1].astype(int), False
    if name == 'optdigits':    # optdigits.tra + optdigits.tes: 64 features, class (0-9)
        a = np.vstack([np.loadtxt(DATA_DIR + '/optdigits.tra', delimiter=','), np.loadtxt(DATA_DIR + '/optdigits.tes', delimiter=',')])
        return a[:, :-1], a[:, -1].astype(int), False
    if name == 'parkinsons':   # name, 22 features with 'status' (label) at column 17
        a = np.genfromtxt(DATA_DIR + '/parkinsons.data', delimiter=',', skip_header=1)[:, 1:]
        return np.delete(a, 16, axis=1), a[:, 16].astype(int), False
    if name == 'mnist':
        m = sio.loadmat(DATA_DIR + '/mnist.mat')
        return m['fea'].astype(np.float64), m['gt'].ravel().astype(int), False
    # natural multi-view data sets (ChuanbinZhang/Multi-view-datasets): X = V x 1 cell of N x d_v, y = labels
    NATURAL = {'caltech': 'Caltech101-7', 'bbcsport': 'BBCSport',
               'citeseer': 'CiteSeer', 'reuters': 'Reuters-1200'}
    if name in NATURAL:
        m = sio.loadmat(f'{DATA_DIR}/{NATURAL[name]}.mat')
        return [m['X'].flat[i].astype(np.float64) for i in range(m['X'].size)], m['y'].ravel().astype(int), True
    raise ValueError(name)


def _minmax(X):
    """Per-feature min-max to [0,1]; constant features -> 0.5 (as ECMOD's data_normalization)."""
    mn, mx = X.min(0), X.max(0); rng = mx - mn
    out = np.full_like(X, 0.5); ok = rng > 0
    out[:, ok] = (X[:, ok] - mn[ok]) / rng[ok]
    return out


def _rowz(v):
    """Per-sample standardisation within a view (each row: subtract its mean, divide by its std)."""
    m = v.mean(1, keepdims=True); sd = v.std(1, keepdims=True); sd[sd == 0] = 1.0
    return (v - m) / sd


# text data sets with standard per-document normalisation, applied to every method's input:
# '<name>_tn' = the same data and the same injected anomalies as '<name>' (same seed), then every sample is
# standardised within each view after injection
TEXT_NORMALISED = {'bbcsport_tn', 'reuters_tn'}


def prepare(name, ratio, seed):
    """Views and labels of data set `name` with anomalies injected at `ratio` ('258', '555' or '852') and seed.
    Names: zoo parkinsons wdbc wine pima optdigits caltech citeseer bbcsport reuters mnist; append '_tn' to
    BBCSport / Reuters for the per-sample z-score used in the paper (bbcsport_tn, reuters_tn)."""
    if name in TEXT_NORMALISED:
        views, y = prepare(name[:-3], ratio, seed)
        return [_rowz(v) for v in views], y
    random.seed(seed)
    X, y, natural = _load_raw(name)
    if natural:
        views = [_minmax(v) for v in X]
    else:
        X = _minmax(X)
        d = X.shape[1]; w = d // VIEW_NUM    # ECMOD cut_data_new: contiguous blocks, last view takes the rest
        views = [X[:, i * w:(i + 1) * w] for i in range(VIEW_NUM - 1)] + [X[:, (VIEW_NUM - 1) * w:]]
    # ECMOD swaps rows with tuple assignment, so each view must be a Python list of separate row objects
    views_list = [[row.copy() for row in v] for v in views]
    a, ac, c = RATIOS[ratio]
    out, _, new_label = data_perturbation_according_percentage(
        views_list, [v.shape[1] for v in views], list(y), a, ac, c)
    return [np.array([np.asarray(r, dtype=np.float64) for r in v]) for v in out], np.array(new_label)


# name used by the original experiment scripts
data_preparation_ecmod = prepare
