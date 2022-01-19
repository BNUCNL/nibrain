import os
import time
import math
import numpy as np
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from magicbox.io.io import CiftiReader, save2cifti
from cxy_visual_dev.lib.predefine import proj_dir, Atlas,\
    get_rois, mmp_map_file

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'predict')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def PC12_predict_ROI1():
    """
    用HCPY-M+T_MMP-vis3-{Hemi}_zscore1_PCA-subj的PC1和PC2
    预测HCP MMP1.0的脑区。将脑区内顶点以3:1的比例拆分成train和test set
    基于train set用logistic回归和linear SVC，基于各自设定的参数范围做grid search
    找到最优组合后，基于整个train set拟合模型，然后在test set上得到分数。
    数据拆分策略使用StratifiedKFold
    数据在拆分后要经过StandardScaler的处理
    """
    vis_name = 'MMP-vis3-R'
    atlas = Atlas('HCP-MMP')
    feat_file = pjoin(
        anal_dir,
        f'decomposition/HCPY-M+T_{vis_name}_zscore1_PCA-subj.dscalar.nii')

    # prepare X, y
    mask = atlas.get_mask(get_rois(vis_name))[0]
    X = nib.load(feat_file).get_fdata()[:2, mask].T
    y = atlas.maps[0, mask]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=True, random_state=7, stratify=y)

    # prepare pipeline and GridSearch
    pipe = Pipeline([('preprocesser', StandardScaler()),
                     ('classifier', LogisticRegression())])
    param_grid = [
        {'classifier': [LogisticRegression(penalty='none')],
         'classifier__solver': ['saga', 'sag', 'lbfgs', 'newton-cg']},
        {'classifier': [LogisticRegression(penalty='l1')],
         'classifier__solver': ['saga', 'liblinear'],
         'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
        {'classifier': [LogisticRegression(penalty='l2')],
         'classifier__solver': ['sag', 'saga', 'liblinear', 'lbfgs', 'newton-cg'],
         'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
        {'classifier': [SVC(kernel='linear')],
         'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    ]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='accuracy')

    # computation
    grid.fit(X_train, y_train)

    # output
    print('best estimator:\n', grid.best_estimator_)
    print('\nbest cross-validation score:', grid.best_score_)
    print('\nTest-set score:', grid.score(X_test, y_test))


def PC12_predict_ROI2():
    """
    1. 以StratifiedKFold划分Train和Test（n_split=10），对于每次split，
    用GridSearchCV在Train上搜索超参数（划分Train和Validation的方法
    也是StratifiedKFold (n_split=10)），并用最优超参数在整个Train上训练模型，
    然后得到在Test上的预测。（n_split取大一点可以让训练集大一点，提高test上的准确率。
    内部的CV用的是总体准确率来选超参数，所以不用担心有些类的validation集太小。
    而外部是汇总所有test的预测值之后再计算总体和类别准确率，所以也不怕有些类的test集太小）
    2. 所有split的基于Test的预测值合在一起就可以得到所有顶点的预测值（可以说是相当严格了）。
    基于这些预测值可以计算总体准确率和类别准确率；
    3. 然后展示这个预测结果的脑图，对比MMP的ground truth脑图，算个相关。
    """
    n_split = 10
    vis_name = 'MMP-vis3-R'
    mask = Atlas('HCP-MMP').get_mask(get_rois(vis_name))[0]
    feat_file = pjoin(
        anal_dir,
        f'decomposition/HCPY-M+T_{vis_name}_zscore1_PCA-subj.dscalar.nii')
    reader = CiftiReader(mmp_map_file)
    out_pkl = pjoin(work_dir, f'PC12_predict_ROI2_{vis_name}.pkl')
    out_cii = pjoin(work_dir, f'PC12_predict_ROI2_{vis_name}.dlabel.nii')

    # prepare X, y
    X = nib.load(feat_file).get_fdata()[:2, mask].T
    y = reader.get_data()[0, mask]

    # prepare parameters for GridSearch
    Cs = [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
    param_grid = [
        {'classifier': [LogisticRegression(penalty='none')],
         'classifier__solver': ['saga', 'sag', 'lbfgs', 'newton-cg']},
        {'classifier': [LogisticRegression(penalty='l1')],
         'classifier__solver': ['saga', 'liblinear'],
         'classifier__C': Cs},
        {'classifier': [LogisticRegression(penalty='l2')],
         'classifier__solver': ['sag', 'saga', 'liblinear', 'lbfgs', 'newton-cg'],
         'classifier__C': Cs},
        # 这样指定linearSVC好像只能是L2正则，以后可以直接用LinearSVC这个接口，是可以调正则方式的
        {'classifier': [SVC(kernel='linear')],
         'classifier__C': Cs}
    ]

    # loop n_split train-test splits
    out_dict = {'model': [], 'train_score': [], 'test_score': []}
    out_map = np.zeros(reader.full_data.shape, np.uint16)
    y_pred = np.zeros_like(y)
    skf = StratifiedKFold(n_split, shuffle=True, random_state=7)
    split_idx = 1
    for train_indices, test_indices in skf.split(X, y):
        time1 = time.time()
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        pipe = Pipeline([('preprocesser', StandardScaler()),
                         ('classifier', LogisticRegression())])
        grid = GridSearchCV(pipe, param_grid, cv=n_split, scoring='accuracy')
        grid.fit(X_train, y_train)
        y_pred[test_indices] = grid.predict(X_test)

        out_dict['model'].append(grid)
        out_dict['train_score'].append(grid.score(X_train, y_train))
        out_dict['test_score'].append(grid.score(X_test, y_test))
        print(f'Finished {split_idx}/{n_split}, '
              f'cost {time.time() - time1} seconds.')
        split_idx += 1
    out_map[0, mask] = y_pred

    # output
    pkl.dump(out_dict, open(out_pkl, 'wb'))
    lbl_tab = nib.cifti2.Cifti2LabelTable()
    lbl_tab_old = reader.label_tables()[0]
    for k in np.unique(out_map):
        lbl_tab[k] = lbl_tab_old[k]
    save2cifti(out_cii, out_map, reader.brain_models(),
               label_tables=[lbl_tab])


def PC12_predict_ROI3():
    """
    1. 以StratifiedKFold划分Train和Test（n_split=5），对于每次split，
    用GridSearchCV在Train上搜索超参数（划分Train和Validation的方法
    也是StratifiedKFold (n_split=5)），并用最优超参数在整个Train上训练模型，
    然后得到在Test上的预测。（n_split取大一点可以让训练集大一点，提高test上的准确率。
    内部的CV用的是总体准确率来选超参数，所以不用担心有些类的validation集太小。
    而外部是汇总所有test的预测值之后再计算总体和类别准确率，所以也不怕有些类的test集太小）
    由之前的结果发现，在训练过程中，小区由于样本少，对最终正确率影响较小，因此训练出的模型
    对有些小区的预测正确率很低。于是我决定在每次split的训练集中为每个区整体复制样本点，直到其数量
    不超过最大的区的数量，这样在优化目标函数的时候，小区的重要性也在，但是信息量应该是不变的。
    不管在训练集上怎么操作，只要测试集的精度高，并且类别准确率合理，那模型就是成功的！
    2. 所有split的基于Test的预测值合在一起就可以得到所有顶点的预测值（可以说是相当严格了）。
    基于这些预测值可以计算总体准确率和类别准确率；
    3. 然后展示这个预测结果的脑图，对比MMP的ground truth脑图，算个相关。
    """
    n_split = 5
    vis_name = 'MMP-vis3-R'
    mask = Atlas('HCP-MMP').get_mask(get_rois(vis_name))[0]
    feat_file = pjoin(
        anal_dir,
        f'decomposition/HCPY-M+T_{vis_name}_zscore1_PCA-subj.dscalar.nii')
    reader = CiftiReader(mmp_map_file)
    out_pkl = pjoin(work_dir, f'PC12_predict_ROI3_{vis_name}.pkl')
    out_cii = pjoin(work_dir, f'PC12_predict_ROI3_{vis_name}.dlabel.nii')

    # prepare X, y
    X = nib.load(feat_file).get_fdata()[:2, mask].T
    y = reader.get_data()[0, mask]

    # prepare parameters for GridSearch
    Cs = [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
    param_grid = [
        {'classifier': [LogisticRegression(penalty='none', max_iter=1000)],
         'classifier__solver': ['saga', 'lbfgs']},
        {'classifier': [LogisticRegression(penalty='l2', max_iter=1000)],
         'classifier__solver': ['saga', 'lbfgs'],
         'classifier__C': Cs},
    ]

    # loop n_split train-test splits
    out_dict = {'model': [], 'train_score': [], 'test_score': [], 'train_score_new': []}
    out_map = np.zeros(reader.full_data.shape, np.uint16)
    y_pred = np.zeros_like(y)
    skf = StratifiedKFold(n_split, shuffle=True, random_state=7)
    split_idx = 1
    for train_indices, test_indices in skf.split(X, y):
        time1 = time.time()
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        labels = np.unique(y_train)
        label_sizes = [np.sum(y_train == lbl) for lbl in labels]
        lbl_sz_max = np.max(label_sizes)
        X_train_new = np.zeros((0, X_train.shape[1]), np.float64)
        y_train_new = np.zeros(0, np.float64)
        label_sizes_new = []
        for lbl, lbl_sz in zip(labels, label_sizes):
            idx_vec = y_train == lbl
            ratio = math.floor(lbl_sz_max / lbl_sz)
            X_train_new = np.r_[X_train_new, np.tile(X_train[idx_vec], (ratio, 1))]
            y_train_new = np.r_[y_train_new, np.tile(y_train[idx_vec], (ratio,))]
            label_sizes_new.append(lbl_sz * ratio)
        print('label size new (max):', np.max(label_sizes_new))
        print('label size new (min):', np.min(label_sizes_new))
        print('X_train_new.shape:', X_train_new.shape)
        print('y_train_new.shape:', y_train_new.shape)

        pipe = Pipeline([('preprocesser', StandardScaler()),
                         ('classifier', LogisticRegression())])
        cv = StratifiedKFold(n_split, shuffle=True, random_state=7)
        grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='accuracy')
        grid.fit(X_train_new, y_train_new)
        y_pred[test_indices] = grid.predict(X_test)

        out_dict['model'].append(grid)
        out_dict['train_score'].append(grid.score(X_train, y_train))
        out_dict['train_score_new'].append(grid.score(X_train_new, y_train_new))
        out_dict['test_score'].append(grid.score(X_test, y_test))
        print(f'Finished {split_idx}/{n_split}, '
              f'cost {time.time() - time1} seconds.')
        split_idx += 1
    out_map[0, mask] = y_pred

    # output
    pkl.dump(out_dict, open(out_pkl, 'wb'))
    lbl_tab = nib.cifti2.Cifti2LabelTable()
    lbl_tab_old = reader.label_tables()[0]
    for k in np.unique(out_map):
        lbl_tab[k] = lbl_tab_old[k]
    save2cifti(out_cii, out_map, reader.brain_models(),
               label_tables=[lbl_tab])


if __name__ == '__main__':
    # PC12_predict_ROI1()
    # PC12_predict_ROI2()
    PC12_predict_ROI3()
