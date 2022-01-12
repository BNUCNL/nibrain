import os
import numpy as np
import nibabel as nib
from os.path import join as pjoin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from cxy_visual_dev.lib.predefine import proj_dir, Atlas, get_rois

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


if __name__ == '__main__':
    PC12_predict_ROI1()
