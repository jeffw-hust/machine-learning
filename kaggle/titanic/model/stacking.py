#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by jeffw on 2018/5/24

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from project.titanic.config.global_config import GlobalConfig
from project.titanic.feature.preprocessing import preprocessing
from project.titanic.feature.preprocessing import preprocessing_test_data


def train_and_predict(df_final, df_test, savepath):
    selection = ['Pclass', 'Sex', 'Embarked', 'age_level', 'title_level', 'family_size_level', 'MPSE']

    # train
    X = df_final[selection]
    Y = df_final['Survived']
    X_dummied = pd.get_dummies(X, columns=selection)

    # test
    df_test_selected = df_test[selection]
    df_test_dummied = pd.get_dummies(df_test_selected, columns=selection)

    df_test_dummied.shape, X_dummied.shape

    ## stacking
    n_train=X_dummied.shape[0]
    n_test=df_test_dummied.shape[0]
    kf=StratifiedKFold(n_splits=5,random_state=1,shuffle=True)


    def get_oof(clf, X, y, test_X):
        oof_train = np.zeros((n_train,))
        oof_test_mean = np.zeros((n_test,))
        # 5 is kf.split
        oof_test_single = np.empty((kf.get_n_splits(), n_test))
        for i, (train_index, val_index) in enumerate(kf.split(X, y)):
            kf_X_train = X.iloc[train_index]
            kf_y_train = y.iloc[train_index]
            kf_X_val = X.iloc[val_index]

            clf.fit(kf_X_train, kf_y_train)

            oof_train[val_index] = clf.predict(kf_X_val)
            oof_test_single[i, :] = clf.predict(test_X)
        # oof_test_single, 将生成一个5行*n_test列的predict value。那么mean(axis=0), 将对5行，每列的值进行求mean。然后reshape返回
        oof_test_mean = oof_test_single.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test_mean.reshape(-1, 1)

    lr = LogisticRegression(C=8, random_state=0)
    svc = SVC(C=12, gamma=0.01, random_state=0, probability=True)
    rf = RandomForestClassifier(max_depth=5, max_features=5, n_estimators=35, random_state=0)
    gbdt = GradientBoostingClassifier(learning_rate=0.1, max_depth=2, max_features=3, n_estimators=200, random_state=0)
    bagging = BaggingClassifier(lr, n_estimators=90, random_state=0)
    abc = AdaBoostClassifier(learning_rate=0.7, n_estimators=160, random_state=0)

    lr_train, lr_test = get_oof(lr, X_dummied, Y, df_test_dummied)
    svc_train, svc_test = get_oof(svc, X_dummied, Y, df_test_dummied)
    rf_train, rf_test = get_oof(rf, X_dummied, Y, df_test_dummied)
    gbdt_train, gbdt_test = get_oof(gbdt, X_dummied, Y, df_test_dummied)
    bagging_train, bagging_test = get_oof(bagging, X_dummied, Y, df_test_dummied)
    abc_train, abc_test = get_oof(abc, X_dummied, Y, df_test_dummied)

    y_train_pred_stack = np.concatenate([lr_train, svc_train, rf_train, gbdt_train, bagging_train, abc_train], axis=1)
    y_train_stack = Y
    y_test_pred_stack = np.concatenate([lr_test, svc_test, rf_test, gbdt_test, bagging_test, abc_test], axis=1)

    scores = cross_val_score(RandomForestClassifier(random_state=0, n_estimators=50), y_train_pred_stack, y_train_stack,
                             cv=5, scoring='roc_auc')
    print scores.mean(), scores

    y_pred = RandomForestClassifier(random_state=0, n_estimators=100).fit(y_train_pred_stack, y_train_stack).predict(
        y_test_pred_stack)
    result_df = pd.DataFrame({'PassengerId': df_test_dummied.index, 'Survived':y_pred}).set_index('PassengerId')
    result_df.to_csv(savepath)


if __name__ == '__main__':
    df_train, gscv = preprocessing(GlobalConfig.INPUT_DIR + 'train.csv')
    df_test = preprocessing_test_data(GlobalConfig.INPUT_DIR + 'test.csv', gscv)
    train_and_predict(df_train, df_test, GlobalConfig.OUTPUT_DIR + 'result.csv')
    print "--------end--------"