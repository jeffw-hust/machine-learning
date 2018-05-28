#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by jeffw on 2018/5/23

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# disable copy warning in pandas
pd.options.mode.chained_assignment = None  # default='warn'
# disable sklearn deprecation warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os


### data clean
path='/Users/jeffw/workspace/jeffwang/github/machine-learning/project/titanic/'
df = pd.read_csv(path + 'train.csv').set_index('PassengerId')

# feature Pclass Sex

# feature Embarked
df.Embarked.fillna(value='S', inplace=True)

df['family_size'] = df.SibSp + df.Parch
df['family_size_level'] = pd.cut(df.family_size, bins=[-1,0, 3.5, 12], labels=['alone', 'middle', 'large'])

# feature title_level
df['title'] = df.Name.apply(lambda x: x.split(', ')[1].split('. ')[0])
df['title'].replace(['Mme', 'Ms', 'Mlle'], ['Mrs', 'Miss', 'Miss'], inplace = True)
df['title_level'] = df.title.apply(lambda x: 'rare' if x not in ['Mr', 'Miss', 'Mrs', 'Master'] else x)
df.title[(df.title_level == 'Miss') & (df.Age < 18)] = 'Mister'
df.title_level[(df.title_level == 'Miss') & (df.Age < 18)] = 'Mister'

# feature age_level
df['age_level'] = pd.cut(df.Age, bins=[0, 15, 60, 100], labels=['child', 'middle', 'older'])
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

#train set
df_age_train = df[df.Age.notnull()]
df_age = df_age_train[['Pclass', 'Sex', 'family_size', 'title_level', 'age_level']]
X_age_dummied = pd.get_dummies(df_age.drop(columns='age_level'), columns=['Pclass', 'Sex', 'title_level'])
Y_age = df_age['age_level']

#train
clf = RandomForestClassifier(random_state=0)
params = {'n_estimators': range(6, 14), 'max_features': [2, 3, 4] }
gscv = GridSearchCV(estimator=clf, param_grid=params, scoring='f1_micro', n_jobs=1, cv=5)
gscv.fit(X_age_dummied, Y_age)
gscv.best_score_, gscv.best_params_, gscv.best_estimator_.feature_importances_

df_age_test = df[df.Age.isnull()]
ab = df_age_test[['Pclass', 'Sex', 'family_size', 'title_level', 'age_level']]
X_age_dummied_test = pd.get_dummies(ab.drop(columns='age_level'), columns=['Pclass', 'Sex', 'title_level'])
X_age_dummied_test['title_level_Mister'] = np.zeros(len(X_age_dummied_test))
X_age_dummied.shape, X_age_dummied_test.shape
X_age_dummied.columns, X_age_dummied_test.columns

df_age_test.age_level = gscv.predict(X_age_dummied_test)
df_final = pd.concat([df_age_test, df_age_train]).sort_index()

df_final['MPSE'] = np.ones(len(df_final))
df_final.MPSE[(df_final.title_level == 'Mr') & (df_final.Pclass == 3) & (df_final.Sex == 'male') & (df_final.Embarked == 'S') & (df_final.family_size_level.isin(['alone', 'middle']))] = 4
df_final.MPSE[(df_final.title_level == 'Mr') & (df_final.Pclass == 2) & (df_final.Sex == 'male') & (df_final.Embarked == 'S') & (df_final.family_size_level.isin(['middle', 'alone']))] = 3
df_final.MPSE[(df_final.title_level == 'Mr') & (df_final.Pclass == 1) & (df_final.Sex == 'male') & (df_final.Embarked == 'S') & (df_final.family_size_level.isin(['middle', 'alone']))] = 2

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

lr = LogisticRegression(C=8, random_state=0)
knn = KNeighborsClassifier(n_neighbors=9)
svc = SVC(C=12, gamma=0.01, random_state=0)
mlp = MLPClassifier(alpha=0.001, hidden_layer_sizes=(50, 80), solver='lbfgs', random_state=0)
rf = RandomForestClassifier(max_depth=5, max_features=5, n_estimators=35, random_state=0)
gbdt = GradientBoostingClassifier(learning_rate=0.1, max_depth=2, max_features=3, n_estimators=200, random_state=0)
bagging = BaggingClassifier(lr, n_estimators=90, random_state=0)
abc = AdaBoostClassifier(learning_rate=0.7, n_estimators=160, random_state=0)

names = ['LR', 'KNN', 'SVC', 'MLP', 'RF', 'GBDT', 'Bagging', 'AdaB']
models = [lr, knn, svc, mlp, rf, gbdt, bagging, abc]


selection = ['Pclass', 'Sex', 'Embarked', 'age_level', 'title_level', 'family_size_level', 'MPSE']
X = df_final[selection]
Y = df_final['Survived']
X_dummied = pd.get_dummies(X, columns=selection)

result_scores = []
for name, model in zip(names, models):
    scores = cross_val_score(model, X_dummied, Y, cv=5, scoring='roc_auc')
    result_scores.append(scores.mean())
    print('{} has a mean score {:.4f} based on {}'.format(name, scores.mean(), scores))

from sklearn.ensemble import VotingClassifier
names = ['LR', 'KNN', 'SVC', 'MLP', 'RF', 'GBDT', 'Bagging', 'AdaB']

lr = LogisticRegression(C=8, random_state=0)
svc = SVC(C=12, gamma=0.01, random_state=0, probability=True)
rf = RandomForestClassifier(max_depth=5, max_features=5, n_estimators=35, random_state=0)
gbdt = GradientBoostingClassifier(learning_rate=0.1, max_depth=2, max_features=3, n_estimators=200, random_state=0)
bagging = BaggingClassifier(lr, n_estimators=90, random_state=0)
abc = AdaBoostClassifier(learning_rate=0.7, n_estimators=160, random_state=0)

# 直接投票，票数多的获胜。
vc_hard = VotingClassifier(estimators=[('LR', lr), ('SVC', svc), ('GBDT', gbdt), ('Bagging', bagging), ('AdaB', abc)], voting='hard')
# 参数里说，soft更加适用于已经调制好的base learners，基于每个learner输出的概率。知乎文章里讲，Soft一般表现的更好。
vc_soft = VotingClassifier(estimators=[('LR', lr), ('SVC', svc), ('RF', rf), ('GBDT', gbdt), ('Bagging', bagging), ('AdaB', abc)], voting='soft')

# 'vc hard:', cross_val_score(vc_hard, X_dummied, Y, cv=5, scoring='roc_auc').mean(),\
'vc soft:', cross_val_score(vc_soft, X_dummied, Y, cv=5, scoring='roc_auc').mean()

#test
df_test = pd.read_csv(path + 'test.csv').set_index('PassengerId')
df_test.info()

df_test[df_test.Fare.isna()]
# 由于是三等舱，这里简单的填入三等舱的平均值好了。
df_test.Fare[df_test.Fare.isna()] = df_test.Fare[df_test.Pclass == 3].mean()

df_test['title'] = df_test.Name.apply(lambda x: x.split(', ')[1].split('. ')[0])
# Major, 少校；Lady，贵妇；Sir，子爵; Capt, 上尉；the Countess，伯爵夫人；Col，上校。Dr,医生？
df_test['title'].replace(['Mme', 'Ms', 'Mlle', 'Dona'], ['Mrs', 'Miss', 'Miss', 'Don'], inplace = True)
df_test['title_level'] = df_test.title.apply(lambda x: 'rare' if x not in ['Mr', 'Miss', 'Mrs', 'Master'] else x)
df_test.title_level[(df_test.title_level == 'Miss') & (df_test.Age < 18)] = 'Mister'
df_test.title[(df_test.title_level == 'Miss') & (df_test.Age < 18)] = 'Mister'

df_test['title_level'].value_counts()

df_test['family_size'] = df_test.SibSp + df_test.Parch
df_test['family_size_level'] = pd.cut(df_test.family_size, bins=[-1,0, 3.5, 12], labels=['alone', 'middle', 'large'])


df_test['age_level'] = pd.cut(df_test.Age, bins=[0, 15, 60, 100], labels=['child', 'middle', 'older'])
df_test.age_level.value_counts()

df_age_X_na = df_test[df_test.Age.isna()][['Pclass', 'Sex', 'family_size', 'title_level']].copy()
df_age_X_na_dummied = pd.get_dummies(df_age_X_na, columns=['Pclass', 'Sex', 'title_level'])

df_age_X_na_dummied['title_level_Mister'] = np.zeros(len(df_age_X_na_dummied))
df_age_X_na_dummied['title_level_rare'] = np.zeros(len(df_age_X_na_dummied))

age_level_pred = gscv.predict(df_age_X_na_dummied)
df_test.age_level.fillna(pd.Series(age_level_pred, index=df_age_X_na.index), inplace=True)
df_test.info()

df_test['MPSE'] = np.ones(len(df_test))
df_test.MPSE[(df_test.title_level == 'Mr') & (df_test.Pclass == 3) & (df_test.Sex == 'male') \
                   & (df_test.Embarked == 'S') & (df_test.family_size_level.isin(['alone', 'middle']))] = 4
df_test.MPSE[(df_test.title_level == 'Mr') & (df_test.Pclass == 2) & (df_test.Sex == 'male') \
                   & (df_test.Embarked == 'S') & (df_test.family_size_level.isin(['middle', 'alone']))] = 3
df_test.MPSE[(df_test.title_level == 'Mr') & (df_test.Pclass == 1) & (df_test.Sex == 'male') \
                   & (df_test.Embarked == 'S') & (df_test.family_size_level.isin(['middle', 'alone']))] = 2

selection = ['Pclass', 'Sex', 'Embarked', 'age_level', 'title_level', 'family_size_level', 'MPSE']
df_test_selected = df_test[selection]
df_test_dummied = pd.get_dummies(df_test_selected, columns=selection)
df_test_dummied.shape, X_dummied.shape


## stacking
from sklearn.model_selection import StratifiedKFold
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
gbdt_train, gbdt_test=get_oof(gbdt, X_dummied, Y, df_test_dummied)
bagging_train, bagging_test = get_oof(bagging, X_dummied, Y, df_test_dummied)
abc_train, abc_test = get_oof(abc, X_dummied, Y, df_test_dummied)

y_train_pred_stack = np.concatenate([lr_train, svc_train, rf_train, gbdt_train, bagging_train, abc_train], axis=1)
y_train_stack = Y
y_test_pred_stack = np.concatenate([lr_test, svc_test, rf_test, gbdt_test, bagging_test, abc_test], axis=1)

y_train_pred_stack.shape, y_train_stack.shape, y_test_pred_stack.shape


scores = cross_val_score(RandomForestClassifier(random_state=0, n_estimators=50), y_train_pred_stack, y_train_stack, cv=5, scoring='roc_auc')
scores.mean(), scores


y_pred = RandomForestClassifier(random_state=0, n_estimators=100).fit(y_train_pred_stack, y_train_stack).predict(y_test_pred_stack)


result_df = pd.DataFrame({'PassengerId': df_test_dummied.index, 'Survived':y_pred}).set_index('PassengerId')
result_df.to_csv('predicted_survived.csv')