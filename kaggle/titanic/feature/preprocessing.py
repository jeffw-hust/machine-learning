#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by jeffw on 2018/5/24

import numpy as np
import pandas as pd
from kaggle.titanic.config.global_config import GlobalConfig
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import warnings

# disable copy warning in pandas
pd.options.mode.chained_assignment = None  # default='warn'
# disable sklearn deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning)


def preprocessing(filepath):
    df = pd.read_csv(filepath).set_index('PassengerId')

    # Embarked fea
    df.Embarked.fillna(value='S', inplace=True)

    # family fea
    df['family_size'] = df.SibSp + df.Parch
    df['family_size_level'] = pd.cut(df.family_size, bins=[-1,0, 3.5, 12], labels=['alone', 'middle', 'large'])

    # title fea
    df['title'] = df.Name.apply(lambda x: x.split(', ')[1].split('. ')[0])
    df['title'].replace(['Mme', 'Ms', 'Mlle'], ['Mrs', 'Miss', 'Miss'], inplace = True)
    df['title_level'] = df.title.apply(lambda x: 'rare' if x not in ['Mr', 'Miss', 'Mrs', 'Master'] else x)
    df.title[(df.title_level == 'Miss') & (df.Age < 18)] = 'Mister'
    df.title_level[(df.title_level == 'Miss') & (df.Age < 18)] = 'Mister'

    # age fea
    df['age_level'] = pd.cut(df.Age, bins=[0, 15, 60, 100], labels=['child', 'middle', 'older'])
    # age_level train_set
    df_age_train = df[df.Age.notnull()]
    df_age = df_age_train[['Pclass', 'Sex', 'family_size', 'title_level', 'age_level']]
    X_age_dummied = pd.get_dummies(df_age.drop(columns='age_level'), columns=['Pclass', 'Sex', 'title_level'])
    Y_age = df_age['age_level']
    # train
    clf = RandomForestClassifier(random_state=0)
    params = {'n_estimators': range(6, 14), 'max_features': [2, 3, 4] }
    gscv = GridSearchCV(estimator=clf, param_grid=params, scoring='f1_micro', n_jobs=1, cv=5)
    gscv.fit(X_age_dummied, Y_age)

    # age_level fillna by predict
    df_age_test = df[df.Age.isnull()]
    ab = df_age_test[['Pclass', 'Sex', 'family_size', 'title_level', 'age_level']]
    X_age_dummied_test = pd.get_dummies(ab.drop(columns='age_level'), columns=['Pclass', 'Sex', 'title_level'])
    X_age_dummied_test['title_level_Mister'] = np.zeros(len(X_age_dummied_test))
    df_age_test.age_level = gscv.predict(X_age_dummied_test)

    df_final = pd.concat([df_age_test, df_age_train]).sort_index()

    # MPSE fea
    df_final['MPSE'] = np.ones(len(df_final))
    df_final.MPSE[(df_final.title_level == 'Mr') & (df_final.Pclass == 3) & (df_final.Sex == 'male') & (df_final.Embarked == 'S') & (df_final.family_size_level.isin(['alone', 'middle']))] = 4
    df_final.MPSE[(df_final.title_level == 'Mr') & (df_final.Pclass == 2) & (df_final.Sex == 'male') & (df_final.Embarked == 'S') & (df_final.family_size_level.isin(['middle', 'alone']))] = 3
    df_final.MPSE[(df_final.title_level == 'Mr') & (df_final.Pclass == 1) & (df_final.Sex == 'male') & (df_final.Embarked == 'S') & (df_final.family_size_level.isin(['middle', 'alone']))] = 2
    return df_final, gscv


def preprocessing_test_data(filepath, gscv):
    df_test = pd.read_csv(filepath).set_index('PassengerId')
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
    return df_test


if __name__ == '__main__':
    df_final, gscv = preprocessing(GlobalConfig.INPUT_DIR + 'train.csv')
    print df_final.head()
    print '-----------------------------'
    print gscv.best_score_, gscv.best_params_, gscv.best_estimator_.feature_importances_
