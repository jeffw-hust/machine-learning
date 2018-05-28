#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by jeffw on 2018/5/25

import pandas as pd
from kaggle.digit_recognizer.global_config import GlobalConfig
from sklearn.externals import joblib

# test data
test_data = pd.read_csv(GlobalConfig.INPUT_DIR+'test.csv')
test_data[test_data > 0] = 1

# predict
clf = joblib.load(GlobalConfig.MODEL_DIR + 'clf.model')
results = clf.predict(test_data)

# dump
df = pd.DataFrame(results)
df.index += 1
df.index.name = 'ImageId'
df.columns = ['Label']
df.to_csv(GlobalConfig.OUTPUT_DIR+'result.csv', header=True)

if __name__ == '__main__':
    pass
