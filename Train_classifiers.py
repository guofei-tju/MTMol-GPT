#!/usr/bin/env python
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn import metrics
from sklearn.metrics import accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
from sklearn.svm import SVC


def evaluation(outputs, labels):
    outputs[outputs >= 0.5] = 1  # 大于等于0.5为正面
    outputs[outputs < 0.5] = 0  # 小于0.5为负面
    correct_cnt = torch.sum(torch.eq(outputs, labels)).item()
    return correct_cnt

def train_prior():
    target = 'HTR1A'
    train_path = f'./data/EXCAPE-DB/{target}_all_train.csv'
    test_path = f'./data/EXCAPE-DB/{target}_all_test.csv'

    train_data = pd.read_csv(train_path, index_col=False)
    test_data = pd.read_csv(test_path, index_col=False)

    X_train_smiles = train_data['SMILES'].values
    X_test_smiles = test_data['SMILES'].values
    y_train = train_data['Targets'].values.astype(int)
    y_test = test_data['Targets'].values.astype(int)

    X_train = [Chem.MolFromSmiles(s) for s in X_train_smiles]
    X_train = [x for x in X_train if x is not None]
    X_train = np.array([AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048).ToList() for x in X_train])

    X_test = [Chem.MolFromSmiles(s) for s in X_test_smiles]
    X_test = [x for x in X_test if x is not None]
    X_test = np.array([AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048).ToList() for x in X_test])

    # define the classifier
    svm_rbf = SVC(C=10, kernel="rbf", probability=True)
    svm_rbf.fit(X_train, y_train)

    print("training accuracy: " + str(accuracy_score(y_train, svm_rbf.predict(X_train))))
    print("training auc: " + str(metrics.roc_auc_score(y_train, svm_rbf.predict_log_proba(X_train)[:,1])))

    print("testing accuracy: " + str(accuracy_score(y_test, svm_rbf.predict(X_test))))
    print("testing auc: " + str(metrics.roc_auc_score(y_test, svm_rbf.predict_proba(X_test)[:,1])))

    joblib.dump(svm_rbf, f'./results/model/model_clf/{target}_svm_rbf_p.pkl')

    return 0


if __name__ == "__main__":
    train_prior()
