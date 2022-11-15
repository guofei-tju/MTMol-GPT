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


def train_prior():
    prop1 = 'DRD2'
    prop2 = 'HTR1A'

    train_path = f'./data/EXCAPE-DB/{prop1}_all_train.csv'
    test_path = f'./data/EXCAPE-DB/{prop1}_all_test.csv'

    train_prop1_data = pd.read_csv(train_path, index_col=False)
    test_prop1_data = pd.read_csv(test_path, index_col=False)

    X_train_prop1_smiles = train_prop1_data['SMILES'].values
    X_test_prop1_smiles = test_prop1_data['SMILES'].values
    y_prop1_train = train_prop1_data['Targets'].values.astype(int)
    y_prop1_test = test_prop1_data['Targets'].values.astype(int)

    X_prop1_train = [Chem.MolFromSmiles(s) for s in X_train_prop1_smiles]
    X_prop1_train = [x for x in X_prop1_train if x is not None]
    X_prop1_train = np.array([AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048).ToList() for x in X_prop1_train])

    X_prop1_test = [Chem.MolFromSmiles(s) for s in X_test_prop1_smiles]
    X_prop1_test = [x for x in X_prop1_test if x is not None]
    X_prop1_test = np.array([AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048).ToList() for x in X_prop1_test])

    # define the classifier
    prop1_clf = joblib.load(f'./results/model/model_clf/best_{prop1.lower()}_svm.m')
    # prop2_clf = joblib.load(f'../results/model/model_clf/{prop2}_svm_rbf_p.pkl')

    print(f" {prop1} training accuracy: " + str(accuracy_score(y_prop1_train, prop1_clf.predict(X_prop1_train))))
    print(f" {prop1} training auc: " + str(metrics.roc_auc_score(y_prop1_train, prop1_clf.predict_log_proba(X_prop1_train)[:, 1])))

    print(f" {prop1} testing accuracy: " + str(accuracy_score(y_prop1_test, prop1_clf.predict(X_prop1_test))))
    print(f" {prop1} testing auc: " + str(metrics.roc_auc_score(y_prop1_test, prop1_clf.predict_proba(X_prop1_test)[:, 1])))

    return 0


if __name__ == "__main__":
    train_prior()
