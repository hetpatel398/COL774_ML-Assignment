import numpy as np
from xclib.data import data_utils

from sklearn.ensemble import RandomForestClassifier

import itertools
from tqdm import tqdm
from time import time

from joblib import Parallel, delayed
import multiprocessing as mp
import matplotlib.pyplot as plt
import math

from DecisionTreeClassifier import DecisionTreeClassifier, DTNode
from utils import load_data, plot_accuracy, plot_accuracies_wo_train, plot_accuracies
from RandomForest import GridSearchUsingOOB, analyseChangeInHyperParameter

import sys, getopt

if __name__=='__main__':
    args = sys.argv[1:]
    main(args)

def main(args):
    try:
        opts, args = getopt.getopt(args, 'f:', ["data_folder=","part=", "record_accuracies", "help"])
    except getopt.GetoptError:
        print('Q1.py [-f/--data_folder] <data_parent_folder> --part <Part of Question1 [a/b/c/d]> [--record_accuracies] [--help]')
        sys.exit(2)
    opts_dict = dict(opts)
    if opts_dict.get('-f')==None and opts_dict.get('--data_folder')==None:
        (X_train, Y_train, X_val, Y_val, X_test, Y_test) = load_data('./data')
    elif opts_dict.get('-f')!=None:
        (X_train, Y_train, X_val, Y_val, X_test, Y_test) = load_data(opts_dict.get('-f'))
    elif opts_dict.get('--data_folder')!=None:
        (X_train, Y_train, X_val, Y_val, X_test, Y_test) = load_data(opts_dict.get('--data_folder'))

    print('Shape of X_train',X_train.shape)
    print('Shape of X_test',X_test.shape)
    print('Shape of X_val',X_val.shape)

    part = opts_dict.get('--part')
    parts = list(part) if part is not None else list('abcd')
    record_accuracies = True if opts_dict.get('--record_accuracies') is not None else False

    if 'a' in parts:
        print('\n\n'+('*'*10)+'Running Part A'+('*'*10)+'\n\n')
        dt=DecisionTreeClassifier(get_train_data=record_accuracies, get_prune_data=record_accuracies)
        t0=time()
        dt.fit(X_train, Y_train, X_test, Y_test, X_val, Y_val)
        print('Generated tree in %.2f minutes'%((time()-t0)/60))
        train_acc = dt.score(X_train, Y_train)
        val_acc = dt.score(X_val, Y_val)
        test_acc = dt.score(X_test, Y_test)
        print('Train Accuracy :',train_acc*100)
        print('Test Accuracy :',test_acc*100)
        print('Validation Accuracy :',val_acc*100)
        print('\n\n'+str(dt))
        if record_accuracies:
            dt.plot_training_data()

    if 'b' in parts:
        if 'a' not in parts:
            dt=DecisionTreeClassifier(get_train_data=False, get_prune_data=record_accuracies)
            t0=time()
            dt.fit(X_train, Y_train, X_test, Y_test, X_val, Y_val)
            print('Generated tree in %.2f minutes'%((time()-t0)/60))
            train_acc = dt.score(X_train, Y_train)
            val_acc = dt.score(X_val, Y_val)
            test_acc = dt.score(X_test, Y_test)
            print('Train Accuracy :',train_acc*100)
            print('Test Accuracy :',test_acc*100)
            print('Validation Accuracy :',val_acc*100)
            print('\n\n'+str(dt))

        print('\n\n'+('*'*10)+'Running Part B'+('*'*10)+'\n\n')

        t0=time()
        dt.prune(X_val, Y_val, X_train, Y_train, X_test, Y_test)
        print('Pruned tree in %.2f seconds'%(time()-t0))

        train_acc = dt.score(X_train, Y_train)
        val_acc = dt.score(X_val, Y_val)
        test_acc = dt.score(X_test, Y_test)

        print('\nAfter Pruning : ')
        print('Train Accuracy :',train_acc*100)
        print('Test Accuracy :',test_acc*100)
        print('Validation Accuracy :',val_acc*100)
        print('\n\n'+str(dt))
        if record_accuracies:
            dt.plot_pruning_data()

    if 'c' in parts:
        print('\n\n'+('*'*10)+'Running Part C'+('*'*10)+'\n\n')

        grid = {'n_estimators':list(range(50, 451,100)),\
        'max_features':[x/10 for x in (range(1, 10, 2))],\
        'min_samples_split':list(range(2,11,2))
        }
        optimal_clf = GridSearchUsingOOB(grid, X_train, Y_train, n_jobs=-2)
        print('\n\noptimal Classifier :\n\n ',optimal_clf)

        optimal_n_estimators = optimal_clf.n_estimators
        optimal_max_features = optimal_clf.max_features
        optimal_min_samples_split = optimal_clf.min_samples_split

        print('These are the hyperparametes for the optimal model \n n_estimators :',optimal_n_estimators)
        print(' max_features :',optimal_max_features)
        print(' min_samples_split :',optimal_min_samples_split)

        print("Accuracies using Optimal classifier:\n Train accuracy :",optimal_clf.score(X_train, Y_train))
        print(" Test accuracy :",optimal_clf.score(X_test, Y_test))
        print(" Validation accuracy :",optimal_clf.score(X_val, Y_val))
        print(" OOB accuracy :",optimal_clf.oob_score_)

    if 'd' in parts:
        print('\n\n'+('*'*10)+'Running Part D'+('*'*10)+'\n\n')

        if 'c' not in parts:
            grid = {'n_estimators':list(range(50, 451,100)),\
                'max_features':[x/10 for x in (range(1, 10, 2))],\
                'min_samples_split':list(range(2,11,2))
                }
            optimal_clf = GridSearchUsingOOB(grid, X_train, Y_train, n_jobs=-2, repeat=1)

            optimal_n_estimators = optimal_clf.n_estimators
            optimal_max_features = optimal_clf.max_features
            optimal_min_samples_split = optimal_clf.min_samples_split

        # Changing n_estimators
        n_est_lst = list(range(10,50,10))
        n_est_lst.extend(list(range(50, 1001, 50)))
        analyseChangeInHyperParameter(n_est_lst,[optimal_max_features], [optimal_min_samples_split], X_train, Y_train, X_val, Y_val, X_test, Y_test, smooth_garph=True)

        # Changing n_estimators
        max_feat_lst = [x/50 for x in (range(1, 50, 2))]
        analyseChangeInHyperParameter([optimal_n_estimators],max_feat_lst,[optimal_min_samples_split], X_train, Y_train, X_val, Y_val, X_test, Y_test, smooth_garph=True)

        # Changing n_estimators
        min_sam_lst = list(range(2,41,2))
        analyseChangeInHyperParameter([optimal_n_estimators],[optimal_max_features],min_sam_lst, X_train, Y_train, X_val, Y_val, X_test, Y_test,smooth_garph=True)
