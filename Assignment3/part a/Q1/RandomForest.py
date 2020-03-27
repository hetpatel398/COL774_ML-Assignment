import matplotlib.gridspec as gridspec
import numpy as np
import math
from time import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import itertools
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from scipy.interpolate import make_interp_spline, BSpline

def GridSearchUsingOOB(param_grid, X_train, Y_train, n_jobs=1, repeat=3):
    """
        This method is the implementation of grid search over parameters n_estimators, max_features and min_samples_split
        using oob_score as scoring criteria
        Arguments:
            param_grid : [Dictionary]
                Dictionary containing keys 'n_estimators', 'max_features' and 'min_samples_split' and
                their corrosponding values are the value of hyperparameters on which you want to perform GridSearch
            repeat : [int]
                Parameter representing how many times you want to run the experiment to reduce stochastic behaviour of bootstrapped samples
    """
    params = list(itertools.product(*(param_grid[key] for key in param_grid.keys()))) #generating list of tuples
    optimal_clf, optimal_oob_score = None, 0 #Initializing values
    for (n_est, max_feat, min_sam_split) in tqdm(params):
        oob_score=0
        for i in range(repeat):
            clf = RandomForestClassifier(n_estimators=n_est, max_features=max_feat, min_samples_split=min_sam_split, oob_score=True, n_jobs=n_jobs).fit(X_train, Y_train)
            oob_score += clf.oob_score_
        if (oob_score/repeat)>optimal_oob_score:
            optimal_clf = clf
            optimal_oob_score = oob_score/repeat
    return optimal_clf

def analyseChangeInHyperParameter(n_est_lst, max_features_lst, min_samples_split_lst, X_train, Y_train, X_val, Y_val, X_test, Y_test, smooth_garph=True, repeat_exp=3):
    """
        This method will plot four different plots to analyse sensitivity of particular hyperparameter while keeping other two parameters fixed
        1. Hyperparameter v/s Train Accuracy
        2. Hyperparameter v/s Test Accuracy
        3. Hyperparameter v/s Validation Accuracy
        4. Hyperparameter v/s Train, Test and Validation Accuracy

        Parameters:
            n_est_lst, max_features_lst, min_samples_split_lst : [list]
                You pass two lists with just one value that you want to fix that corrosponding hyperparameter to
                and remaining one list with values that you want to test
            smooth_garph : [boolean]
                Flag that will represent if you want smooth graph or not(Smoothened using interpolation)
    """
    val_score_lst, train_score_lst, test_score_lst, oob_score_lst = [], [], [], []
    params = list(itertools.product(n_est_lst, max_features_lst, min_samples_split_lst))
    for n_est, max_feat, min_sam_split in tqdm(params):
        train_score, test_score, val_score, oob_score = 0,0,0,0
        for i in range(repeat_exp):
            clf = RandomForestClassifier(n_estimators=n_est, max_features=max_feat, min_samples_split=min_sam_split\
                                         , oob_score=False, n_jobs=-2).fit(X_train, Y_train)
            train_score+=(clf.score(X_train, Y_train)*100)
            test_score+=(clf.score(X_test, Y_test)*100)
            val_score+=(clf.score(X_val, Y_val)*100)
        train_score_lst.append(train_score/3)
        test_score_lst.append(test_score/3)
        val_score_lst.append(val_score/3)
    if len(n_est_lst)!=1:
        xlabel = 'n_estimators'
        title = 'n_estimators v/s Accuracy'
        X=n_est_lst
    elif len(max_features_lst)!=1:
        xlabel = 'max_features'
        title = 'max_features v/s Accuracy'
        X=max_features_lst
    else:
        xlabel = 'min_samples_split'
        title = 'min_samples_split v/s Accuracy'
        X=min_samples_split_lst

    Y_train_smoothed = make_interp_spline(X, train_score_lst, k=3)(xnew)
    Y_test_smoothed = make_interp_spline(X, test_score_lst, k=3)(xnew)
    Y_val_smoothed = make_interp_spline(X, val_score_lst, k=3)(xnew)
    if not smooth_garph:
        xnew = X
        Y_train_smoothed=train_score_lst
        Y_test_smoothed=test_score_lst
        Y_val_smoothed=val_score_lst
    gs = gridspec.GridSpec(2, 2)

    fig = plt.figure(figsize=(30,20))
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(xnew, Y_train_smoothed, label='Train Score')
    ax1.legend()
    ax1.set_title(xlabel+' v/s Train Accuracy')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Train Accuracy')

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(xnew, Y_test_smoothed, label='Tes Score')
    ax1.legend()
    ax1.set_title(xlabel+' v/s Test Accuracy')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Test Accuracy')

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(xnew, Y_val_smoothed, label='Validation Score')
    ax1.legend()
    ax1.set_title(xlabel+' v/s Validation Accuracy')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Validation Accuracy')

    ax1 = fig.add_subplot(gs[1, 1])
    ax1.plot(xnew, Y_test_smoothed, label='Test Score')
    ax1.plot(xnew, Y_val_smoothed, label='Validation Score')
    ax1.plot(xnew, Y_train_smoothed, label='Train Score')
    ax1.legend()
    ax1.set_title(xlabel+' v/s Test, Validation and Train Accuracy')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Accuracy')

    plt.show()
