import numpy as np

from time import time
import matplotlib.pyplot as plt
import math

from MLPClassifier import MLP, load_data

import sys, getopt

def main(args):
    try:
        opts, args = getopt.getopt(args, 'f:', ["data_folder=","part=", "help"])
    except getopt.GetoptError:
        print('Q1.py [-f/--data_folder] <data_parent_folder> --part <Part of Question1 [b/c/d/e/f]> [--help]')
        sys.exit(2)

    opts_dict = dict(opts)
    if opts_dict.get('-f')==None and opts_dict.get('--data_folder')==None:
        X_train, Y_train, X_test, Y_test = load_data('./data')
    elif opts_dict.get('-f')!=None:
        X_train, Y_train, X_test, Y_test = load_data(opts_dict.get('-f'))
    elif opts_dict.get('--data_folder')!=None:
        X_train, Y_train, X_test, Y_test = load_data(opts_dict.get('--data_folder'))

    print('Shape of X_train',X_train.shape)
    print('Shape of X_test',X_test.shape)

    part = opts_dict.get('--part')
    parts = list(part) if part is not None else list('bcdef')

    if 'b' in parts:
        print('\n\n'+('*'*10)+'Running Part B'+('*'*10)+'\n\n')
        n_neurons_lst = [1,5,10,50,100]
        print('''Running MLP with following parameters:

        input_size=784
        batch_size=100
        output_size=26
        lr=0.1
        adaptive_learning=False
        activation_fn='sigmoid'
        max_epoch=3000
        tol=1e-5
        n_iter_no_change=50
        initialization='he-uniform'
        early_stopping=False
        loss_fn='squared-loss'

        ''')
        for n_neurons in n_neurons_lst:
            print('Training MLP model with %d neurons in hidden layer:'%(n_neurons))
            ml=MLP(layers=[n_neurons], batch_size=100, activation_fn='sigmoid', max_epoch=3000,\
               lr=0.1, adaptive_lr=False, tol=1e-5, n_iter_no_change=50, initialization='he-uniform')
            t0=time()
            ml.fit(X_train, Y_train)
            print('Trained in %.2f minutes'%((time()-t0)/60))
            print('Test Accuracy : %.2f %c'%(ml.score(X_test, Y_test)*100,'%'))
            print('Train Accuracy : %.2f %c'%(ml.score(X_train, Y_train)*100,'%'))
            print('\n')

    if 'c' in parts:
        print('\n\n'+('*'*10)+'Running Part C'+('*'*10)+'\n\n')
        print('''Running MLP with following parameters:

        input_size=784
        batch_size=100
        output_size=26
        lr=0.5
        adaptive_learning=True
        activation_fn='sigmoid'
        max_epoch=3000
        tol=1e-5
        n_iter_no_change=50
        initialization='he-uniform'
        early_stopping=False
        loss_fn='squared-loss'

        ''')
        n_neurons_lst = [1,5,10,50,100]
        for n_neurons in n_neurons_lst:
            print('Training MLP model with %d neurons in hidden layer:'%(n_neurons))
            ml=MLP(layers=[n_neurons], batch_size=100, activation_fn='sigmoid', max_epoch=3000,\
               lr=0.5, adaptive_lr=True, tol=1e-5, n_iter_no_change=50, initialization='he-uniform')
            t0=time()
            ml.fit(X_train, Y_train)
            print('Trained in %.2f minutes'%((time()-t0)/60))
            print('Test Accuracy : %.2f %c'%(ml.score(X_test, Y_test)*100,'%'))
            print('Train Accuracy : %.2f %c'%(ml.score(X_train, Y_train)*100,'%'))
            print('\n')

    if 'd' in parts:
        print('\n\n'+('*'*10)+'Running Part D'+('*'*10)+'\n\n')
        print('''
        Running MLP with following parameters:

        input_size=784
        n_layers=[100,100]
        batch_size=100
        output_size=26
        lr=0.5
        adaptive_learning=True
        activation_fn='sigmoid'
        max_epoch=3000
        tol=1e-5
        n_iter_no_change=50
        initialization='he-uniform'
        early_stopping=False
        loss_fn='squared-loss'
        ''')
        ml=MLP(layers=[100,100], batch_size=100, activation_fn='sigmoid', max_epoch=3000,\
                lr=0.5, adaptive_lr=True, tol=1e-5, n_iter_no_change=50, initialization='he-uniform')
        t0=time()
        ml.fit(X_train, Y_train)
        print('Trained in %.2f minutes'%((time()-t0)/60))
        print('Test Accuracy : %.2f %c'%(ml.score(X_test, Y_test)*100,'%'))
        print('Train Accuracy : %.2f %c'%(ml.score(X_train, Y_train)*100,'%'))

        print('\nRunning with same parameters as above except activation function as relu in hidden layers\n')
        ml=MLP(layers=[100,100], batch_size=100, activation_fn='relu', max_epoch=3000,\
                lr=0.5, adaptive_lr=True, tol=1e-5, n_iter_no_change=50, initialization='he-uniform')
        t0=time()
        ml.fit(X_train, Y_train)
        print('Trained in %.2f minutes'%((time()-t0)/60))
        print('Test Accuracy : %.2f %c'%(ml.score(X_test, Y_test)*100,'%'))
        print('Train Accuracy : %.2f %c'%(ml.score(X_train, Y_train)*100,'%'))

        print('\nRunnig both of above parts with learning rate as 0.1 constant\n')
        print('Running with activation function ReLU')
        ml=MLP(layers=[100,100], batch_size=100, activation_fn='relu', max_epoch=3000,\
               lr=0.1, adaptive_lr=False, tol=1e-5, n_iter_no_change=50, initialization='he-uniform')
        t0=time()
        ml.fit(X_train, Y_train)
        print('Trained in %.2f minutes'%((time()-t0)/60))
        print('Test Accuracy : %.2f %c'%(ml.score(X_test, Y_test)*100,'%'))
        print('Train Accuracy : %.2f %c'%(ml.score(X_train, Y_train)*100,'%'))
        print('\nRunning with activation function sigmoid')
        ml=MLP(layers=[100,100], batch_size=100, activation_fn='sigmoid', max_epoch=3000,\
               lr=0.1, adaptive_lr=False, tol=1e-5, n_iter_no_change=50, initialization='he-uniform')
        t0=time()
        ml.fit(X_train, Y_train)
        print('Trained in %.2f minutes'%((time()-t0)/60))
        print('Test Accuracy : %.2f %c'%(ml.score(X_test, Y_test)*100,'%'))
        print('Train Accuracy : %.2f %c'%(ml.score(X_train, Y_train)*100,'%'))

    if 'e' in parts:
        print('\n\n'+('*'*10)+'Running Part F'+('*'*10)+'\n\n')

    if 'f' in parts:
        print('\n\n'+('*'*10)+'Running Part F'+('*'*10)+'\n\n')
        print('In this part I will be running part B,C and D with binary cross-entropy loss. Everything else remains same')
        print('\nRunning Part B with binary cross-entropy loss(n_iter_no_change=10)')
        n_neurons_lst = [1,5,10,50,100]
        for n_neurons in n_neurons_lst:
            print('Training MLP model with %d neurons in hidden layer:'%(n_neurons))
            ml=MLP(layers=[n_neurons], batch_size=100, activation_fn='sigmoid', max_epoch=3000,\
               lr=0.1, adaptive_lr=False, tol=1e-5, n_iter_no_change=10, initialization='he-uniform', loss_fn='cross-entropy')
            t0=time()
            ml.fit(X_train, Y_train)
            print('Trained in %.2f minutes'%((time()-t0)/60))
            print('Test Accuracy : %.2f %c'%(ml.score(X_test, Y_test)*100,'%'))
            print('Train Accuracy : %.2f %c'%(ml.score(X_train, Y_train)*100,'%'))
            print('\n')

        print('\nRunning Part C with binary cross-entropy loss(n_iter_no_change=10)')
        n_neurons_lst = [1,5,10,50,100]
        for n_neurons in n_neurons_lst:
            print('Training MLP model with %d neurons in hidden layer:'%(n_neurons))
            ml=MLP(layers=[n_neurons], batch_size=100, activation_fn='sigmoid', max_epoch=3000,\
               lr=0.5, adaptive_lr=True, tol=1e-5, n_iter_no_change=10, initialization='he-uniform', loss_fn='cross-entropy')
            t0=time()
            ml.fit(X_train, Y_train)
            print('Trained in %.2f minutes'%((time()-t0)/60))
            print('Test Accuracy : %.2f %c'%(ml.score(X_test, Y_test)*100,'%'))
            print('Train Accuracy : %.2f %c'%(ml.score(X_train, Y_train)*100,'%'))
            print('\n')

        print('\nRunning Part C with binary cross-entropy loss(n_iter_no_change=10, tol=1e-4)')
        print('Two hidden layers with 100 neurons each with relu activation function and 0.1 learning rate')
        ml=MLP(layers=[100,100], batch_size=100, activation_fn='relu', max_epoch=3000,\
               lr=0.1, adaptive_lr=False, tol=1e-4, n_iter_no_change=10, initialization='he-uniform', loss_fn='cross-entropy')
        t0=time()
        ml.fit(X_train, Y_train)
        print('Trained in %.2f minutes'%((time()-t0)/60))
        print('Test Accuracy : %.2f %c'%(ml.score(X_test, Y_test)*100,'%'))
        print('Train Accuracy : %.2f %c'%(ml.score(X_train, Y_train)*100,'%'))
        print('\nTwo hidden layers with 100 neurons each with relu activation function and 0.1 learning rate')
        ml=MLP(layers=[100,100], batch_size=100, activation_fn='sigmoid', max_epoch=3000,\
               lr=0.1, adaptive_lr=False, tol=1e-4, n_iter_no_change=10, initialization='he-uniform', loss_fn='cross-entropy')
        t0=time()
        ml.fit(X_train, Y_train)
        print('Trained in %.2f minutes'%((time()-t0)/60))
        print('Test Accuracy : %.2f %c'%(ml.score(X_test, Y_test)*100,'%'))
        print('Train Accuracy : %.2f %c'%(ml.score(X_train, Y_train)*100,'%'))

if __name__=='__main__':
    args = sys.argv[1:]
    main(args)
