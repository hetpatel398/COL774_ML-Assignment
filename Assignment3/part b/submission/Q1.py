import numpy as np

from time import time
import matplotlib.pyplot as plt
import math
from sklearn.neural_network import MLPClassifier

from MLPClassifier import MLP
from helper import load_data, plot_time, plot_train_test_acc

import sys, getopt

if __name__=='__main__':
    args = sys.argv[1:]
    try:
        opts, args = getopt.getopt(args, 'f:', ["data_folder=","part=", "help"])
    except getopt.GetoptError:
        print('Q1.py [-f/--data_folder] <data_parent_folder> --part <Part of Question1 [b/c/d/e/f]> [--help]')
        sys.exit(2)

    print('\n\n'+('*'*10)+'Loading Datasets'+('*'*10)+'\n\n')
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
        time_lst, train_acc_lst, test_acc_lst = [],[],[]
        for n_neurons in n_neurons_lst:
            print('Training MLP model with %d neurons in hidden layer:'%(n_neurons))
            ml=MLP(layers=[n_neurons], batch_size=100, activation_fn='sigmoid', max_epoch=3000,\
               lr=0.1, adaptive_lr=False, tol=1e-5, n_iter_no_change=50, initialization='he-uniform')
            t0=time()
            ml.fit(X_train, Y_train)
            time_taken=(time()-t0)
            test_acc=ml.score(X_test, Y_test)
            train_acc=ml.score(X_train, Y_train)
            time_lst.append(time_taken)
            train_acc_lst.append(train_acc)
            test_acc_lst.append(test_acc)
            print('Trained in %.2f minutes'%(time_taken/60))
            print('Test Accuracy : %.2f %c'%(test_acc*100,'%'))
            print('Train Accuracy : %.2f %c'%(train_acc*100,'%'))
            print('\n')
        plot_time(n_neurons_lst, time_lst)
        plot_train_test_acc(n_neurons_lst, train_acc_lst, test_acc_lst)
        
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
        time_lst, train_acc_lst, test_acc_lst = [],[],[]
        for n_neurons in n_neurons_lst:
            print('Training MLP model with %d neurons in hidden layer:'%(n_neurons))
            ml=MLP(layers=[n_neurons], batch_size=100, activation_fn='sigmoid', max_epoch=3000,\
               lr=0.5, adaptive_lr=True, tol=1e-5, n_iter_no_change=50, initialization='he-uniform')
            t0=time()
            ml.fit(X_train, Y_train)
            time_taken=(time()-t0)
            test_acc=ml.score(X_test, Y_test)
            train_acc=ml.score(X_train, Y_train)
            time_lst.append(time_taken)
            train_acc_lst.append(train_acc)
            test_acc_lst.append(test_acc)
            print('Trained in %.2f minutes'%(time_taken/60))
            print('Test Accuracy : %.2f %c'%(test_acc*100,'%'))
            print('Train Accuracy : %.2f %c'%(train_acc*100,'%'))
            print('\n')
        plot_time(n_neurons_lst, time_lst)
        plot_train_test_acc(n_neurons_lst, train_acc_lst, test_acc_lst)

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

        print('''
        Running with same parameters as above except activation function as relu in hidden layers
        ''')
        ml=MLP(layers=[100,100], batch_size=100, activation_fn='relu', max_epoch=3000,\
                lr=0.5, adaptive_lr=True, tol=1e-5, n_iter_no_change=50, initialization='he-uniform')
        t0=time()
        ml.fit(X_train, Y_train)
        print('Trained in %.2f minutes'%((time()-t0)/60))
        print('Test Accuracy : %.2f %c'%(ml.score(X_test, Y_test)*100,'%'))
        print('Train Accuracy : %.2f %c'%(ml.score(X_train, Y_train)*100,'%'))

        print('''
        Runnig both of above parts with learning rate as 0.1 constant
        ''')
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
        print('\n\n'+('*'*10)+'Running Part E'+('*'*10)+'\n\n')
        def score(clf, X, Y):
            proba = clf.predict_log_proba(X)
            Y_pred = proba.argmax(axis=1)
            Y_true = Y.argmax(axis=1)
            return np.sum(Y_true==Y_pred)/len(Y_true)
        
        print('Training MLPClassifier with same architecture in part D and learning_rate_init=0.5 and learning_rate=\'invscaling\'\n')
        
        t0=time()
        mlp = MLPClassifier(hidden_layer_sizes=(100,100), solver='sgd', learning_rate='invscaling', learning_rate_init=0.5,\
                            max_iter=2000, batch_size=100, random_state=0, momentum=0, tol=1e-4,\
                            n_iter_no_change=10, alpha=0).fit(X_train, Y_train)
        print('Trained in %.2f minutes'%((time()-t0)/60))
        print('Test accuracy %.2f %c'%(score(mlp, X_test, Y_test)*100, '%'))
        print('Train accuracy %.2f %c'%(score(mlp, X_train, Y_train)*100, '%'))
        print('Number of epochs till convergence %d'%(len(mlp.loss_curve_)))
        print('Output activation',mlp.out_activation_)
        print('Number of neurons in output layer',mlp.n_outputs_)
        print('Loss type :',mlp.loss)
        print('Loss Value :',mlp.loss_)
        
        print('\nRunning same thing as above with activation_fn=\'logistic\'\n')
        t0=time()
        mlp = MLPClassifier(hidden_layer_sizes=(100,100), solver='sgd', learning_rate='invscaling', learning_rate_init=0.5,\
                           max_iter=2000, batch_size=100, random_state=0, momentum=0, tol=1e-4,\
                           activation='logistic', n_iter_no_change=10, alpha=0).fit(X_train, Y_train)
        print('Trained in %.2f minutes'%((time()-t0)/60))
        print('Test accuracy %.2f %c'%(score(mlp, X_test, Y_test)*100, '%'))
        print('Train accuracy %.2f %c'%(score(mlp, X_train, Y_train)*100, '%'))
        print('Number of epochs till convergence %d'%(len(mlp.loss_curve_)))
        print('Output activation',mlp.out_activation_)
        print('Number of neurons in output layer',mlp.n_outputs_)
        print('Loss type :',mlp.loss)
        print('Loss Value :',mlp.loss_)
        
        print('\nRunning both part that I ran above with learning_rate=\'adaptive\' and learning_rate_init=0.5')
        print('\nWith ReLU Activation function')
        t0=time()
        mlp = MLPClassifier(hidden_layer_sizes=(100,100), solver='sgd', learning_rate='adaptive', learning_rate_init=0.5,\
                            max_iter=2000, batch_size=100, random_state=0, momentum=0, tol=1e-4,\
                            n_iter_no_change=10, alpha=0).fit(X_train, Y_train)
        print('Trained in %.2f minutes'%((time()-t0)/60))
        print('Test accuracy %.2f %c'%(score(mlp, X_test, Y_test)*100, '%'))
        print('Train accuracy %.2f %c'%(score(mlp, X_train, Y_train)*100, '%'))
        print('Number of epochs till convergence %d'%(len(mlp.loss_curve_)))
        print('Output activation',mlp.out_activation_)
        print('Number of neurons in output layer',mlp.n_outputs_)
        print('Loss type :',mlp.loss)
        print('Loss Value :',mlp.loss_)
        
        print('\nRunning same thing as above with activation_fn=\'logistic\'\n')
        t0=time()
        mlp = MLPClassifier(hidden_layer_sizes=(100,100), solver='sgd', learning_rate='adaptive', learning_rate_init=0.5,\
                           max_iter=2000, batch_size=100, random_state=0, momentum=0, tol=1e-4,\
                           activation='logistic', n_iter_no_change=10, alpha=0).fit(X_train, Y_train)
        print('Trained in %.2f minutes'%((time()-t0)/60))
        print('Test accuracy %.2f %c'%(score(mlp, X_test, Y_test)*100, '%'))
        print('Train accuracy %.2f %c'%(score(mlp, X_train, Y_train)*100, '%'))
        print('Number of epochs till convergence %d'%(len(mlp.loss_curve_)))
        print('Output activation',mlp.out_activation_)
        print('Number of neurons in output layer',mlp.n_outputs_)
        print('Loss type :',mlp.loss)
        print('Loss Value :',mlp.loss_)
        
        

    if 'f' in parts:
        print('\n\n'+('*'*10)+'Running Part F'+('*'*10)+'\n\n')
        print('In this part I will be running part B,C and D with binary cross-entropy loss. Everything else remains same')
        print('\nRunning Part B with binary cross-entropy loss(n_iter_no_change=10)\n')
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

        print('\nRunning Part C with binary cross-entropy loss(n_iter_no_change=10)\n')
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

        print('\nRunning Part D with binary cross-entropy loss(n_iter_no_change=10, tol=1e-4)\n')
        
        print('\nTwo hidden layers with 100 neurons each with relu activation function and adaptive learning rate with lr=0.5\n')
        ml=MLP(layers=[100,100], batch_size=100, activation_fn='relu', max_epoch=3000,\
               lr=0.5, adaptive_lr=True, tol=1e-4, n_iter_no_change=10, initialization='he-uniform', loss_fn='cross-entropy')
        t0=time()
        ml.fit(X_train, Y_train)
        print('Trained in %.2f minutes'%((time()-t0)/60))
        print('Test Accuracy : %.2f %c'%(ml.score(X_test, Y_test)*100,'%'))
        print('Train Accuracy : %.2f %c'%(ml.score(X_train, Y_train)*100,'%'))
        
        print('\nTwo hidden layers with 100 neurons each with sigmoid activation function and adaptive learning rate with lr=0.5\n')
        ml=MLP(layers=[100,100], batch_size=100, activation_fn='sigmoid', max_epoch=3000,\
               lr=0.5, adaptive_lr=True, tol=1e-4, n_iter_no_change=10, initialization='he-uniform', loss_fn='cross-entropy')
        t0=time()
        ml.fit(X_train, Y_train)
        print('Trained in %.2f minutes'%((time()-t0)/60))
        print('Test Accuracy : %.2f %c'%(ml.score(X_test, Y_test)*100,'%'))
        print('Train Accuracy : %.2f %c'%(ml.score(X_train, Y_train)*100,'%'))

        print('\nTwo hidden layers with 100 neurons each with relu activation function and 0.1 learning rate\n')
        ml=MLP(layers=[100,100], batch_size=100, activation_fn='relu', max_epoch=3000,\
               lr=0.1, adaptive_lr=False, tol=1e-4, n_iter_no_change=10, initialization='he-uniform', loss_fn='cross-entropy')
        t0=time()
        ml.fit(X_train, Y_train)
        print('Trained in %.2f minutes'%((time()-t0)/60))
        print('Test Accuracy : %.2f %c'%(ml.score(X_test, Y_test)*100,'%'))
        print('Train Accuracy : %.2f %c'%(ml.score(X_train, Y_train)*100,'%'))
        
        print('\nTwo hidden layers with 100 neurons each with sigmoid activation function and 0.1 learning rate\n')
        ml=MLP(layers=[100,100], batch_size=100, activation_fn='sigmoid', max_epoch=3000,\
               lr=0.1, adaptive_lr=False, tol=1e-4, n_iter_no_change=10, initialization='he-uniform', loss_fn='cross-entropy')
        t0=time()
        ml.fit(X_train, Y_train)
        print('Trained in %.2f minutes'%((time()-t0)/60))
        print('Test Accuracy : %.2f %c'%(ml.score(X_test, Y_test)*100,'%'))
        print('Train Accuracy : %.2f %c'%(ml.score(X_train, Y_train)*100,'%'))
