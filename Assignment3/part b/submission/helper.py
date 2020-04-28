import numpy as np
import matplotlib.pyplot as plt

def load_data(folder):
    train_data = np.genfromtxt(folder+'/train.csv', delimiter=',')
    test_data = np.genfromtxt(folder+'/test.csv', delimiter=',')
    X_train = train_data[:,:-1]/255
    Y_train_orig = train_data[:,-1]
    X_test = test_data[:,:-1]/255
    Y_test_orig = test_data[:,-1]

    Y_train = one_hot_encode(Y_train_orig)
    Y_test = one_hot_encode(Y_test_orig)

    return X_train, Y_train, X_test, Y_test

def one_hot_encode(Y):
    classes_ = np.unique(Y)
    n_classes = classes_.shape[0]
    Y_oh = np.zeros((Y.shape[0], n_classes), dtype=np.bool)
    i=0
    for y in Y:
        Y_oh[int(i)][int(y)]=1
        i+=1
    return Y_oh

def plot_time(X,Y):
    plt.figure(figsize=(12*0.75, 9*0.75))
    plt.plot([1,2,3,4,5],Y,linestyle='-', marker='P', color='b', label='Time to train(in seconds)')
    plt.xticks([1,2,3,4,5],X)
    for i_x, i_y in zip([1,2,3,4,5], Y):
        plt.annotate('(%d, %.1f)'%(i_x, i_y), (i_x+0.05, i_y-2))
    plt.xlabel('Number of neurons in hidden layer')
    plt.ylabel('Time taken to train')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('Number of neurons in hidden layer v/s Time taken to trin MLP')
    plt.xlim([0.5,5.5])
    plt.show()

def plot_train_test_acc(X,train,test):
    plt.figure(figsize=(12*0.75, 9*0.75))
    plt.plot([1,2,3,4,5],train,linestyle='-', marker='3', color='r', label='Train accuracy')
    plt.plot([1,2,3,4,5],test,linestyle='--', marker='o', color='g', label='Test accuracy')
    plt.xticks([1,2,3,4,5],X)
    for i_x, i_train, i_test in zip([1,2,3,4,5], train, test):
        plt.annotate('(%d, %.2f)'%(i_x, i_test), (i_x+0.05, i_test-0.025))
        plt.annotate('(%d, %.2f)'%(i_x, i_train), (i_x-0.35, i_train+0.025))
    plt.xlabel('Number of neurons in hidden layer')
    plt.ylabel('Accuracy')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('Number of neurons in hidden layer v/s Train and test accuracy')
    plt.xlim([0.5,5.5])
    plt.ylim([-0.1,1.1])
    plt.show()