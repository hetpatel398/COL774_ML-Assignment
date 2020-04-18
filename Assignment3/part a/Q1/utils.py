import numpy as np
from xclib.data import data_utils

def load_data(folder, remove_zero_columns=True):
    """
        Method that will load train test and validation data fromm the specified folder path
        This method also removes 219 columns that have only 0 values
        So returned datasets will have subset of original features
    """
    X_train = data_utils.read_sparse_file(folder+'/train_x.txt', n_features=482).toarray()
    X_test = data_utils.read_sparse_file(folder+'/test_x.txt', n_features=482).toarray()
    X_val = data_utils.read_sparse_file(folder+'/valid_x.txt', n_features=482).toarray()

    Y_train = np.array(open(folder+'/train_y.txt','r').read().strip().split('\n'), dtype=int)
    Y_test = np.array(open(folder+'/test_y.txt','r').read().strip().split('\n'), dtype=int)
    Y_val = np.array(open(folder+'/valid_y.txt','r').read().strip().split('\n'), dtype=int)

    if remove_zero_columns:
        ind_lst = []
        for i in range(X_train.shape[1]):
            if (X_train[:,i].min() == 0) and (X_train[:,i].max() == 0):
                ind_lst.append(i)

        #Removing columns with zero values
        X_train = np.delete(X_train, ind_lst, axis=1)
        X_test = np.delete(X_test, ind_lst, axis=1)
        X_val = np.delete(X_val, ind_lst, axis=1)

    return (X_train, Y_train, X_val, Y_val, X_test, Y_test)
