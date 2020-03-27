import numpy as np
import math
from time import time
import matplotlib.pyplot as plt

class DTNode:
    '''
        Class representing node of
    '''
    def __init__(self, splitFeat=None, splitVal=None, majorityClass=None, parent=None, isLeft=True, X=None, Y=None, isLeaf=False):
        self.splitFeat = splitFeat
        self.splitVal = splitVal
        self.majorityClass = majorityClass
        self.isLeaf = False
        self.left = None
        self.right = None
        self.X, self.Y = X, Y
        self.nodes_subtree = None

    def is_leaf(self):
        return self.splitFeat is None or self.isLeaf

    def __str__(self):
        return 'SplitFeat : '+str(self.splitFeat)+'\nSplitVal : '+str(self.splitVal)+'\nMajorityClass : '+str(self.majorityClass)+'\n'

    def __repr__(self):
        return 'SplitFeat : '+str(self.splitFeat)+'\nSplitVal : '+str(self.splitVal)+'\nMajorityClass : '+str(self.majorityClass)+'\n'

class DecisionTreeClassifier:
    '''
        Implementation of decision tree classifier

        Parameters:
        *******************

        get_train_data(Default:False) : [boolean]
            If true then train, test and Validation accuracy recorded
            after adding record_data_frequency number of nodes

        record_data_frequency(Default:100) : [int]
            After how many nodes to record accuracy data

        get_prune_data(Default:False) : [boolean]
            If true then train, test and Validation accuracy recorded
            during pruning as we prune the nodeds from the tree

        Attributes:
        *******************

        root : [DTNode]
            Root of the decision Tree
        train_data : [Dictionary]
            Dictionary containing accuracies over Train, Test and Validation Datasets
            recorded as tree grows (You need to set get_train_data=True to get this)
        prune_data : [Dictionary]
            Dictionary containing accuracies over Train, Test and Validation Datasets
            recorded as we prune tree (You need to set get_prune_data=True to get this)

        Methods:
        *******************

        fit(self, X, Y[, X_test, Y_test, X_val, Y_val]) : [None]
            This method builds decision tree iteratively using
            X and Y as training Dataset [and records accuracies].
            (If you have set get_train_data=True then pass X_test, Y_test and X_val, Y_val)

        prune(self, X_val, Y_val, [X_train=None, Y_train=None, X_test=None, Y_test=None]) : [None]
            This method post punes the tree that was trained during fit method using
            methodology called Reduced error pruning which uses different pruning/validation
            dataset to post prune the tree to avoid overfitting on fully grown tree

        predict(self, X) : [list]
            Returns list of predicted class labels for datapoints in X

        score(self, X, Y_true) : [float]
            Returns accuracy over dataset X and ground truth Y_true

        get_number_of_nodes(self) : [tuple containing (total_nodes, total_leaves)]
            Returns tuple containg two elements first one containing total number of nodes
            and second one containing total number of leaves in the tree

        plot_training_data(self) : [None]
            Plots number of nodes v/s accuracies if get_train_data=True

        plot_pruning_data(self) : [None]
            Plots number of nodes v/s accuracies if get_prune_data=True
    '''
    def __init__(self, get_train_data=False, record_data_frequency=100, get_prune_data=False):
        self.root = DTNode()
        # self.cnt = 0
        self.leafNodes = 0
        self.get_train_data = get_train_data
        self.record_data_frequency = record_data_frequency
        self.get_prune_data=get_prune_data
        if self.get_train_data:
            self.train_data = {'train_acc':[], 'test_acc':[], 'val_acc':[]}
        if self.get_prune_data:
            self.prune_data = {'train_acc':[], 'test_acc':[], 'val_acc':[], 'nodes':[]}

    def __str__(self):
        return 'DecisionTreeClassifier\n\nParameters : \n\tget_train_data='+str(self.get_train_data)+'\n\tget_prune_data='+str(self.get_prune_data)+'\n\trecord_data_frequency='+str(self.record_data_frequency)+\
                '\n\nAttributes : \n\tTotal Number of nodes='+str(self.get_number_of_nodes()[0])+'\n\tNumber of leafe nodes='+str(self.get_number_of_nodes()[1])

    def __repr__(self):
        return 'DecisionTreeClassifier\n\nParameters : \n\tget_train_data='+str(self.get_train_data)+'\n\trecord_data_frequency='+str(self.record_data_frequency)+\
                '\n\nAttributes : \n\tTotal Number of nodes='+str(self.get_number_of_nodes()[0])+'\n\tNumber of leafe nodes='+str(self.get_number_of_nodes()[1])

    def fit(self, X, Y, X_test=None, Y_test=None, X_val=None, Y_val=None):
        '''
            Builds Tree using viven Train Dataset
        '''
        if self.get_train_data and (X_test is None or Y_test is None or X_val is None or Y_val is None):
            raise Exception('Please pass test and validation data to get train_data or pass get_train_data=False')

        queue = []
        X_train = X.copy()
        Y_train = Y.copy()
        self.root = DTNode(X=X, Y=Y)
        queue.append(self.root)
        cnt=0
        while queue!=[]:
            cnt+=1

            node = queue.pop(0)
            X_, Y_ = node.X, node.Y
            n_samples_, n_feat_ = X_.shape
            nY1_ = (Y_==1).sum()
            nY0_ = n_samples_ - nY1_
            majClass = 1 if nY1_>nY0_ else 0
            node.majorityClass=majClass
            if (nY1_ == 0) or (nY0_==0):
                self.leafNodes+=1
                node.isLeaf=True
                if self.get_train_data and (cnt-1)%self.record_data_frequency==0:
                    self._record_accuracies(X_train, Y_train, X_test, Y_test, X_val, Y_val)
                continue
            else:
                bestSplit_, median_, maxIG_ = self._getBestSplit(X_, Y_)
                if(bestSplit_==-np.inf):# or maxIG_==0:
                    self.leafNodes+=1
                    node.isLeaf=True
                    if self.get_train_data and (cnt-1)%self.record_data_frequency==0:
                        self._record_accuracies(X_train, Y_train, X_test, Y_test, X_val, Y_val)
                    continue
                node.splitFeat = bestSplit_
                node.splitVal = median_
                node.left = DTNode(majorityClass=majClass,\
                                   X=X_[X_[:,bestSplit_]<=median_], Y=Y_[X_[:,bestSplit_]<=median_])
                node.right = DTNode(majorityClass=majClass,\
                                   X=X_[X_[:,bestSplit_]>median_], Y=Y_[X_[:,bestSplit_]>median_])
                queue.append(node.left)
                queue.append(node.right)
                node.X=None
                node.Y=None
            if self.get_train_data and (cnt-1)%self.record_data_frequency==0:
                self._record_accuracies(X_train, Y_train, X_test, Y_test, X_val, Y_val)
        if self.get_train_data:
                self._record_accuracies(X_train, Y_train, X_test, Y_test, X_val, Y_val)



    def _record_accuracies(self, X_train, Y_train, X_test, Y_test, X_val, Y_val):
        '''
            Records train, test and validation accuracy in train_data dictionary at given state of the tree
        '''
        self.train_data['train_acc'].append(self.score(X_train, Y_train))
        self.train_data['test_acc'].append(self.score(X_test, Y_test))
        self.train_data['val_acc'].append(self.score(X_val, Y_val))

    def _getBestSplit(self, X, Y):
        '''
            Returns (optimal_split, optimal_median, max_IG)
                optimal_split : feature index of dataset X splitting which maximizes Info. gain
                optimal_median : median of optimal_split feature
                max_IG : maximum Information gain across all features
        '''
        n_samples, n_feat = X.shape
        optimal_split, optimal_median = -np.inf, -np.inf
        max_IG = -np.inf
        if X.size%2 == 0:
            median_arr = np.median(np.vstack((np.min(X, axis=0), X)), axis=0)
        else:
            median_arr = np.median(X, axis=0)
        for i in range(n_feat):
            median = median_arr[i]
            ig = self._getInformationGain(X,Y,i,median)
            if ig > max_IG:
                max_IG = ig
                optimal_split = i
                optimal_median = median
        return (optimal_split, optimal_median, max_IG)

    def _entropy(self, X, Y):
        '''
            This method computes entropy of given node
        '''
        n_samples, n_feat = X.shape
        nY1 = (Y==1).sum()
        nY0 = n_samples-nY1

        if n_samples==0:
            return np.inf
        elif nY1==0 or nY0==0:
            return 0
        else:
            return (-nY0*math.log2(nY0/n_samples)-nY1*math.log2(nY1/n_samples))/n_samples

    def _getInformationGain(self, X, Y, i, median):
        '''
            return information gain if we split with ith Feature
        '''
        entBefore = self._entropy(X,Y)

        X_lte = X[X[:,i]<=median]
        Y_lte = Y[X[:,i]<=median]
        X_gt = X[X[:,i]>median]
        Y_gt = Y[X[:,i]>median]

        entAfter = ((Y_lte.shape[0])*self._entropy(X_lte, Y_lte) + (Y_gt.shape[0])*self._entropy(X_gt, Y_gt))/(Y.shape[0])
        return (entBefore-entAfter)

    def score(self, X, Y_true):
        '''
            Return accuracy score over dataset X and actual label Y
        '''
        return (self.predict(X)==Y_true).sum()/Y_true.shape[0]

    def predict(self, X):
        '''
            Returns prediction for given Dataset X
        '''
        pred=[]
        for row in X:
            node=self.root
            while(not node.is_leaf()):
                if row[node.splitFeat]<=node.splitVal:
                    node=node.left
                elif row[node.splitFeat]>node.splitVal:
                    node=node.right
            pred.append(node.majorityClass)
        return pred

    def _set_indices(self, node, X, indices):
        if node.is_leaf():
            node.val_indices = indices
        else:
            lt_indices = np.logical_and((X[:,node.splitFeat]<=node.splitVal), indices)
            rt_indices = np.logical_and((X[:,node.splitFeat]>node.splitVal), indices)
            node.val_indices = np.argwhere(indices).flatten()
            self._set_indices(node.left, X, lt_indices)
            self._set_indices(node.right, X, rt_indices)

    def _post_order_lst(self, node, lst):
        if node:
            self._post_order_lst(node.left, lst)
            self._post_order_lst(node.right, lst)
            lst.append(node)

    def _accuracy_score(self, Y_pred, Y_true):
        return (Y_pred == Y_true).sum()/Y_true.size

    def get_number_of_nodes(self):
        cnt=0
        leaves=0
        queue = [self.root]
        while queue!=[]:
            cnt+=1
            node=queue.pop(0)
            if node.is_leaf():
                leaves+=1
                continue
            else:
                queue.append(node.left); queue.append(node.right)
        return (cnt, leaves)

    def prune(self, X_val, Y_val, X_train=None, Y_train=None, X_test=None, Y_test=None):
        if self.get_train_data and (X_test is None or Y_test is None or X_train is None or Y_train is None):
            raise Exception('Please pass train and Test data to get prune_data or pass get_prune_data=False')

        self._set_indices(self.root, X_val, [True]*Y_val.size)
        node_lst=[]
        self._post_order_lst(self.root, node_lst)

        pred = np.array(self.predict(X_val))
        best_val_acc = self._accuracy_score(pred, Y_val)
        cnt=0
        for node in (node_lst):
            if self.get_prune_data and (cnt%self.record_data_frequency)==0:
                self._record_prune_data(X_train, Y_train, X_val, Y_val, X_test, Y_test)
            cnt+=1
            if node.is_leaf() or (node.val_indices.size)==0:
                continue
            else:
                pred_tmp = pred.copy()
                pred_tmp[node.val_indices]=node.majorityClass
                val_acc = self._accuracy_score(pred_tmp, Y_val)
                if val_acc >= best_val_acc:
                    pred, best_val_acc = pred_tmp, val_acc
                    node.isLeaf, node.left, node.right, node.splitFeat, node.splitVal=True, None, None, None, None
                else:
                    node.isLeaf=False

        if self.get_prune_data:
            self._record_prune_data(X_train, Y_train, X_val, Y_val, X_test, Y_test)

    def _record_prune_data(self, X_train, Y_train, X_val, Y_val, X_test, Y_test):
        self.prune_data['nodes'].append(self.get_number_of_nodes()[0])
        self.prune_data['val_acc'].append(self.score(X_val, Y_val))
        self.prune_data['test_acc'].append(self.score(X_test, Y_test))
        self.prune_data['train_acc'].append(self.score(X_train, Y_train))

    def plot_training_data(self):
        '''
            Plots train, test and validation accuracy as number of nodes increases during training
        '''

        if self.get_train_data:
            plt.figure(figsize=(12,8))
            n_nodes_lst = list(range(1,(len(self.train_data['train_acc'])-1)*self.record_data_frequency, self.record_data_frequency))
            n_nodes_lst.append(self.get_number_of_nodes()[0])
            plt.plot(n_nodes_lst, self.train_data['train_acc'], label='Train Accuracy')
            plt.plot(n_nodes_lst, self.train_data['test_acc'], label='Test Accuracy')
            plt.plot(n_nodes_lst, self.train_data['val_acc'], label='Validation Accuracy')
            plt.legend()
            plt.xlabel('Number of nodes')
            plt.ylabel('Accuracies')
            plt.title('Number of nodes v/s Accuracies as we train the classifier')
            plt.show()
        else:
            raise Exception('Please pass get_train_data=True to get this plot')

    def plot_pruning_data(self):
        '''
            Plots train, test and validation accuracy as number of nodes increases during training
        '''
        if self.get_prune_data:
            plt.figure(figsize=(12,8))
            n_nodes_lst = self.prune_data['nodes']
            plt.plot(n_nodes_lst, self.prune_data['train_acc'], label='Train Accuracy')
            plt.plot(n_nodes_lst, self.prune_data['test_acc'], label='Test Accuracy')
            plt.plot(n_nodes_lst, self.prune_data['val_acc'], label='Validation Accuracy')
            plt.legend()
            plt.xlabel('Number of nodes')
            plt.ylabel('Accuracies')
            plt.title('Number of nodes v/s Accuracies as we prune the tree')
            plt.gca().invert_xaxis()
            plt.show()
        else:
            raise Exception('Please pass get_prune_data=True to get this plot')
