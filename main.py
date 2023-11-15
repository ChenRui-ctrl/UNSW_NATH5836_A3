# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Source: https://www.geeksforgeeks.org/decision-tree-implementation-python/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import export_graphviz
from subprocess import call


# Function importing Dataset
def importdata():
    data = pd.read_csv('C:/Users/chenr/Desktop/UNSW_NATH5836_A3/abalone.data')
    data = data.values
    data[:,0] = np.where(data[:,0] == -1, 0, 1)
    for i in range(1, data.shape[1]):
        data = np.delete(data, data[:, i] == 0, axis=0)
    for j in range(data.shape[0]):
        mid = data[j, -1]
        if mid < 8:
            data[j, -1] = 1
        elif mid < 11:
            data[j, -1] = 2
        elif mid < 16:
            data[j, -1] = 3
        else:
            data[j, -1] = 4
    data = data.astype("float")
    return data

def read_data():
    abalone = pd.read_csv("C:/Users/chenr/Desktop/pythonProject/abalone.data", sep=',', header=None)
    abalone = abalone.values
    ### combine female and male feature as adult
    abalone[:, 0] = np.where(abalone[:, 0] == -1, 0, 1)
    ### clean 0 data in this dataset
    for i in range(1, abalone.shape[1]):
        abalone = np.delete(abalone, abalone[:, i] == 0, axis=0)

    ### treat the output part by label code which means 0-7 years as 1, 8-10 years as 2, and so on
    for j in range(abalone.shape[0]):
        mid = abalone[j, -1]
        if mid < 8:
            abalone[j, -1] = 1
        elif mid < 11:
            abalone[j, -1] = 2
        elif mid < 16:
            abalone[j, -1] = 3
        else:
            abalone[j, -1] = 4
    ### change to float data type
    abalone = abalone.astype("float")

    return (abalone)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.08, 1.03*height, '%s' % int(height), size=10, family="Times new roman")


def multi_class(data, class_name):
    # data_classified = data[8]
    # df_data_classified = pd.DataFrame(data_classified)
    df_data = pd.DataFrame(data)
    data_classified = df_data[8]
    bins = [0,7,10,15,100]
    rings_bins = pd.cut(data_classified, bins)
    # print(rings_bins)
    df_ring_bins = df_data.groupby(rings_bins)[8].count()
    X = class_name
    a = plt.bar(X,df_ring_bins)
    plt.xlabel('Rings')
    plt.title('Multi Class')
    autolabel(a)

    plt.savefig('Multi Classes')
    # data = df_data.values
    # return data


# Function to split the dataset
def splitdataset(data):
    # Separating the target variable
    X = data[:, :-1]
    Y = data[:, -1]

    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=100)

    return  X_train, X_test, y_train, y_test


def tree_make(X_train,y_train):
    decision_tree = DecisionTreeClassifier(max_depth=5, criterion='gini')
    detree = decision_tree.fit(X_train, y_train)

    from sklearn.tree import plot_tree
    plt.figure(figsize=(25, 10))
    plot_tree(detree,
              feature_names=['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight',
                             'Shell weight'],
              class_names=['0-7', '8-10', '10-15', '>15'],
              filled=True,
              rounded=True,
              fontsize=14)

    plt.savefig('Decision tree')
    return detree

def tree_choose(data,numebr_depth):
    arr_tr = np.zeros(numebr_depth)
    arr_t = np.zeros(numebr_depth)
    for i in range(1,numebr_depth+1):
        X_train, X_test, y_train, y_test = splitdataset(data)
        decision_tree = DecisionTreeClassifier(max_depth=i, criterion='gini')
        decision_tree.fit(X_train, y_train)
        acc_tr = accuracy_score(y_true=y_train, y_pred=decision_tree.predict(X_train))
        acc_t =  accuracy_score(y_true=y_test, y_pred=decision_tree.predict(X_test))
        arr_tr[i-1] = acc_tr
        arr_t[i-1] = acc_t
    plt.figure(figsize=(10, 5))
    plt.plot([i for i in range(numebr_depth)], arr_tr, c='orange')
    plt.title('Decision Tree for Train Data')
    plt.xlabel('Max_Depth')
    plt.ylabel('Train Accuracy')
    plt.savefig('Acc_Train.png')
    plt.figure(figsize=(10, 5))
    plt.plot([i for i in range(numebr_depth)], arr_t, c='orange')
    plt.title('Decision Tree for Test Data')
    plt.xlabel('Max_Depth')
    plt.ylabel('Test Accuracy')
    plt.savefig('Acc_T.png')

def post_pruning(decision_tree,X_train, X_test, y_train, y_test):
    path = decision_tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")
    plt.savefig('Impurity vs Effective.png')

    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)
    print(
        "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
            clfs[-1].tree_.node_count, ccp_alphas[-1]
        )
    )
    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]

    node_counts = [clf.tree_.node_count for clf in clfs]
    depth = [clf.tree_.max_depth for clf in clfs]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs alpha")
    fig.tight_layout()
    plt.savefig('Node vs Alpha vs Depth.png')

    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    plt.savefig('Accuracy vs Alpha.png')

# Driver code
def main():
    # import data
    data = importdata()

    # create multiiclass
    class_name = ['0-7', '8-10', '10-15','>15']
    multi_class(data, class_name)

    # create decision tree
    X_train, X_test, y_train, y_test = splitdataset(data)
    decision_tree = tree_make(X_train,y_train)

    #choose the best tree
    numebr_depth = 30
    tree_choose(data,numebr_depth)

    #prune
    post_pruning(decision_tree,X_train, X_test, y_train, y_test)








# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
