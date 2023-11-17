# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Source: https://www.geeksforgeeks.org/decision-tree-implementation-python/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import matplotlib as mpl
mpl.rcParams['figure.max_open_warning'] = 0

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import log_loss
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import ensemble
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor





# Function importing Dataset
def importdata():
    data = pd.read_csv('C:/Users/chenr/Desktop/UNSW_NATH5836_A3/abalone.data')
    data = data.values
    data[:,0] = np.where(data[:,0] == -1, 0, 1)
    for i in range(1, data.shape[1]):
        data = np.delete(data, data[:, i] == 0, axis=0)
    data = data.astype("float")
    return data



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
    decision_tree = DecisionTreeRegressor(max_depth=5, criterion='squared_error')
    detree = decision_tree.fit(X_train, y_train)

    from sklearn.tree import plot_tree
    plt.figure(figsize=(90, 15))
    plot_tree(detree,
              feature_names=['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight',
                             'Shell weight'],
              class_names=['0-7', '8-10', '10-15', '>15'],
              filled=True,
              rounded=True,
              fontsize=14)

    plt.savefig('Regression Decision tree')
    return detree

def tree_choose(data,numebr_depth):
    arr_tr = np.zeros(numebr_depth)
    arr_t = np.zeros(numebr_depth)
    for i in range(1,numebr_depth+1):
        buff_tr = np.zeros(5)
        buff_t = np.zeros(5)
        for j in range(5):
            X_train, X_test, y_train, y_test = splitdataset(data)
            decision_tree = DecisionTreeRegressor (max_depth=i, criterion='squared_error')
            decision_tree.fit(X_train, y_train)
            buff_tr[j] = mean_squared_error(y_true=y_train, y_pred=decision_tree.predict(X_train))
            buff_t[j] =  mean_squared_error(y_true=y_test, y_pred=decision_tree.predict(X_test))
        arr_tr[i-1] = buff_tr.mean()
        arr_t[i-1] = buff_t.mean()
    plt.figure(figsize=(10, 5))
    plt.plot([i for i in range(numebr_depth)], arr_tr, c='orange')
    plt.title('Decision Tree for Train Data')
    plt.xlabel('Max_Depth')
    plt.ylabel('Regression Train MSE')
    plt.savefig('Regression Decision Tree Train MSE.png')
    plt.figure(figsize=(10, 5))
    plt.plot([i for i in range(numebr_depth)], arr_t, c='orange')
    plt.title('Decision Tree for Test Data')
    plt.xlabel('Max_Depth')
    plt.ylabel('Test MSE')
    plt.savefig('Regression Decision Tree Test MSE.png')

def post_pruning(decision_tree,X_train, X_test, y_train, y_test):
    path = decision_tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")
    plt.savefig('Regression Impurity vs Effective.png')

    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeRegressor(random_state=0, ccp_alpha=ccp_alpha,)
        clf.fit(X_train, y_train)
        clfs.append(clf)
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
    plt.savefig('Regression Node vs Alpha vs Depth.png')


    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("MSE vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    plt.savefig('Regression MSE vs Alpha.png')

def pre_prunig(X_train, X_test, y_train, y_test):
    decision_tree = DecisionTreeRegressor(criterion='squared_error', max_depth=4, min_samples_leaf=5, min_samples_split=12,
                                 splitter='random')
    decision_tree.fit(X_train, y_train)
    y_predicted = decision_tree.predict(X_test)
    print('MSE score of pre pruning: ',mean_squared_error(y_test, y_predicted))

def random_forest_make(X_train, X_test, y_train, y_test,number_tree):
    arr_tr = np.zeros(number_tree)
    for i in range(1, number_tree + 1):
        random_forest = RandomForestRegressor(n_estimators=i, max_leaf_nodes=16, n_jobs=-10)
        random_forest.fit(X_train, y_train)
        y_pred = random_forest.predict(X_test)
        arr_tr[i-1] = mean_squared_error(y_test,y_pred)
    plt.figure(figsize=(10, 5))
    plt.plot([i for i in range(number_tree)], arr_tr, c='orange')
    plt.title('Random Forest for Train Data')
    plt.xlabel('Max_Number_Tree')
    plt.ylabel('Train MSE')
    plt.savefig('Regression Random Forest MSE.png')

def GDBT(X_train, X_test, y_train, y_test):
    original_params = {'n_estimators': 1000, 'max_leaf_nodes': 4, 'max_depth': None, 'random_state': 2,
                       'min_samples_split': 5}

    plt.figure()

    for label, color, setting in [('No shrinkage', 'orange',
                                   {'learning_rate': 1.0, 'subsample': 1.0}),
                                  ('learning_rate=0.2', 'turquoise',
                                   {'learning_rate': 0.2, 'subsample': 1.0}),
                                  ('subsample=0.5', 'blue',
                                   {'learning_rate': 1.0, 'subsample': 0.5}),
                                  ('learning_rate=0.2, subsample=0.5', 'gray',
                                   {'learning_rate': 0.2, 'subsample': 0.5}),
                                  ('learning_rate=0.2, max_features=2', 'magenta',
                                   {'learning_rate': 0.2, 'max_features': 2})]:
        params = dict(original_params)
        params.update(setting)

        clf = ensemble.GradientBoostingRegressor(**params)
        clf.fit(X_train, y_train)

        # compute test set deviance
        test_deviance = np.zeros((params['n_estimators'],), dtype=np.float64)

        for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
            # clf.loss_ assumes that y_test[i] in {0, 1}
            test_deviance[i] = 2 * log_loss(y_test, y_pred)

        plt.plot((np.arange(test_deviance.shape[0]) + 1)[::5], test_deviance[::5],
                 '-', color=color, label=label)

    plt.legend(loc='upper left')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Test Set Deviance')
    plt.savefig('Regression GDBT.png')

def XGBoost(data, expruns):

    # clf_tr = xgb.DMatrix(X_train, y_train, enable_categorical=True)
    # clf_t = xgb.DMatrix(X_test, y_test, enable_categorical=True)

    arr_xgb = np.zeros(expruns)
    for i in range(0, expruns):
        X_train, X_test, y_train, y_test = splitdataset(data)
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        xgb_classifier = xgb.XGBRegressor(colsample_bytree = 0.3, learning_rate = 0.1,
            max_depth = 0, alpha = 5, n_estimators = i)
        xgb_classifier.fit(X_train, y_train)
        y_pred = xgb_classifier.predict(X_test)
        arr_xgb[i] = mean_squared_error(y_test, y_pred)

    plt.figure(figsize=(10, 5))
    plt.plot([i for i in range(expruns)], arr_xgb, c='orange')
    plt.title('XGBoost for Train Data')
    plt.xlabel('Max_Experience_Run')
    plt.ylabel('Train MSE')
    plt.savefig('Regression XGBoost MSE.png')

def Adam(data, i):
    X_train, X_test, y_train, y_test = splitdataset(data)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    xgb_classifier = MLPRegressor(random_state=i,solver='adam')
    xgb_classifier.fit(X_train, y_train)
    y_pred = xgb_classifier.predict(X_test)
    arr_adam = mean_squared_error(y_test, y_pred)
    return arr_adam


def SGD(data, i):
    X_train, X_test, y_train, y_test = splitdataset(data)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    xgb_classifier = MLPRegressor(random_state=i,solver='sgd')
    xgb_classifier.fit(X_train, y_train)
    y_pred = xgb_classifier.predict(X_test)
    arr_sgb = mean_squared_error(y_test, y_pred)
    return(arr_sgb)



# Driver code
def main():

    # import data
    data = importdata()

    # create decision tree
    X_train, X_test, y_train, y_test = splitdataset(data)
    decision_tree = tree_make(X_train,y_train)

    #choose the best tree
    numebr_depth = 30
    tree_choose(data,numebr_depth)

    #prune
    post_pruning(decision_tree,X_train, X_test, y_train, y_test)
    pre_prunig(X_train, X_test, y_train, y_test)
    decision_tree_post_pruning = DecisionTreeRegressor(random_state=0, ccp_alpha=0.02,max_depth=3)
    decision_tree_post_pruning.fit(X_train,y_train)
    plt.figure(figsize = (10,5))
    tree.plot_tree(decision_tree_post_pruning,rounded=True,filled=True)
    plt.savefig('Regression Decision Tree After Post Pruning')
    print('MSE score of post pruning:',mean_squared_error(y_test,decision_tree_post_pruning.predict((X_test))))

    #random forest
    number_tree = 60
    random_forest_make(X_train, X_test, y_train, y_test,number_tree)


    #GDBT
    expruns = 50

    # GDBT(X_train, X_test, y_train, y_test)

    #XGBoost
    XGBoost(data, expruns)

    #adam
    arr_adam = np.zeros(expruns)
    for i in range(1, expruns + 1):
        arr_adam[i-1] = Adam(data,i)

    # SGD
    arr_sgd = np.zeros(expruns)
    for i in range(1, expruns + 1):
        arr_sgd[i-1] = SGD(data, i)

    plt.figure(figsize=(10, 5))
    plt.plot([i for i in range(expruns)], arr_adam, c='orange', label='Adam')
    plt.plot([i for i in range(expruns)], arr_sgd, c='blue', label='SGD')
    plt.legend()
    plt.title('Sgd & Adam for Train Data')
    plt.xlabel('Max_Experience_Run')
    plt.ylabel('Train Accuracy')
    plt.savefig('Regression Sgd vs Adam MSE.png')



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/