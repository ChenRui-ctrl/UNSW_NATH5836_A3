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


# Function importing Dataset
def importdata():
    data = pd.read_csv('C:/Users/chenr/Desktop/UNSW_NATH5836_A3/abalone.csv')
    # print(data.keys)

    # Printing the dataswet shape
    print("Dataset Length: ", len(data))
    print("Dataset Shape: ", data.shape)

    # Printing the dataset obseravtions
    print("Dataset: ", data.head())
    return data

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.08, 1.03*height, '%s' % int(height), size=10, family="Times new roman")


def multi_class(data):
    data_classified = data['Rings']
    bins = [0,7,10,15,100]
    rings_bins = pd.cut(data_classified, bins)
    # print(rings_bins)
    df_ring_bins = data.groupby(rings_bins)['Rings'].count()
    X = ['0-7', '8-10', '10-15','>15']
    a = plt.bar(X,df_ring_bins)
    plt.xlabel('Rings')
    plt.title('Multi Class')
    autolabel(a)

    plt.savefig('Multi Classes')


# Function to split the dataset
def splitdataset(data):
    # Separating the target variable
    X = data.values[:, 1:5]
    Y = data.values[:, 0]

    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=100)

    return X, Y, X_train, X_test, y_train, y_test





# Driver code
def main():
    # Building Phase
    data = importdata()
    multi_class(data)

    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)

    






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
