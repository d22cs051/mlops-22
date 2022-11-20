# PART: library dependencies: -- sklearn, tenserflow, numpy
# import required packages
import matplotlib.pyplot as plt
import random
import numpy as np
# import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, tree, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import get_all_h_params_comb_tree, preprocess_digits, data_viz, train_dev_test_split, h_param_tuning, get_all_h_params_comb, train_save_model
from joblib import dump,load



# 1. set the range of hyperparameters
gamma_list  = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]


# hyper params for svm
params = {}
params['gamma'] = gamma_list
params['C'] = c_list

#hyper params for tree
params_tree = {}
params_tree['criterion'] = ["gini" ,"entropy", "log_loss"]
params_tree['min_samples_split'] = [2,3,4]
params_tree['min_samples_leaf'] = [1,2,3]



# 2. for every combiation of hyper parameter values
h_param_comb = get_all_h_params_comb(params)
h_param_comb_tree = get_all_h_params_comb_tree(params_tree)

assert len(h_param_comb) == len(gamma_list)*len(c_list)

def h_param():
    return h_param_comb



train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1


# PART: -load dataset --data from files, csv, tsv, json, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)

# housekeeping
del digits



# PART: define train/dev/test splite of experiment protocol
# trin to train model
# dev to set hyperperameter of the model
# test to evaluate the performane of the model

# 80:10:10 train:dev:test


# if testing on the same as traning set: the performance matrics may overestimate 
# the goodness of the model. 
# We want to test on "unseen" sample. 

# X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(
#     data, label, train_frac, dev_frac, 1 - (train_frac + dev_frac), random_state = 5
# )

# X_train, y_train, X_test, y_test = train_test_split(
#     data, label, test_size=0.2, random_state = 5
# )


svm_acc = []
tree_acc = []

for i in range(5):

    X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(
        data, label, train_frac, dev_frac, 1 - (train_frac + dev_frac)
    )
    svm_clf = svm.SVC(probability=True)
    model_path, svm_clf = train_save_model(svm_clf,X_train, y_train, X_dev, y_dev, None, h_param_comb)
    
    # tree classifier
    tree_clf = tree.DecisionTreeClassifier()
    # tree_clf = tree_clf.fit(X_train, y_train)

    # tree_pre = tree_clf.predict(X_test)
    model_path_tree, tree_clf = train_save_model(tree_clf,X_train,y_train,X_dev,y_dev,None,h_param_comb_tree)






    best_model = load(model_path)
    best_model_tree = load(model_path_tree)


    predicted= best_model.predict(X_test) 
    predicted_tree= best_model_tree.predict(X_test) 

    # PART: sanity check visulization of data
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

    svm_acc.append(accuracy_score(y_test, predicted))
    tree_acc.append(accuracy_score(y_test, predicted_tree))


# PART: Compute evaluation Matrics 
# 4. report the best set accuracy with that best model.

print('svm accuracy:')
print(*svm_acc)
print('tree accuracy:')
print(*tree_acc)


svm_acc = np.array(svm_acc)
tree_acc = np.array(tree_acc)

svm_mean = np.mean(svm_acc)
tree_mean = np.mean(tree_acc)

svm_variance = np.var(svm_acc)
tree_variance = np.var(tree_acc)

print(f"svm mean: {svm_mean} \t variance: {svm_variance}")
print(f'tree mean: {tree_mean} \t variance: {tree_variance}')

if svm_mean > tree_mean:
    print("svm is best")
    print(
        f"Classification report for classifier {svm_clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    ) 

    print(f"Best hyperparameters were: {best_model}")

else:
    print("tree is best")
    print(
        f"Classification report for classifier {tree_clf}:\n"
        f"{metrics.classification_report(y_test, predicted_tree)}\n"
    ) 

    print(f"Best hyperparameters were: {best_model_tree}")
