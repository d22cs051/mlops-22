# PART: library dependencies -- sklear, torch, tensorflow, numpy, transformers

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from utils import data_viz,preprocess_digits,h_param_tuning,train_dev_test_split,print_all_acc



# 1. set the ranges of hyper parameters
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

h_param_comb = [{"gamma": g, "C": c} for g in gamma_list for c in c_list]

# Code stops here if condition fails
assert len(h_param_comb) == len(gamma_list) * len(c_list)


# # Defining fractions
# train_frac = 0.8
# test_frac = 0.1
# dev_frac = 0.1

# Defining New fractions
train_frac = 0.7
test_frac = 0.2
dev_frac = 0.1

# PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)
# housekeeping
del digits



# [1,2,3,4,5,6,7,8,9]

# other types of preprocessing
# - image : 8x8 : resize 16x16, 32x32, 4x4 : flatteing
# - normalize data: mean normalization: [x - mean(X)]
#                 - min-max normalization
# - smoothing the image: blur on the image


# PART: define train/dev/test splits of experiment protocol
# train to train model
# dev to set hyperparameters of the model
# test to evaluate the performance of the model

x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
    data, label, train_frac, dev_frac
)

# PART: Define the model
# Create a classifier: a support vector classifier
clf = svm.SVC()
metric=metrics.accuracy_score
best_model, best_metric, best_h_params = h_param_tuning(h_param_comb, clf, x_train, y_train, x_dev, y_dev,x_test,y_test,metric)

# print_all_acc(clf,x_train, y_train, x_dev, y_dev, x_test,y_test,metric)
# PART: Get test set predictions
# Predict the value of the digit on the test subset
predicted = best_model.predict(x_test)

# # PART: Sanity check of predictions
# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, prediction in zip(axes, x_test, predicted):
#     ax.set_axis_off()
#     image = image.reshape(8, 8)
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title(f"Prediction: {prediction}")

# 4. report the test set accurancy with that best model.
# PART: Compute evaluation metrics
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

print("Best hyperparameters were:")
print(best_h_params)