
# PART: Importing Depenndencies
# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

GAMMA = 0.001
train_frac = 0.8
dev_frac = 0.1
test_frac = 0.1

# PART: Loading Dataset - from any soucre
digits = datasets.load_digits()

# PART: Sanity check, and Visulization of data
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

# PART: data pre-processing -- to normalized data, format the data to consumed by mode
# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))


# PART: Define the model
# Create a classifier: a support vector classifier
clf = svm.SVC()
# PART: setting up hyper parameter
hyper_parms = {'gamma': GAMMA}
clf.set_params(**hyper_parms)

# PART: define  train/dev/test data splits
# Split data into 50% train and 50% test subsets
X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size=1-train_frac, shuffle=True
)


X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_frac)/(test_frac+dev_frac), shuffle=True
)
# Train to train model
# Dev to tune hyperparameters
# test to evalute the performance of the model

# Test on unseen data


# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

# PART: Sanity check for predection
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

# PART: evulation Matrix computations
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)


disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

# plt.show()
