import sys
import os
from sklearn import datasets,svm
from sklearn.model_selection import train_test_split
sys.path.append('.')
from utils import get_all_h_params_comb, train_save_model, data_viz, preprocess_digits
from plot_digit_classification import h_param
from joblib import dump,load

def test_not_all_to_same_class():
    model_path = None
    digits = datasets.load_digits()
    data_viz(digits)
    data, label = preprocess_digits(digits)
    smaple_data = data[:500]
    smaple_label = label[:500]
    h_param_comb = h_param()
    clf = svm.SVC()
    actual_modle_path, clf = train_save_model(clf, smaple_data, smaple_label, smaple_data, smaple_label, model_path, h_param_comb)
    # clf = load(model_path)
    predicted = clf.predict(smaple_data)
    # not all mapping to same class
    assert len(set(predicted)) != 1
    

def test_all_class_predicted():
    model_path = None
    digits = datasets.load_digits()
    data_viz(digits)
    data, label = preprocess_digits(digits)
    smaple_data = data[:500]
    smaple_label = label[:500]
    h_param_comb = h_param()
    # clf = load(model_path)
    clf = svm.SVC()
    actual_modle_path, clf = train_save_model(clf,smaple_data, smaple_label, smaple_data, smaple_label, model_path, h_param_comb)

    predicted = clf.predict(smaple_data)
    # not all mapping to same class
    assert len(set(predicted)) == 10