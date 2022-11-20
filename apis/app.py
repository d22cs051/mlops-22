from flask import Flask,request#,flash,redirect,url_for
# from werkzeug.utils import secure_filename
import json
# import os
from joblib import load
# from PIL import Image
# from numpy import asarray
# from skimage.transform import resize,rescale
# from sklearn.preprocessing import StandardScaler

import numpy as np
import os

PATH_NAME = 'models/'
# loading clf
# clf = load("./"+PATH_NAME+'clf_gamma_0.001_C_0.2.joblib')
# making SS object
# scaler = StandardScaler()
# UPLOAD_FOLDER = 'apis/uploads/'
# ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}    

app = Flask(__name__)
# app.secret_key = 'super secret key'
 
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def root():
    return "<b> hello world!!! </b>"

# route sum query params
# @app.route("/sum")
# def sum():
#     args = request.args
#     return args


# route sum to post req.
@app.route("/sum",methods = ["POST"])
def sum():
    print(request.json)
    x = request.json['x']
    y = request.json['y']
    
    z = int(x) + int(y)
    return json.dumps({"sum":z})

# global func to prdict
# def predict(ip_mat):
#     # global clf
#     processed_ip_mat = ip_mat.flatten().reshape(1,-1)
#     # print(clf.predict_proba(processed_ip_mat))
#     return clf.predict(processed_ip_mat)
    
#route to predict with array input
# @app.route('/predict', methods=['POST'])
# def predict_digit():
#     ip_mat1 = request.json["mat1"]
#     ip_mat2 = request.json["mat1"]
#     # ip_mat = np.fromstring(ip_mat, dtype=np.float, sep='\n')
#     # ip_mat = list(map(int,ip_mat))
#     ip_mat1 = np.array(ip_mat1,dtype=np.float)
#     ip_mat2 = np.array(ip_mat2,dtype=np.float)
#     # print(ip_mat1)
#     # print(ip_mat.shape)
#     print(predict(ip_mat1))
#     if predict(ip_mat1) == predict(ip_mat2):
#         return {"result":"Both images are of same number"}
#     return {"result":"Both images are of different numbers"}

# route to model test image input
# file upload code

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/upload', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         # check if the post request has the file part
#         if 'file1' not in request.files and 'file2' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file1 = request.files['file1']
#         file2 = request.files['file2']
#         print(f'file 1: {file1.filename}; file 2: {file2.filename}')
#         # If the user does not select a file, the browser submits an
#         # empty file without a filename.
#         if file1.filename == '' and file2.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         if file1 and file2 and allowed_file(file1.filename) and allowed_file(file2.filename):
#             filename1 = secure_filename(file1.filename)
#             filename2 = secure_filename(file1.filename)
#             file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
#             file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))
#             # clf = load('clf_criterion_entropy_min_samples_split_2_min_samples_leaf_1.joblib')
            
#             image1 = Image.open(f'apis/uploads/{filename1}')
#             image2 = Image.open(f'apis/uploads/{filename2}')
#             numpy_img_data1 = asarray(image1,dtype=np.float)
#             numpy_img_data2 = asarray(image2,dtype=np.float)
#             # resized_img = rescale(numpy_img_data,(8,8))
#             resized_img1 = resize(numpy_img_data1,(8,8))
#             resized_img2 = resize(numpy_img_data2,(8,8))
#             # print(resized_img.shape)
#             # for i in resized_img.flatten():
#             #     print(i)
#             resized_img1 = scaler.fit_transform(resized_img1)
#             print(resized_img1)
#             print(predict(resized_img1))
#             resized_img2 = scaler.fit_transform(resized_img2)
#             print(resized_img2)
#             print(predict(resized_img2))
#             return {"op":json.dumps("")}
#     return '''
#     <!doctype html>
#     <title>Upload new File</title>
#     <h1>Upload new File</h1>
#     <form method=post enctype=multipart/form-data>
#       <input type=file name=file1></br>
#       <input type=file name=file2></br>
#       <input type=submit value=Upload>
#     </form>
#     '''




#route to predict with array input
@app.route('/predict_digit_docker', methods=['POST'])
def predict_digit_docker():
    image = request.json["image"]
    # print(request.json)
    if dict(request.json).get("model_name",None) == None:
        os.system(f'python3 args_accepter.py --clf_name svm')
        with open(f'results/svm_None.txt',"r") as fp:
            s = fp.readlines()
        return {"res":s}
    model_name = str(request.json["model_name"])
    random_state = int(request.json["random_state"])
    # ip_mat = np.fromstring(ip_mat, dtype=np.float, sep='\n')
    # ip_mat = list(map(int,ip_mat))
    image = np.array(image,dtype=np.float)
        
    # print(ip_mat1)
    # print(ip_mat.shape)
    # print(predict(ip_mat1))
    if model_name == None:
        os.system(f'python3 args_accepter.py --clf_name svm')
        with open(f'results/svm_None.txt',"r") as fp:
            s = fp.readlines()
        return {"res":s}
    os.system(f'python3 args_accepter.py --clf_name {model_name} --random_state {random_state}')
    with open(f'results/{model_name}_{random_state}.txt',"r") as fp:
        s = fp.readlines()
    return {"res":s}


if __name__ == "__main__":
    app.run(host ='0.0.0.0', port = 5000, debug = True) 