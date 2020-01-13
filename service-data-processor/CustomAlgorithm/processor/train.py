#
# train_diabetes.py
#
#   MLflow model using ElasticNet (sklearn) and Plots ElasticNet Descent Paths
#
#   Uses the sklearn Diabetes dataset to predict diabetes progression using ElasticNet
#       The predicted "progression" column is a quantitative measure of disease progression one year after baseline
#       http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
#   Combines the above with the Lasso Coordinate Descent Path Plot
#       http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_coordinate_descent_path.html
#       Original author: Alexandre Gramfort <alexandre.gramfort@inria.fr>; License: BSD 3 clause
#
#  Usage:
#    python train_diabetes.py 0.01 0.01
#    python train_diabetes.py 0.01 0.75
#    python train_diabetes.py 0.01 1.0
#
import os
import warnings
import sys

if __name__ == "__main__":

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    l1_ratio = int(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    file = str(sys.argv[3])
    file2 = str(sys.argv[4])
    storage_dir= str(sys.argv[5])
    output = {}
    output['alpha']=alpha
    output['l1_ratio']=l1_ratio
    output['file']=file
    output['file2']=file2
    print(output)

    file_name = storage_dir+'/output'
    with open(file_name,'w') as file_obj:
        file_obj.write("parameter1=1")
        file_obj.write('\r\n')
        file_obj.write("parameter2=strtest")
        file_obj.write('\r\n')
        file_obj.write("file1=")
        file_obj.write(storage_dir+'/temp1')
        file_obj.write('\r\n')
        file_obj.write("file2=")
        file_obj.write(storage_dir+'/temp2')
        file_obj.write('\r\n')
        file_obj.write("dir1=")
        file_obj.write(storage_dir+'/dir1/')
    temp1 = storage_dir+'/temp1'
    temp2 = storage_dir + '/temp2'
    with open(temp1,'w') as file_obj:
        file_obj.write("Hello Python=1")
    with open(temp2,'w') as file_obj:
        file_obj.write("Hello Java=1")
    data_dir = storage_dir+'/dir1/'
    if data_dir is not None and not os.path.exists(data_dir):
        os.makedirs(data_dir)
    file_name = data_dir+'temp3'
    with open(file_name,'w') as file_obj:
        file_obj.write("Hello C++=1")

