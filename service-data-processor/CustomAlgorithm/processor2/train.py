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

    parameter1 = float(sys.argv[1])
    parameter2 = str(sys.argv[2])
    answerData = str(sys.argv[3])
    file1 = str(sys.argv[4])
    file2 = str(sys.argv[5])
    dir1 = str(sys.argv[6])
    storage_dir = str(sys.argv[7])
    output = {}
    output['parameter1']=parameter1
    output['parameter2']=parameter2
    output['answerData']=answerData
    output['file1']=file1
    output['file2']=file2
    output['dir1']=dir1
    print(output)
    file_name = storage_dir+'/output'
    with open(file_name,'w') as file_obj:
        file_obj.write("parameter1=1")
        file_obj.write('\r\n')
        file_obj.write("parameter2=strtest")
        file_obj.write('\r\n')
        file_obj.write("file1=")
        file_obj.write(storage_dir+'/temp1')
    temp1 = storage_dir+'/temp1'
    with open(temp1,'w') as file_obj:
        file_obj.write("Hello Python=1")

