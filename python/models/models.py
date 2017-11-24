
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import LinearRegression
from .constants import *

from .newDataFormating import *
from .newEvaluation import *

def decisionTree(final_data_training, final_data_testing) :
    """  Model and predictions : """
    decisionTree = DecisionTreeRegressor(labelCol='demand', maxDepth=3)
    dt_model = decisionTree.fit(final_data_training)
    predictions = dt_model.transform(final_data_testing)
    return predictions

def linearRegression(final_data_training, final_data_testing):
    """  Model and predictions : """
    linearRegression = LinearRegression(labelCol='demand')
    lr_model = linearRegression.fit(final_data_training)
    predictions = lr_model.evaluate(final_data_testing)
    return predictions


final_data_training, final_data_testing = getFeatures()

print('1. TeDy\n2.Lara')
choice = int(input())
if choice == 1 :
    predictions = decisionTree(final_data_training, final_data_testing)
    methodName = 'decision_tree'
else:
    predictions = linearRegression(final_data_training, final_data_testing)
    methodName = 'linear_regression'

evaluation (predictions, methodName)