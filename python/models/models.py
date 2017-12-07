
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import LinearRegression
import dataformating as df
import newEvaluation as ne


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


final_data_training, final_data_testing = df.getFeatures()
while(1) :
    print('1. TeDy\n2.Lara')
    choice = int(input())
    if choice == 1 :
        predictions = decisionTree(final_data_training, final_data_testing)
        methodName = 'decision_tree'
    elif choice == 2:
        predictions = linearRegression(final_data_training, final_data_testing)
        methodName = 'linear_regression'
    else:
        break
    ne.evaluation (predictions, methodName)