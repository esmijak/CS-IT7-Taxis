import numpy as np
from pyspark.sql import *
from schemas import *
from demand_cache import invDemandCache
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import datetime

from .constants import *

def evaluation (predictions, methodName) :
    errorsRMSE = []
    errorsR2 = []

    """ Evaluation rmse : """
    evaluatorRMSE = RegressionEvaluator(labelCol="demand", predictionCol="prediction", metricName="rmse")
    rmse = evaluatorRMSE.evaluate(predictions)
    errorsRMSE.append(rmse)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    evaluatorR2 = RegressionEvaluator(labelCol="demand", predictionCol="prediction", metricName="r2")
    r2 = evaluatorR2.evaluate(predictions)
    errorsR2.append(r2)
    print("R Squared Error (R2) on test data = %g" % r2)


    """ Writing the errors in the files : """
    file = open(methodName + "_rmse.txt", "w")
    file.write("Training set contains " + str(N_DAYS_TRAIN) + " days i.e. "+ str(N_OF_TIME_SLOTS_TRAIN) + " time slots \nTest set contains "+ str(N_DAYS_TEST)+ " days i.e. "+ str(N_OF_TIME_SLOTS_TEST) + " time slots \n")
    for errorIndex in range(N_OF_CLUSTERS):
        file.write("RMSE for cluster " + str(errorIndex) + " is " + str(errorsRMSE[errorIndex]) + "\n")
    file.close()

    file = open(methodName + "_r2.txt", "w")
    file.write("Training set contains " + str(N_DAYS_TRAIN) + " days i.e. "+ str(N_OF_TIME_SLOTS_TRAIN) + " time slots \nTest set contains "+ str(N_DAYS_TEST)+ " days i.e. "+ str(N_OF_TIME_SLOTS_TEST) + " time slots \n")
    for errorIndex in range(N_OF_CLUSTERS):
        file.write("R2 for cluster " + str(errorIndex) + " is " + str(errorsR2[errorIndex]) + "\n")
    file.close()