from schemas import *
from pyspark.sql import *
import numpy as np
from fi_features_cache import demandCache
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression

"""Cluster related constant: """
N_OF_CLUSTERS = 358   # number of clusters used : all of them
"""Time related constant: """
TIME_SLOTS_WITHIN_DAY = 144    # day is divided into that number of slots
WEEK_NB_TEST = 23 # start of june
FIRST_WEEK = 1
LAST_WEEK = 27
DAY_IN_WEEK = 7
spark = SparkSession.builder.master('spark://csit7-master:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)
demandCache.init(spark, sqlCtx)



def extract_feature(curFeature) :
    week = curFeature['week']
    day = curFeature['day']
    time_of_day_code = curFeature['time_of_day_code']
    day_of_week = curFeature['day_of_week']
    hour = curFeature['hour']
    minute = curFeature['minute']
    origin = curFeature['origin']
    is_manhattan = curFeature['is_manhattan']
    is_airport = curFeature['is_airport']
    amount = curFeature['amount']
    return time_of_day_code, origin, day_of_week, day, week, hour, minute, is_manhattan, is_airport, amount



def main():
    errorsRMSE_LR = []
    errorsR2_LR = []
    errorsR2_DT = []
    errorsR2_DT5 = []
    errorsRMSE_DT = []
    errorsRMSE_DT5 = []
    rows_training = []
    rows_testing = [[] for i in range(N_OF_CLUSTERS)]

    for week_nb in range (FIRST_WEEK, LAST_WEEK + 1) :
        print('week nb : ', week_nb)
        for day_of_week in range (DAY_IN_WEEK) :
            for time_of_day_code in range (TIME_SLOTS_WITHIN_DAY) :
                for cid in range(N_OF_CLUSTERS):
                    curFeature = demandCache.get_demand(week_nb, day_of_week, time_of_day_code, cid)
                    if curFeature != [] :
                        time_of_day_code, origin, day_of_week, day, week, hour, minute, is_manhattan, is_airport, amount = extract_feature(curFeature)

                        if (week_nb < WEEK_NB_TEST):
                            rows_training.append((time_of_day_code, origin,day_of_week, day, week, hour, minute, is_manhattan, is_airport, amount))
                        else:
                            rows_testing[cid].append((time_of_day_code, origin,day_of_week, day, week, hour, minute, is_manhattan, is_airport, amount))

    df_training = spark.createDataFrame(rows_training,
                               ["time_of_day_code", "origin", "day_of_week", "day", "week", "hour", "minute", "is_manhattan", "is_airport", "amount"])

    assembler = VectorAssembler(inputCols=["time_of_day_code", "origin", "day_of_week", "day", "week", "hour",
                                             "minute", "is_manhattan", "is_airport"],
                                    outputCol='features')
    output_training = assembler.transform(df_training)


    final_data_training = output_training.select('features', 'amount')



    decisionTree = DecisionTreeRegressor(labelCol='amount', maxDepth=3)
    dt_model = decisionTree.fit(final_data_training)

    #print(dt_model.toDebugString)

    decisionTree5 = DecisionTreeRegressor(labelCol='amount', maxDepth=5)
    dt_model5 = decisionTree5.fit(final_data_training)

    #print(dt_model5.toDebugString)

    file = open("DT_final_features_one_model_INFO.txt", "w")
    file.write("DT maxDepth 3 : \n" + dt_model.toDebugString)
    file.write("DT maxDepth 5 : \n"+ dt_model5.toDebugString)
    file.close()


    linearRegression = LinearRegression(labelCol='amount')
    lr_model = linearRegression.fit(final_data_training)

    for cid in range(N_OF_CLUSTERS):
        print('cluster: ', cid)
        df_testing = spark.createDataFrame(rows_testing[cid],
                                           ["time_of_day_code", "origin", "day_of_week", "day", "week", "hour",
                                            "minute", "is_manhattan", "is_airport", "amount"])
        #df_testing.show()
        output_testing = assembler.transform(df_testing)
        final_data_testing = output_testing.select('features', 'amount')
        predictionsDT = dt_model.transform(final_data_testing)
        predictionsDT5 = dt_model5.transform(final_data_testing)
        predictionsLR = lr_model.evaluate(final_data_testing)


        """ Evaluation rmse : """
        rmse = predictionsLR.rootMeanSquaredError
        errorsRMSE_LR.append(rmse)
        #print("Root Mean Squared Error (RMSE) for LR on test data = %g" % rmse)

        r2 = predictionsLR.r2
        errorsR2_LR.append(r2)
        #print("R Squared Error (R2) for LR on test data = %g" % r2)

        """ Evaluation rmse : """
        evaluatorRMSE = RegressionEvaluator(labelCol="amount", predictionCol="prediction", metricName="rmse")
        rmse = evaluatorRMSE.evaluate(predictionsDT)
        rmse5 = evaluatorRMSE.evaluate(predictionsDT5)
        errorsRMSE_DT.append(rmse)
        errorsRMSE_DT5.append(rmse5)
        #print("Root Mean Squared Error (RMSE) for DT on test data = %g" % rmse)

        evaluatorR2 = RegressionEvaluator(labelCol="amount", predictionCol="prediction", metricName="r2")
        r2 = evaluatorR2.evaluate(predictionsDT)
        r25 = evaluatorR2.evaluate(predictionsDT5)
        errorsR2_DT.append(r2)
        errorsR2_DT5.append(r25)
        #print("R Squared Error (R2) for DT on test data = %g" % r2)
    return  errorsRMSE_LR, errorsR2_LR, errorsRMSE_DT, errorsR2_DT, errorsRMSE_DT5, errorsR2_DT5

errorsRMSE_LR, errorsR2_LR, errorsRMSE_DT, errorsR2_DT, errorsRMSE_DT5, errorsR2_DT5 = main()


def writing_error(errorsRMSE_LR, errorsR2_LR, errorsRMSE_DT, errorsR2_DT, errorsRMSE_DT5, errorsR2_DT5) :
    """ Writing the errors in the files : """
    N_WEEK_TRAIN = WEEK_NB_TEST - 1
    N_TS_TRAIN = N_WEEK_TRAIN * DAY_IN_WEEK * TIME_SLOTS_WITHIN_DAY
    N_WEEK_TEST = LAST_WEEK - N_WEEK_TRAIN
    N_TS_TEST = N_WEEK_TEST * DAY_IN_WEEK * TIME_SLOTS_WITHIN_DAY


    file_DT_rmse = open("decision_tree_rmse_final_features_one_model_12_12.txt", "w")
    file_DT_r2 = open("decision_tree_r2_final_features_one_model_12_12.txt", "w")
    file_DT5_rmse = open("decision_tree5_rmse_final_features_one_model_12_12.txt", "w")
    file_DT5_r2 = open("decision_tree5_r2_final_features_one_model_12_12.txt", "w")
    file_LR_rmse = open("linear_regression_rmse_final_features_one_model_12_12.txt", "w")
    file_LR_r2 = open("linear_regression_r2_final_features_one_model_12_12.txt", "w")


    file_DT_rmse.write("Training set contains " + str(N_WEEK_TRAIN) + " weeks i.e. "+ str(N_TS_TRAIN) + " time slots \nTest set contains "+ str(N_WEEK_TEST)+ " weeks i.e. "+ str(N_TS_TEST) + " time slots \n")
    file_DT_r2.write("Training set contains " + str(N_WEEK_TRAIN) + " weeks i.e. "+ str(N_TS_TRAIN) + " time slots \nTest set contains "+ str(N_WEEK_TEST)+ " weeks i.e. "+ str(N_TS_TEST) + " time slots \n")
    file_DT5_rmse.write("Training set contains " + str(N_WEEK_TRAIN) + " weeks i.e. "+ str(N_TS_TRAIN) + " time slots \nTest set contains "+ str(N_WEEK_TEST)+ " weeks i.e. "+ str(N_TS_TEST) + " time slots \n")
    file_DT5_r2.write("Training set contains " + str(N_WEEK_TRAIN) + " weeks i.e. "+ str(N_TS_TRAIN) + " time slots \nTest set contains "+ str(N_WEEK_TEST)+ " weeks i.e. "+ str(N_TS_TEST) + " time slots \n")
    file_LR_rmse.write("Training set contains " + str(N_WEEK_TRAIN) + " weeks i.e. "+ str(N_TS_TRAIN) + " time slots \nTest set contains "+ str(N_WEEK_TEST)+ " weeks i.e. "+ str(N_TS_TEST) + " time slots \n")
    file_LR_r2.write("Training set contains " + str(N_WEEK_TRAIN) + " weeks i.e. "+ str(N_TS_TRAIN) + " time slots \nTest set contains "+ str(N_WEEK_TEST)+ " weeks i.e. "+ str(N_TS_TEST) + " time slots \n")

    for errorIndex in range(N_OF_CLUSTERS):
        file_DT_rmse.write("RMSE for cluster " + str(errorIndex) + " is " + str(errorsRMSE_DT[errorIndex]) + "\n")
        file_DT_r2.write("R2 for cluster " + str(errorIndex) + " is " + str(errorsR2_DT[errorIndex]) + "\n")
        file_DT5_rmse.write("RMSE for cluster " + str(errorIndex) + " is " + str(errorsRMSE_DT5[errorIndex]) + "\n")
        file_DT5_r2.write("R2 for cluster " + str(errorIndex) + " is " + str(errorsR2_DT5[errorIndex]) + "\n")
        file_LR_rmse.write("RMSE for cluster " + str(errorIndex) + " is " + str(errorsRMSE_LR[errorIndex]) + "\n")
        file_LR_r2.write("RMSE for cluster " + str(errorIndex) + " is " + str(errorsR2_LR[errorIndex]) + "\n")

    file_DT_rmse.close()
    file_DT_r2.close()
    file_DT5_rmse.close()
    file_DT5_r2.close()
    file_LR_rmse.close()
    file_LR_r2.close()

