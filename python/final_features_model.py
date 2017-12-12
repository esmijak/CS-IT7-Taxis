from schemas import *
from pyspark.sql import *
from fi_features_cache import demandCache
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression

"""Cluster related constant: """
N_OF_CLUSTERS = 358   # number of clusters used : all of them
"""Time related constant: """
TIME_SLOTS_WITHIN_DAY = 144    # day is divided into that number of slots
N_DAYS_JAN = 31
N_DAYS_FEB = 28
N_DAYS_MAR = 31
N_DAYS_APR = 30
N_DAYS_MAY = 31
N_DAYS_JUN = 29
FIRST_DAY_OF_WEEK = 3   # which day of the week was the first day of the year 2015 (0 - Monday, 1 - Tuesday, etc.)
N_DAYS_TRAIN = N_DAYS_JAN + N_DAYS_FEB + N_DAYS_MAR + N_DAYS_APR + N_DAYS_MAY # number of days used for the learning

#N_DAYS_TEST = N_DAYS_JUN
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

errorsRMSE_LR = []
errorsR2_LR = []
errorsRMSE_DT = []
errorsR2_DT = []

def main():
    for cid in range(N_OF_CLUSTERS):
        rows_training = []
        rows_testing = []
        for week_nb in range (FIRST_WEEK, LAST_WEEK + 1) :
            print('week nb : ', week_nb)
            for day_of_week in range (DAY_IN_WEEK) :
                for time_of_day_code in range (TIME_SLOTS_WITHIN_DAY) :
                    #for tid in range(TOTAL_SLOTS_FOR_LOOP): #TODO do the loop per week, per day and day slot and change fi_features_cache too

                    curFeature = demandCache.get_demand(week_nb, day_of_week, time_of_day_code, cid)
                    if curFeature != [] :
                        time_of_day_code, origin, day_of_week, day, week, hour, minute, is_manhattan, is_airport, amount = extract_feature(curFeature)

                        if (week_nb < WEEK_NB_TEST):
                            rows_training.append((time_of_day_code, origin,day_of_week, day, week, hour, minute, amount))
                        else:
                            rows_testing.append((time_of_day_code, origin,day_of_week, day, week, hour, minute, amount))
        df_training = spark.createDataFrame(rows_training,
                                   ["time_of_day_code", "origin", "day_of_week", "day", "week", "hour", "minute", "amount"])
        df_testing = spark.createDataFrame(rows_testing,
                                                ["time_of_day_code", "origin", "day_of_week", "day", "week", "hour",
                                                 "minute", "amount"])

        assembler = VectorAssembler(inputCols=["time_of_day_code", "origin", "day_of_week", "day", "week", "hour",
                                                 "minute"],
                                        outputCol='features')
        output_training = assembler.transform(df_training)
        output_testing = assembler.transform(df_testing)

        final_data_training = output_training.select('features', 'amount')
        final_data_testing = output_testing.select('features', 'amount')

        decisionTree = DecisionTreeRegressor(labelCol='amount', maxDepth=3)
        dt_model = decisionTree.fit(final_data_training)
        predictionsDT = dt_model.transform(final_data_testing)

        linearRegression = LinearRegression(labelCol='amount')
        lr_model = linearRegression.fit(final_data_training)
        predictionsLR = lr_model.evaluate(final_data_testing)

        # print("Decision tree model max depth = %g" % decisionTree.getMaxDepth())
        # print(dt_model.toDebugString)


        """ Evaluation rmse : """
        rmse = predictionsLR.rootMeanSquaredError
        errorsRMSE_LR.append(rmse)
        print("Root Mean Squared Error (RMSE) for LR on test data = %g" % rmse)

        r2 = predictionsLR.r2
        errorsR2_LR.append(r2)
        print("R Squared Error (R2) for LR on test data = %g" % r2)

        """ Evaluation rmse : """
        evaluatorRMSE = RegressionEvaluator(labelCol="amount", predictionCol="prediction", metricName="rmse")
        rmse = evaluatorRMSE.evaluate(predictionsDT)
        errorsRMSE_DT.append(rmse)
        print("Root Mean Squared Error (RMSE) for DT on test data = %g" % rmse)

        evaluatorR2 = RegressionEvaluator(labelCol="amount", predictionCol="prediction", metricName="r2")
        r2 = evaluatorR2.evaluate(predictionsDT)
        errorsR2_DT.append(r2)
        print("R Squared Error (R2) for DT on test data = %g" % r2)

main()

""" Writing the errors in the files : """
N_WEEK_TRAIN = WEEK_NB_TEST - 1
N_TS_TRAIN = N_WEEK_TRAIN * DAY_IN_WEEK * TIME_SLOTS_WITHIN_DAY
N_WEEK_TEST = LAST_WEEK - N_WEEK_TRAIN
N_TS_TEST = N_WEEK_TEST * DAY_IN_WEEK * TIME_SLOTS_WITHIN_DAY
file = open("decision_tree_rmse_final_features.txt", "w")
file.write("Training set contains " + str(N_WEEK_TRAIN) + " weeks i.e. "+ str(N_TS_TRAIN) + " time slots \nTest set contains "+ str(N_WEEK_TEST)+ " weeks i.e. "+ str(N_TS_TEST) + " time slots \n")
for errorIndex in range(N_OF_CLUSTERS):
    file.write("RMSE for cluster " + str(errorIndex) + " is " + str(errorsRMSE_DT[errorIndex]) + "\n")
file.close()

file = open("decision_tree_r2_final_features.txt", "w")
file.write("Training set contains " + str(N_WEEK_TRAIN) + " weeks i.e. "+ str(N_TS_TRAIN) + " time slots \nTest set contains "+ str(N_WEEK_TEST)+ " weeks i.e. "+ str(N_TS_TEST) + " time slots \n")
for errorIndex in range(N_OF_CLUSTERS):
    file.write("R2 for cluster " + str(errorIndex) + " is " + str(errorsR2_DT[errorIndex]) + "\n")
file.close()


""" Writing the errors in the files : """
file = open("linear_regression_rmse_final_features.txt", "w")
file.write("Training set contains " + str(N_WEEK_TRAIN) + " weeks i.e. "+ str(N_TS_TRAIN) + " time slots \nTest set contains "+ str(N_WEEK_TEST)+ " weeks i.e. "+ str(N_TS_TEST) + " time slots \n")
for errorIndex in range(N_OF_CLUSTERS):
    file.write("RMSE for cluster " + str(errorIndex) + " is " + str(errorsRMSE_LR[errorIndex]) + "\n")
file.close()

file = open("linear_regression_r2_final_features.txt", "w")
file.write("Training set contains " + str(N_WEEK_TRAIN) + " weeks i.e. "+ str(N_TS_TRAIN) + " time slots \nTest set contains "+ str(N_WEEK_TEST)+ " weeks i.e. "+ str(N_TS_TEST) + " time slots \n")
for errorIndex in range(N_OF_CLUSTERS):
    file.write("R2 for cluster " + str(errorIndex) + " is " + str(errorsR2_LR[errorIndex]) + "\n")
file.close