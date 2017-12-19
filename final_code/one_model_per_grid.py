""" This script compute the linear regression and decision tree models for one model per grid cell. """
from schemas import *
from pyspark.sql import *
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

"""Time related constant: """
TIME_SLOTS_WITHIN_DAY = 144    # day is divided into that number of slots
WEEK_NB_TEST = 23 # start of june
FIRST_WEEK = 1
LAST_WEEK = 27
DAY_IN_WEEK = 7
spark = SparkSession.builder.master('spark://csit7-master:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)

def get_features_for_grid(sqlCtx, lat, long, split_train_test=True):
    df = spark.read.parquet(hadoopify('grids/final_with_zero/{}_{}'.format(long, lat))) \
        .orderBy('pickup_lat_slot') \
        .drop('pickup_timeslot_id')
    if split_train_test:
        train = df.filter('week < 23')
        test = df.filter('week >= 23')

        return train, test

    return df.collect()

def get_features_for_all_grid(sqlCtx, split_train_test=True):
    all = []
    for x in range (24):
        for y in range(24):
            df = spark.read.parquet(hadoopify('grids/final_features_grid')).filter('pickup_lat_slot = {}'.format(x)) \
                .filter('pickup_long_slot = {}'.format(y)).orderBy('pickup_lat_slot')
            all.extend(df.collect())
    fin_df = spark.createDataFrame(all,
                               ["amount", "day","day_of_week", "hour", "is_airport","is_manhattan","minute", 'pickup_lat_slot', 'pickup_long_slot', "time_of_day_code", "week" ])
    if split_train_test:
        train = fin_df.filter('week < 23')
        test = fin_df.filter('week >= 23')

        return train, test

    return fin_df.collect()

def doGrid_one():
    grid_data = getGridData(sqlCtx, '_ngrid2500')
    errorsRMSE_LR = []
    errorsR2_LR = []
    errorsRMSE_DT = []
    errorsR2_DT = []
    hor = grid_data['horizontal_slots']
    vert = grid_data['vertical_slots']
    print(hor, vert)
    hor = 24
    vert = 24
    for x in range(hor):
        print('grid hor:', x)
        for y in range(vert):
            train, test = get_features_for_grid(spark, x, y)
            assembler = VectorAssembler(inputCols=["day","day_of_week", "hour", "is_airport","is_manhattan","minute", 'pickup_lat_slot', 'pickup_long_slot', "time_of_day_code", "week"],
                                        outputCol='features')
            output_training = assembler.transform(train)
            output_testing = assembler.transform(test)

            final_data_training = output_training.select('features', 'amount')
            final_data_testing = output_testing.select('features', 'amount')

            final_data_training.describe().show()
            final_data_testing.describe().show()
            

            decisionTree = DecisionTreeRegressor(labelCol='amount', maxDepth=3)
            dt_model = decisionTree.fit(final_data_training)
            predictionsDT = dt_model.transform(final_data_testing)
            print(dt_model.toDebugString)

            linearRegression = LinearRegression(labelCol='amount')
            lr_model = linearRegression.fit(final_data_training)
            predictionsLR = lr_model.evaluate(final_data_testing)

            """ Evaluation LR : """
            rmse = predictionsLR.rootMeanSquaredError
            errorsRMSE_LR.append(rmse)
            #print("Root Mean Squared Error (RMSE) for LR on test data = ", rmse)

            r2 = predictionsLR.r2
            errorsR2_LR.append(r2)
            #print("R Squared Error (R2) for LR on test data = ", r2)

            """ Evaluation DT : """
            evaluatorRMSE = RegressionEvaluator(labelCol="amount", predictionCol="prediction", metricName="rmse")
            rmse = evaluatorRMSE.evaluate(predictionsDT)
            errorsRMSE_DT.append(rmse)
            #print("Root Mean Squared Error (RMSE) for DT on test data = ", rmse)

            evaluatorR2 = RegressionEvaluator(labelCol="amount", predictionCol="prediction", metricName="r2")
            r2 = evaluatorR2.evaluate(predictionsDT)
            errorsR2_DT.append(r2)
            #print("R Squared Error (R2) for DT on test data = ", r2)

    return hor, vert, errorsR2_DT, errorsRMSE_DT, errorsRMSE_LR, errorsR2_LR



def write_in_files (hor, vert, errorsR2_DT, errorsRMSE_DT, errorsRMSE_LR, errorsR2_LR, one) :
    """ Writing the errors in the files : """
    N_WEEK_TRAIN = WEEK_NB_TEST - 1
    N_TS_TRAIN = N_WEEK_TRAIN * DAY_IN_WEEK * TIME_SLOTS_WITHIN_DAY
    N_WEEK_TEST = LAST_WEEK - N_WEEK_TRAIN
    N_TS_TEST = N_WEEK_TEST * DAY_IN_WEEK * TIME_SLOTS_WITHIN_DAY


    file_rmse_dt = open("decision_tree_rmse_final_features_grid"+one+".txt", "w")
    file_r2_dt = open("decision_tree_r2_final_features_grid"+one+".txt", "w")
    file_rmse_lr = open("linear_regression_rmse_final_features_grid"+one+".txt", "w")
    file_r2_lr = open("linear_regression_r2_final_features_grid"+one+".txt", "w")

    strTxt = "Training set contains " + str(N_WEEK_TRAIN) + " weeks i.e. "+ str(N_TS_TRAIN) + " time slots \nTest set contains "+ str(N_WEEK_TEST)+ " weeks i.e. "+ str(N_TS_TEST) + " time slots \n"
    file_rmse_dt.write(strTxt)
    file_r2_dt.write(strTxt)
    file_rmse_lr.write(strTxt)
    file_r2_lr.write(strTxt)

    errorIndex = 0
    for x in range(hor):
        for y in range(vert):
            file_rmse_dt.write("RMSE for grid : (" + str(x)+', '+ str(y) + ") is " + str(errorsRMSE_DT[errorIndex]) + "\n")
            file_r2_dt.write("R2 for grid : (" + str(x)+', '+ str(y) + ") is " + str(errorsR2_DT[errorIndex]) + "\n")
            file_rmse_lr.write("RMSE for grid : (" + str(x)+', '+ str(y) + ") is " + str(errorsRMSE_LR[errorIndex]) + "\n")
            file_r2_lr.write("R2 for grid : (" + str(x)+', '+ str(y) + ") is " + str(errorsR2_LR[errorIndex]) + "\n")
            errorIndex += 1

    file_rmse_dt.close()
    file_r2_dt.close()
    file_rmse_lr.close()
    file_r2_lr.close()


hor, vert, errorsR2_DT, errorsRMSE_DT, errorsRMSE_LR, errorsR2_LR = doGrid_one()
write_in_files (hor, vert, errorsR2_DT, errorsRMSE_DT, errorsRMSE_LR, errorsR2_LR, "")
