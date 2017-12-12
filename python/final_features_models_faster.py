from schemas import *
from pyspark.sql import *
from fi_features_cache import demandCache
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler



"""Cluster related constant: """
N_OF_CLUSTERS = 1#358   # number of clusters used : all of them
"""Time related constant: """
TIME_SLOTS_WITHIN_DAY = 144    # day is divided into that number of slots
WEEK_NB_TEST = 23 # start of june
FIRST_WEEK = 1
LAST_WEEK = 27
DAY_IN_WEEK = 7
spark = SparkSession.builder.master('spark://csit7-master:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)
registerTable(sqlCtx, Table.FINAL_FEATURES)

def get_features_for_cluster(sqlCtx, cluster, split_train_test=True):
    df = loadDataFrame(sqlCtx, Table.FINAL_FEATURES).filter('origin = {}'.format(cluster)).orderBy('origin')\
        .drop('origin', 'pickup_timeslot_id')
    if split_train_test:
        train = df.filter('week < 23')
        test = df.filter('week >= 23')

        return train, test

    return df.collect()



errorsRMSE_LR = []
errorsR2_LR = []
errorsRMSE_DT = []
errorsR2_DT = []

for cluster in range (N_OF_CLUSTERS):
    tr, te = get_features_for_cluster(sqlCtx, cluster)


    assembler = VectorAssembler(inputCols=["time_of_day_code", "day_of_week", "day", "week", "hour","minute"],
                                outputCol='features')
    output_training = assembler.transform(tr)
    output_testing = assembler.transform(te)

    final_data_training = output_training.select('features', 'amount')
    final_data_testing = output_testing.select('features', 'amount')


    decisionTree = DecisionTreeRegressor(labelCol='amount', maxDepth=3)
    dt_model = decisionTree.fit(final_data_training)
    predictionsDT = dt_model.transform(final_data_testing)

    linearRegression = LinearRegression(labelCol='amount')
    lr_model = linearRegression.fit(final_data_training)
    predictionsLR = lr_model.evaluate(final_data_testing)


    """ Evaluation rmse : """
    rmse = predictionsLR.rootMeanSquaredError
    errorsRMSE_LR.append(rmse)
    #print("Root Mean Squared Error (RMSE) for LR on test data = ", rmse)

    r2 = predictionsLR.r2
    errorsR2_LR.append(r2)
    #print("R Squared Error (R2) for LR on test data = ", r2)

    """ Evaluation rmse : """
    evaluatorRMSE = RegressionEvaluator(labelCol="amount", predictionCol="prediction", metricName="rmse")
    rmse = evaluatorRMSE.evaluate(predictionsDT)
    errorsRMSE_DT.append(rmse)
    #print("Root Mean Squared Error (RMSE) for DT on test data = ", rmse)

    evaluatorR2 = RegressionEvaluator(labelCol="amount", predictionCol="prediction", metricName="r2")
    r2 = evaluatorR2.evaluate(predictionsDT)
    errorsR2_DT.append(r2)
    #print("R Squared Error (R2) for DT on test data = ", r2)


""" Writing the errors in the files : """
N_WEEK_TRAIN = WEEK_NB_TEST - 1
N_TS_TRAIN = N_WEEK_TRAIN * DAY_IN_WEEK * TIME_SLOTS_WITHIN_DAY
N_WEEK_TEST = LAST_WEEK - N_WEEK_TRAIN
N_TS_TEST = N_WEEK_TEST * DAY_IN_WEEK * TIME_SLOTS_WITHIN_DAY


file = open("decision_tree_rmse_final_features_fast.txt", "w")
file.write("Training set contains " + str(N_WEEK_TRAIN) + " weeks i.e. "+ str(N_TS_TRAIN) + " time slots \nTest set contains "+ str(N_WEEK_TEST)+ " weeks i.e. "+ str(N_TS_TEST) + " time slots \n")
for errorIndex in range(N_OF_CLUSTERS):
    file.write("RMSE for cluster " + str(errorIndex) + " is " + str(errorsRMSE_DT[errorIndex]) + "\n")
file.close()

file = open("decision_tree_r2_final_features_fast.txt", "w")
file.write("Training set contains " + str(N_WEEK_TRAIN) + " weeks i.e. "+ str(N_TS_TRAIN) + " time slots \nTest set contains "+ str(N_WEEK_TEST)+ " weeks i.e. "+ str(N_TS_TEST) + " time slots \n")
for errorIndex in range(N_OF_CLUSTERS):
    file.write("R2 for cluster " + str(errorIndex) + " is " + str(errorsR2_DT[errorIndex]) + "\n")
file.close()


""" Writing the errors in the files : """
file = open("linear_regression_rmse_final_features_fast.txt", "w")
file.write("Training set contains " + str(N_WEEK_TRAIN) + " weeks i.e. "+ str(N_TS_TRAIN) + " time slots \nTest set contains "+ str(N_WEEK_TEST)+ " weeks i.e. "+ str(N_TS_TEST) + " time slots \n")
for errorIndex in range(N_OF_CLUSTERS):
    file.write("RMSE for cluster " + str(errorIndex) + " is " + str([errorIndex]) + "\n")
file.close()

file = open("linear_regression_r2_final_features_fast.txt", "w")
file.write("Training set contains " + str(N_WEEK_TRAIN) + " weeks i.e. "+ str(N_TS_TRAIN) + " time slots \nTest set contains "+ str(N_WEEK_TEST)+ " weeks i.e. "+ str(N_TS_TEST) + " time slots \n")
for errorIndex in range(N_OF_CLUSTERS):
    file.write("R2 for cluster " + str(errorIndex) + " is " + str(errorsR2_LR[errorIndex]) + "\n")
file.close