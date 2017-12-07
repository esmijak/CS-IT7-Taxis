import numpy as np
from pyspark.sql import *
from schemas import *
from demand_cache import invDemandCache
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import datetime

"""Cluster related constant: """
N_OF_CLUSTERS = 358   # number of clusters used : all of them

"""Time related constant: """
TIME_SLOTS_WITHIN_DAY = 144     # day is divided into that number of slots
N_DAYS_JAN = 31
N_DAYS_FEB = 28
N_DAYS_MAR = 31
N_DAYS_APR = 30
N_DAYS_MAY = 31
N_DAYS_JUN = 30
FIRST_DAY_DAY_OF_WEEK = 3   # which day of the week was the first day of the year 2015 (0 - Monday, 1 - Tuesday, etc.)
N_DAYS_TRAIN = N_DAYS_JAN + N_DAYS_FEB + N_DAYS_MAR + N_DAYS_APR + N_DAYS_MAY # number of days used for the learning
N_OF_TIME_SLOTS_TRAIN = N_DAYS_TRAIN * TIME_SLOTS_WITHIN_DAY # number of time slots that are being used for training
N_DAYS_TEST = N_DAYS_JUN
N_OF_TIME_SLOTS_TEST =  N_DAYS_TEST * TIME_SLOTS_WITHIN_DAY


"""Spark initialization: """
spark = SparkSession.builder.master('spark://csit7-master:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)

demandTable = loadDataFrame(sqlCtx, Table.DEMAND_SAMPLE)
#demandTable.show(10)
slotsTable = loadDataFrame(sqlCtx, Table.TIME_SLOTS)
slotsTableCol = slotsTable.collect()
#slotsTable.show()

slot_count = len(slotsTableCol)
time_per_slot = (60 * 60 * 24 * 180) / slot_count
start_time = datetime.datetime(2015, 1, 1, 0).timestamp()

def find_slot(time):
    return int((time.timestamp() - start_time) / time_per_slot)

def get_time_slot_per_day(hour, minute):
    nb_minute_slot = minute / 10
    time_slot_per_hour = 6
    slot_nb_day = hour * time_slot_per_hour + nb_minute_slot
    return slot_nb_day

"""Tables requests: """
registerTable(sqlCtx, Table.DEMAND)
registerTable(sqlCtx, Table.TIME_SLOTS)
""" df is data frame containing the time slot id, demand"""
df = spark.sql('SELECT pickup_timeslot_id, pickup_cid, cnt, from, to FROM demand INNER JOIN time_slots ON (pickup_timeslot_id = id) ORDER BY 1,2')
#df.show(10)

errorsRMSE = []
errorsR2 = []

#cc = [253] cluster to test
for curCluster in range (N_OF_CLUSTERS):
#for curCluster in cc: to test a specific cluster
    print('current cluster number is: ', curCluster)

    """ df is data frame containing the time slot id, demand for the current cluster"""
    df_for_one_cluster = df[df.pickup_cid == curCluster].select('pickup_timeslot_id', 'cnt')
    #df_for_one_cluster.show()

    demandListDict = df_for_one_cluster.collect()
    #ld_size = round(0.7 * slot_count)

    rows_training = []
    rows_testing = []
    demandCount = 0
    TOTAL_SLOTS_FOR_LOOP = N_OF_TIME_SLOTS_TEST + N_OF_TIME_SLOTS_TRAIN
    #print('total time slots in loop: ', TOTAL_SLOTS_FOR_LOOP)
    for instance in slotsTableCol :

        slot_nb = find_slot(instance[0])
        """Test whether the training and testing sets are filled with the aimed number of instances."""
        if slot_nb > TOTAL_SLOTS_FOR_LOOP - 1:#  or demandCount > len(demandListDict) - 1:
            #print('slot nb', slot_nb, 'demC:', demandCount, ' li: ', len(demandListDict))
            break
        """Compute the time slot information: """
        weekday = instance[0].weekday()
        day = instance[0].day
        week_nb =  instance[0].isocalendar()[1]
        hour = instance[0].hour
        minute = instance[0].minute

        if (demandCount < len(demandListDict)):
            """Extract the demand of the given time slot for the current cluster: """
            demandDict = demandListDict[demandCount].asDict()
            demandSlot = demandDict['pickup_timeslot_id']

            """Since the table doesn't take into account the demand that are = 0: """
            if slot_nb == demandSlot :
                demand = demandDict['cnt']
                demandCount += 1
            elif slot_nb < demandSlot :
                demand = 0
            else :
                print('coucou should not come here: ', slot_nb, demandSlot)
                demand = 0
        else:
            demand = 0

        """Add the current instance to the right test: """
        slot_nb_day = get_time_slot_per_day(hour, minute)
        if (len(rows_training) < N_OF_TIME_SLOTS_TRAIN):
            rows_training.append((slot_nb_day, weekday, day, week_nb, hour, minute, demand))
        else:
            rows_testing.append((slot_nb_day, weekday, day, week_nb, hour, minute, demand))


    #print('train rows len: ', len(rows_training), 'test rows len: ', len(rows_testing))

    """Create 2 dataframes corresponding to the training and testing set previously computed: """
    df_training = spark.createDataFrame(rows_training, ["slot_id", "day_of_week", "day_of_month", "week_nb", "hour", "minute", "demand"])
    df_testing = spark.createDataFrame(rows_testing, ["slot_id", "day_of_week", "day_of_month", "week_nb", "hour", "minute", "demand"])


    assembler = VectorAssembler(inputCols=["slot_id", "day_of_week", "day_of_month", "week_nb", "hour", "minute"], outputCol='features')
    output_training = assembler.transform(df_training)
    output_testing = assembler.transform(df_testing)

    final_data_training = output_training.select('features', 'demand')
    final_data_testing = output_testing.select('features', 'demand')


    final_data_training.describe().show()
    final_data_testing.describe().show()



    """  Model and predictions : """
    linearRegression = LinearRegression(labelCol='demand')
    lr_model = linearRegression.fit(final_data_training)
    predictions = lr_model.evaluate(final_data_testing)

    #print("Decision tree model max depth = %g" % decisionTree.getMaxDepth())
    #print(dt_model.toDebugString)


    """ Evaluation rmse : """
    rmse = predictions.rootMeanSquaredError
    errorsRMSE.append(rmse)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    r2 = predictions.r2
    errorsR2.append(r2)
    print("R Squared Error (R2) on test data = %g" % r2)


""" Writing the errors in the files : """
file = open("linear_regression_rmse.txt", "w")
file.write("Training set contains " + str(N_DAYS_TRAIN) + " days i.e. "+ str(N_OF_TIME_SLOTS_TRAIN) + " time slots \nTest set contains "+ str(N_DAYS_TEST)+ " days i.e. "+ str(N_OF_TIME_SLOTS_TEST) + " time slots \n")
for errorIndex in range(N_OF_CLUSTERS):
    file.write("RMSE for cluster " + str(errorIndex) + " is " + str(errorsRMSE[errorIndex]) + "\n")
file.close()

file = open("linear_regression_r2.txt", "w")
file.write("Training set contains " + str(N_DAYS_TRAIN) + " days i.e. "+ str(N_OF_TIME_SLOTS_TRAIN) + " time slots \nTest set contains "+ str(N_DAYS_TEST)+ " days i.e. "+ str(N_OF_TIME_SLOTS_TEST) + " time slots \n")
for errorIndex in range(N_OF_CLUSTERS):
    file.write("R2 for cluster " + str(errorIndex) + " is " + str(errorsR2[errorIndex]) + "\n")
file.close()