import numpy as np
from pyspark.sql import *
from schemas import *
from pyspark.ml.feature import VectorAssembler
import time
from .constants import *


def get_time_slot_per_day(hour, minute):
    nb_minute_slot = minute / 10
    time_slot_per_hour = 6
    slot_nb_day = hour * time_slot_per_hour + nb_minute_slot
    return slot_nb_day


def getSlotInfo(slot_nb, slotsTable):
    df_slot = slotsTable[slotsTable.id == slot_nb].select('from').collect()
    test = df_slot[0]
    day_of_week = test[0].weekday()
    day = test[0].day
    week = test[0].isocalendar()[1]
    hour = test[0].hour
    minute = test[0].minute
    time_of_day_code = get_time_slot_per_day(hour, minute)
    return week, day, day_of_week, time_of_day_code, hour, minute


def getFeatures() :

    """Spark initialization: """
    spark = SparkSession.builder.master('spark://csit7-master:7077').getOrCreate()
    sqlCtx = SQLContext(spark.sparkContext, spark)
    featuresTable = loadDataFrame(sqlCtx, Table.FINAL_DATA_500)
    slotsTable = loadDataFrame(sqlCtx, Table.TIME_SLOTS)
    #registerTable(sqlCtx, Table.TIME_SLOTS)  only do that if you want a sql query

    final_data_training = np.zeros(N_OF_CLUSTERS)
    final_data_testing = np.zeros(N_OF_CLUSTERS)


    for curCluster in range (N_OF_CLUSTERS):
    #for curCluster in cc: to test a specific cluster
        print('current cluster number is: ', curCluster)

        """ df is data frame containing the time slot id, demand for the current cluster"""
        df_for_one_cluster = featuresTable[featuresTable.origin == curCluster].select('month','week','day','day_of_week','time_of_day_code','hour','minute','origin','amount','pickup_timeslot_id')
        #df_for_one_cluster.show()

        print("data downloaded!")

        #demandListDict = df_for_one_cluster.collect()

        rows_training = []
        rows_testing = []
        demandCount = 0
        TOTAL_SLOTS_FOR_LOOP = N_OF_TIME_SLOTS_TEST + N_OF_TIME_SLOTS_TRAIN
        print('total time slots in loop: ', TOTAL_SLOTS_FOR_LOOP)
        start_time = time.time()
        for curSlotNb in range (TOTAL_SLOTS_FOR_LOOP) :
            if curSlotNb % 1000 == 0 :
                print('cur slot: ', curSlotNb, " time: ", time.time() - start_time)
            df_for_one_cluster_for_one_ts = df_for_one_cluster[featuresTable.pickup_timeslot_id == curSlotNb].select('week',
                                                                                                           'day',
                                                                                                           'day_of_week',
                                                                                                           'time_of_day_code',
                                                                                                           'hour', 'minute',
                                                                                                           'origin',
                                                                                                           'amount',
                                                                                                           'pickup_timeslot_id')
            df_for_one_cluster_for_one_ts_col = df_for_one_cluster_for_one_ts.collect()

            if len(df_for_one_cluster_for_one_ts_col) < 1 :
                week, day, day_of_week, time_of_day_code, hour, minute = getSlotInfo(curSlotNb, slotsTable)
                demand = 0

            else :
                """Compute the time slot information: """
                cur = df_for_one_cluster_for_one_ts_col[0]
                weekday = cur.day_of_week
                day = cur.day
                week_nb =  cur.week
                hour = cur.hour
                minute = cur.minute
                demand = cur.amount
                slot_nb_day = cur.time_of_day_code

            if (len(rows_training) < N_OF_TIME_SLOTS_TRAIN):
                rows_training.append((slot_nb_day, weekday, day, week_nb, hour, minute, demand))
            else:
                rows_testing.append((slot_nb_day, weekday, day, week_nb, hour, minute, demand))


        print('train rows len: ', len(rows_training), 'test rows len: ', len(rows_testing))

        """Create 2 dataframes corresponding to the training and testing set previously computed: """
        df_training = spark.createDataFrame(rows_training, ["slot_id", "day_of_week", "day_of_month", "week_nb", "hour", "minute", "demand"])
        df_testing = spark.createDataFrame(rows_testing, ["slot_id", "day_of_week", "day_of_month", "week_nb", "hour", "minute", "demand"])


        assembler = VectorAssembler(inputCols=["slot_id", "day_of_week", "day_of_month", "week_nb", "hour", "minute"], outputCol='features')
        output_training = assembler.transform(df_training)
        output_testing = assembler.transform(df_testing)

        final_data_training[curCluster] = output_training.select('features', 'demand')
        final_data_testing[curCluster] = output_testing.select('features', 'demand')


        final_data_training[curCluster].describe().show()
        final_data_testing[curCluster].describe().show()
    return final_data_training, final_data_testing

