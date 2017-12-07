from schemas import *

def get_features_for_cluster(sqlCtx, cluster, split_train_test=True):
    df = loadDataFrame(sqlCtx, Table.FINAL_DATA).filter('origin = {}'.format(cluster)).orderBy('origin')\
        .drop('origin', 'pickup_timeslot_id')
    if split_train_test:
        train = df.filter('week < 23')
        test = df.filter('week >= 23')

        return (train.drop('amount').collect(), [row[0] for row in train.select('amount').collect()]), \
               (test.drop('amount').collect(), [row[0] for row in test.select('amount').collect()])

    return df.drop('amount').collect(), [row[0] for row in df.select('amount').collect()]

def get_all_data(spark, sqlCtx):
    registerTable(sqlCtx, Table.FINAL_FEATURES)
    clusters = [row[0] for row in spark.sql('SELECT DISTINCT origin FROM final_features').collect()]

    first_week_of_june = 23

    def get(query, cid):
        return spark.sql(query.format(cid, first_week_of_june)).collect()

    return spark.sparkContext.parallelize([(
        get("SELECT time_of_day_code, day_of_week, day, week, hour, minute, " + \
                  "is_manhattan, is_airport FROM final_features WHERE origin = {} AND week < {}", cid),
        get("SELECT amount FROM final_features WHERE origin = {} AND week < {}", cid),
        get("SELECT time_of_day_code, day_of_week, day, week, hour, minute, " + \
                  "is_manhattan, is_airport FROM final_features WHERE origin = {} AND week >= {}", cid),
        get("SELECT amount FROM final_features WHERE origin = {} AND week >= {}", cid)
    ) for cid in clusters], len(clusters))
