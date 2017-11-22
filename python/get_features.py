from schemas import *

def get_features_for_cluster(sqlCtx, cluster, split_train_test=True):
    df = loadDataFrame(sqlCtx, Table.FINAL_DATA).filter('origin = {}'.format(cluster)).drop('origin')
    if split_train_test:
        train = df.filter('month < 6')
        test = df.filter('month = 6')

        return (train.drop('amount').collect(), [row[0] for row in train.select('amount').collect()]), \
               (test.drop('amount').collect(), [row[0] for row in test.select('amount').collect()])

    return df.drop('amount').collect(), [row[0] for row in df.select('amount').collect()]
