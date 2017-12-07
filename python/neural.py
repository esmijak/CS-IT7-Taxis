from pyspark.sql import *
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from schemas import *
from get_features import *
from math import sqrt
from numpy import mean, std

spark = SparkSession.builder.master('spark://csit7-master:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)

registerTable(sqlCtx, Table.DEMAND)
registerTable(sqlCtx, Table.EX_TIME_SLOTS)

def neuralNet(cluster, full_set_fraction = 1.0, test_fraction = 0.1667, layers = (10,10)):
    data = spark.sql('SELECT month, day_of_week, time_of_day_code AS time_of_day, pickup_cid AS origin_location, cnt AS demand, id'
                     ' FROM demand INNER JOIN extended_timeslots ON (pickup_timeslot_id = id)'
                     ' WHERE pickup_cid = {}'
                     ' ORDER BY id'.format(cluster)).collect()

    if (len(data) < 10):
        return None

    test_size = int(test_fraction * len(data))
    train_set = data[test_size:]
    test_set = data[:test_size]

    network = MLPRegressor(hidden_layer_sizes=layers, max_iter=1000)

    scaler = StandardScaler()

    train_X = [[row['month'], row['day_of_week'], row['time_of_day'], row['origin_location']] for row in train_set]
    train_y = [row['demand'] for row in train_set]
    test_X = [[row['month'], row['day_of_week'], row['time_of_day'], row['origin_location']] for row in test_set]
    test_y = [row['demand'] for row in test_set]

    #train_X = scaler.fit_transform(train_X)
    #test_X = scaler.transform(test_X)

    network.fit(train_X, train_y)
    rSquared = network.score(test_X, test_y)
    predictions = network.predict(test_X)
    error = 0.0
    for i in range(0, len(test_set) - 1):
        error += (predictions[i] - test_y[i]) ** 2
    error /= len(test_set)
    rmse = sqrt(error)
    print(cluster)
    return rSquared, rmse

def newNeuralNet(train_features, train_labels, test_features, test_labels, layers=(10, 10), max_iter=1000, **kwargs):
    if (len(train_features) < 5 or len(test_features) < 3): return None
    network = MLPRegressor(hidden_layer_sizes=layers, max_iter=max_iter, **kwargs)
    network.fit(train_features, train_labels)
    rSquared = network.score(test_features, test_labels)
    predictions = network.predict(test_features)
    error = 0.0
    for i in range(0, len(test_features) - 1):
        error += (predictions[i] - test_labels[i]) ** 2
    error /= len(test_features)
    rmse = sqrt(error)
    return rSquared, rmse

def rddBasedNeuralNet(feature_tuple, **kwargs):
    train_features, train_labels, test_features, test_labels = feature_tuple
    network = MLPRegressor(**kwargs)
    network.fit(train_features, train_labels)
    rSquared = network.score(test_features, test_labels)
    predictions = network.predict(test_features)
    error = 0.0
    for i in range(0, len(test_features) - 1):
        error += (predictions[i] - test_labels[i]) ** 2
    error /= len(test_features)
    rmse = sqrt(error)
    return rSquared, rmse

#data = get_all_data(spark, sqlCtx)
#mapped = data.map(lambda tup: rddBasedNeuralNet(tup, layers=(5,5), max_iter=1000))
#print(mapped.count())

registerTable(sqlCtx, Table.FINAL_FEATURES)
clusters = [row[0] for row in spark.sql("SELECT DISTINCT origin FROM final_features").collect()]
results = {}
r2 = []
rmse = []
for cluster in clusters:
    (tr_f, tr_l), (te_f, te_l) = get_features_for_cluster(sqlCtx, cluster)
    res = newNeuralNet(tr_f, tr_l, te_f, te_l, layers=(10, 10), max_iter=1000)
    if res is not None:
        _r2, _rmse = res
        results[cluster] = (_r2, _rmse)
        r2.append(_r2)
        rmse.append(_rmse)

print(results)

for i in range(0, len(r2) - 1):
    print('Cluster #{}: R^2 = {}, RMSE = {}'.format(i, r2[i], rmse[i]))
print('R^2: Mean = {}, Std. Dev = {}'.format(mean(r2), std(r2)))
print('RMSE: Mean = {}, Std. Dev = {}'.format(mean(rmse), std(rmse)))
