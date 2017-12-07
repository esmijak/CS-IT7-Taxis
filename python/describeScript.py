from schemas import *
from pyspark.sql import *

N_CLUSTER = 358

spark = SparkSession.builder.master('spark://csit7-master:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)
df = loadDataFrame(sqlCtx, Table.FINAL_FEATURES)
registerTable(sqlCtx, Table.FINAL_FEATURES)

for cluster in range(N_CLUSTER) :
    print('cluster number : ', cluster)
    df_cur_clust = spark.sql('SELECT * FROM final_features WHERE origin = {}'.format(cluster))
    df_cur_clust.describe('amount').show()