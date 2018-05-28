import findspark
findspark.init('/media/light/UbuntuDrive/work/spark-2.3.0-bin-hadoop2.7')

from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.clustering import KMeans

spark = SparkSession.builder.master(
    'local').appName('clustering').getOrCreate()

data = spark.read.csv('/home/light/Downloads/hack_data.csv',
                      inferSchema=True, header=True)

data.show()
data.printSchema()
data.columns

data = data.drop('Location')

columns = data.columns

assembler = VectorAssembler(inputCols=columns, outputCol='unscaled_features')

output = assembler.transform(data)

output.printSchema()

scaler = StandardScaler(inputCol='unscaled_features', outputCol='features')

scaled = scaler.fit(output).transform(output)

final_data = scaled.select('features')

KMeans

model2 = KMeans(featuresCol='features', k=2)
model3 = KMeans(featuresCol='features', k=3)

modelfit2 = model2.fit(final_data)
modelfit3 = model3.fit(final_data)

cluster_final_data2 = modelfit2.transform(final_data)

cluster_final_data2.groupby('prediction').count().show()

cluster_final_data3 = modelfit3.transform(final_data)
cluster_final_data3.groupby('prediction').count().show()

# as we can see in k =2 they are evenly splitted so we can confirm that there were 2 hackers

spark.stop()
