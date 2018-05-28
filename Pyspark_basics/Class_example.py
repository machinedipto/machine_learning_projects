import findspark
findspark.init('/media/light/UbuntuDrive/work/spark-2.3.0-bin-hadoop2.7')
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

spark = SparkSession.builder.master('local').appName('rf_class').getOrCreate()


data = spark.read.csv('/home/light/Downloads/College.csv',
                      inferSchema=True, header=True)

data.show()

data.describe().show()

data.columns

data.dtypes

data.groupby('Private').count().show()

num_columns = [item[0] for item in data.dtypes if (
    item[1] == 'double' or item[1] == 'int')]

num_columns

indexer = StringIndexer(inputCol='Private', outputCol='Privateindexed')
indexed = indexer.fit(data).transform(data)

assembler = VectorAssembler(inputCols=num_columns, outputCol='features')

output = assembler.transform(indexed)
output.show()

final_data = output.select(['features', 'Privateindexed'])

final_data = final_data.withColumnRenamed('Privateindexed', 'label')

traindata, testdata = final_data.randomSplit([0.7, 0.3])

rf = RandomForestClassifier()

rf.params

param_grid = ParamGridBuilder().addGrid(rf.maxDepth, [1, 2, 3, 4, 5, 6]).addGrid(
    rf.minInstancesPerNode, [1, 3, 5, 10, 50, 100]).addGrid(rf.numTrees, [100, 200, 300, 500]).build()

crossval = CrossValidator(estimator=rf, estimatorParamMaps=param_grid,
                          evaluator=BinaryClassificationEvaluator(), numFolds=3)

cvModel = crossval.fit(traindata)

cvModel.bestModel._java_obj.getMinInstancesPerNode()

prediction = cvModel.transform(testdata)

prediction.show()

acc = MulticlassClassificationEvaluator()

acc_eval = acc.evaluate(prediction)

acc_eval

spark.stop()
