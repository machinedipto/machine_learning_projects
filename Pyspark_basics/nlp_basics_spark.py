import findspark
findspark.init('/media/light/UbuntuDrive/work/spark-2.1.1-bin-hadoop2.6/')

from pyspark.sql import SparkSession
from pyspark.sql.functions import length
from pyspark.ml.feature import VectorAssembler, CountVectorizer, Tokenizer, IDF, StopWordsRemover, StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.master('local').appName('nlp').getOrCreate()

data = spark.read.csv('/media/light/208C3A738C3A4396/Accesories/Udemy - Spark and Python for Big Data with PySpark/01 Introduction to Course/attached_files/002 Course Overview/Python-and-Spark-for-Big-Data-master/Spark_for_Machine_Learning/Natural_Language_Processing/smsspamcollection/SMSSpamCollection', inferSchema=True, sep='\t')

data.show()
data = data.withColumnRenamed('_c0', 'class').withColumnRenamed('_c1', 'text')

data.show()

data = data.withColumn('length', length(data['text']))

data.show()

tokenizer = Tokenizer(inputCol='text', outputCol='tokens')
stop_remove = StopWordsRemover(inputCol='tokens', outputCol='stop_tokens')
count_vec = CountVectorizer(inputCol='stop_tokens', outputCol='count_vec')
idf = IDF(inputCol='count_vec', outputCol='tf_idf')

label_index = StringIndexer(inputCol='class', outputCol='label')

clean_up = VectorAssembler(
    inputCols=['tf_idf', 'length'], outputCol='features')

nb = NaiveBayes()

data_pipe = Pipeline(
    stages=[label_index, tokenizer, stop_remove, count_vec, idf, clean_up])

cleaned = data_pipe.fit(data).transform(data)

cleaned.show()

final_data = cleaned.select(['features', 'label'])

train, test = final_data.randomSplit([0.7, 0.3])

train.show()

spam_fit = nb.fit(train)

test_results = spam_fit.transform(test)

test_results.show()

acc_eval = MulticlassClassificationEvaluator()

acc = acc_eval.evaluate(test_results)

acc

spark.stop()
