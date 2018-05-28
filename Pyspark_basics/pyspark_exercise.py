import findspark
findspark.init('/media/light/UbuntuDrive/work/spark-2.1.1-bin-hadoop2.6/')

from pyspark.sql import SparkSession
from pyspark.sql.functions import format_number, mean, max, min, count, year, month

spark = SparkSession.builder.master(
    'spark://Light-Ubuntu:7077').appName('Exercise').getOrCreate()

# read walmart csv

walmartdf = spark.read.csv('/media/light/208C3A738C3A4396/Accesories/Udemy - Spark and Python for Big Data with PySpark/01 Introduction to Course/attached_files/002 Course Overview/Python-and-Spark-for-Big-Data-master/Spark_DataFrame_Project_Exercise/walmart_stock.csv', inferSchema=True, header=True)

walmartdf.show()

walmartdf.columns

walmartdf.printSchema()
walmartdf.head(5)
walmartdf.describe().show()


result_desc = walmartdf.describe()

result_desc.printSchema()

result_desc.select(result_desc['summary'],
                   format_number(result_desc['Open'].cast(
                       'float'), 2).alias('Open'),
                   format_number(result_desc['High'].cast(
                       'float'), 2).alias('High'),
                   format_number(result_desc['Low'].cast(
                       'float'), 2).alias('Low'),
                   format_number(result_desc['Close'].cast(
                       'float'), 2).alias('Close'),
                   result_desc['Volume'].cast('int').alias('Volume')
                   ).show()

hv_ratio = walmartdf.withColumn(
    "HV ratio", walmartdf['High'] / walmartdf['Volume'])

hv_ratio.select('HV Ratio').show()

# finding highest value date
walmartdf.orderBy(walmartdf['High'].desc()).head(1)[0][0]


walmartdf.agg(mean(walmartdf['Close'])).show()

walmartdf.select(max(walmartdf['Volume']), min('Volume')).show()

walmartdf.filter(walmartdf['Close'] < 60).count()

(walmartdf.filter(walmartdf['High'] > 80).count(
) / walmartdf.agg(count(walmartdf['Date'])).head(1)[0][0]) * 100

newdf = walmartdf.withColumn("year", year(walmartdf['Date']))

newdf.groupby("year").max().select('year', 'max(High)').show()

newdf2 = walmartdf.withColumn("month", month(walmartdf['Date']))

newdf2.groupBy("month").mean().select(
    'month', 'avg(Close)').orderBy('month').show()


spark.stop()
