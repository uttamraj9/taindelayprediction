
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql.functions import count
from pyspark.sql.functions import count, when, isnull, col
from pyspark.sql.functions import col, sum
from pyspark.sql import SparkSession
from pyspark.sql.functions import corr
from pyspark.sql.functions import corr, unix_timestamp
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, hash
from pyspark.sql.functions import udf
from pyspark.sql.functions import to_timestamp
from cryptography.fernet import Fernet
from pyspark.sql.types import StringType
from pyspark.sql.functions import from_unixtime
from pyspark.sql.functions import from_utc_timestamp
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import *
from datetime import datetime
from pyspark.sql.functions import lit

appName= "traindelay"
master= "local"

spark = SparkSession.builder \
    .master(master).appName(appName).enableHiveSupport().getOrCreate()

# Get current date and time
current_datetime = datetime.now().strftime('%Y-%m-%d')

# Convert to PySpark string format and assign to a variable
current_datetime_str = "'{}'".format(current_datetime)

# Use the variable in your PySpark code
df = spark.sql("SELECT vehicleid, currentlocation, towards, timetolive, destinationName, timestamp, expectedArrival FROM tfl.arrivalu WHERE timestamp < {}".format(current_datetime_str))

#df=spark.sql("SELECT vehicleid,currentlocation,towards,timetolive,destinationName,timestamp,expectedArrival FROM tfl.arrivalu where `timestamp` <'2023-04-18'")
#df.show(1)

# Convert timestamp and expectedArrival columns to timestamp columns
df = df.withColumn("timestamp", to_timestamp(col("timestamp"), "yyyy-MM-dd'T'HH:mm:ss"))
df = df.withColumn("expectedArrival", to_timestamp(col("expectedArrival"), "yyyy-MM-dd'T'HH:mm:ss"))

# Compute the time difference between the two columns in seconds
time_diff = unix_timestamp(col("expectedArrival")) - unix_timestamp(col("timestamp"))

# Add the time difference as a new column to the dataframe
df = df.withColumn("timedifference", time_diff)

# Convert the time difference back to a timestamp in the format HH:mm:ss
df = df.withColumn("timedifference", from_unixtime(col("timedifference"), "HH:mm:ss"))


data = df.select('vehicleid','currentlocation','towards','timetolive','destinationName','timedifference')
data.show(2)

cols = ['currentlocation','towards', 'timetolive', 'destinationName']

for col in cols:
    stringIndexer = StringIndexer(inputCol=col, outputCol=col+'_indexed')
    data = stringIndexer.fit(data).transform(data)
    data = data.drop(col).withColumnRenamed(col+'_indexed', col)

from pyspark.sql.functions import hour, minute, second, col

# Convert the timedifference column to seconds
timediff_seconds = hour(col('timedifference')) * 3600 + minute(col('timedifference')) * 60 + second(col('timedifference'))

# Cast the timedifference column to double
data = data.withColumn('timedifference', timediff_seconds.cast('double'))

#train_data, test_data = data.randomSplit([0.7, 0.3], seed=123)

from pyspark.ml.feature import VectorAssembler

# Define the input columns
inputCols = ['currentlocation', 'towards', 'timetolive', 'destinationName']

# Create a vector assembler object
assembler = VectorAssembler(inputCols=inputCols, outputCol='features')

# Use the vector assembler to transform the train data
train_data = assembler.transform(data)


# Fit the linear regression model to the train data
from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol='features', labelCol='timedifference')
lr_model = lr.fit(train_data)

summary = lr_model.summary

print("Root Mean Squared Error (RMSE):", summary.rootMeanSquaredError)



######----------------------------------------------------------------------------------------------------------###################

# Get current date and time
current_datetime = datetime.now().strftime('%Y-%m-%d')

# Convert to PySpark string format and assign to a variable
current_datetime_str = "'{}'".format(current_datetime)

# Use the variable in your PySpark code
df = spark.sql("SELECT vehicleid, currentlocation, towards, timetolive, destinationName, timestamp, expectedArrival FROM tfl.arrivalu WHERE timestamp >= {}".format(current_datetime_str))

#df=spark.sql("SELECT vehicleid,currentlocation,towards,timetolive,destinationName,timestamp,expectedArrival FROM tfl.arrivalu where `timestamp` <'2023-04-18'")
#df.show(1)

# Convert timestamp and expectedArrival columns to timestamp columns
df = df.withColumn("timestamp", to_timestamp(col("timestamp"), "yyyy-MM-dd'T'HH:mm:ss"))
df = df.withColumn("expectedArrival", to_timestamp(col("expectedArrival"), "yyyy-MM-dd'T'HH:mm:ss"))

# Compute the time difference between the two columns in seconds
time_diff = unix_timestamp(col("expectedArrival")) - unix_timestamp(col("timestamp"))

# Add the time difference as a new column to the dataframe
df = df.withColumn("timedifference", time_diff)

# Convert the time difference back to a timestamp in the format HH:mm:ss
df = df.withColumn("timedifference", from_unixtime(col("timedifference"), "HH:mm:ss"))


data = df.select('vehicleid','currentlocation','towards','timetolive','destinationName','timedifference')
data.show(2)

cols = ['currentlocation','towards', 'timetolive', 'destinationName']

for col in cols:
    stringIndexer = StringIndexer(inputCol=col, outputCol=col+'_indexed')
    data = stringIndexer.fit(data).transform(data)
    data = data.drop(col).withColumnRenamed(col+'_indexed', col)

from pyspark.sql.functions import hour, minute, second, col

# Convert the timedifference column to seconds
timediff_seconds = hour(col('timedifference')) * 3600 + minute(col('timedifference')) * 60 + second(col('timedifference'))

# Cast the timedifference column to double
test_data = data.withColumn('timedifference', timediff_seconds.cast('double'))

#train_data, test_data = data.randomSplit([0.7, 0.3], seed=123)

from pyspark.ml.feature import VectorAssembler

# Define the input columns
inputCols = ['currentlocation', 'towards', 'timetolive', 'destinationName']
# Apply feature transformers on the test data
test_data = assembler.transform(test_data)

# Make predictions on the test data
predictions = lr_model.transform(test_data)

# Select columns to display from the predictions
predictions.select("timedifference", "prediction").show()

pr_data=df.select('vehicleid','currentlocation','towards','destinationName','timestamp')

# Join the two dataframes on the vehicleid column
pr_data_with_predictions = pr_data.join(predictions.select('vehicleid', 'prediction'),
                                        on='vehicleid',
                                        how='left_outer')

pr_data_with_predictions = pr_data_with_predictions.withColumnRenamed("prediction", "delay")

# Save the dataframe as a table in Hive
pr_data_with_predictions.write.mode('overwrite').saveAsTable('tfl.dailydelayprediction')
