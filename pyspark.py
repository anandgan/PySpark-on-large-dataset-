# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, when, avg
from pyspark.sql.functions import unix_timestamp, col
from pyspark.sql.functions import corr


# COMMAND ----------

display(dbutils.fs.ls("/FileStore/tables/"))



# COMMAND ----------

file_path = "dbfs:/FileStore/tables/yellow_tripdata_2016_03-1.csv"

df = spark.read.csv(file_path, header=True, inferSchema=True)

df.show(5)


# COMMAND ----------


df.printSchema()


# COMMAND ----------


df = df.withColumn(
    "trip_duration_minutes",
    (unix_timestamp("tpep_dropoff_datetime") - unix_timestamp("tpep_pickup_datetime")) / 60
)
df = df.filter((col("trip_distance") > 0) & (col("total_amount") > 0))
df.select("trip_duration_minutes", "trip_distance", "total_amount").show(5)


# COMMAND ----------

df = df.filter(col("trip_duration_minutes") > 0)
df.select("trip_duration_minutes", "trip_distance", "total_amount").show(5)


# COMMAND ----------

df.explain()


# COMMAND ----------

#data filtering
from pyspark.sql.functions import col
df = df.filter((col("trip_distance") > 0) & (col("trip_distance") < 50))
df = df.filter((col("total_amount") > 2) & (col("total_amount") < 300))
df = df.filter((col("trip_duration_minutes") > 1) & (col("trip_duration_minutes") < 300))
df.select("trip_distance", "trip_duration_minutes", "total_amount").show(5)


# COMMAND ----------

from pyspark.sql.functions import corr
distance_amount_corr = df.stat.corr("trip_distance", "total_amount")
duration_amount_corr = df.stat.corr("trip_duration_minutes", "total_amount")

print(f"Correlation between trip_distance and total_amount: {distance_amount_corr}")
print(f"Correlation between trip_duration_minutes and total_amount: {duration_amount_corr}")


# COMMAND ----------

import matplotlib.pyplot as plt
data = df.select("trip_distance", "total_amount", "trip_duration_minutes").sample(fraction=0.01).toPandas()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(data["trip_distance"], data["total_amount"], alpha=0.5)
axes[0].set_title("Trip Distance vs Total Amount")
axes[0].set_xlabel("Trip Distance")
axes[0].set_ylabel("Total Amount")
axes[1].scatter(data["trip_duration_minutes"], data["total_amount"], alpha=0.5)
axes[1].set_title("Trip Duration vs Total Amount")
axes[1].set_xlabel("Trip Duration (minutes)")
axes[1].set_ylabel("Total Amount")

plt.tight_layout()
plt.show()


# COMMAND ----------

lazy_filtered_data = df.filter(
    (col("trip_distance") > 10) & (col("total_amount") > 0)
).withColumn(
    "fare_category",
    when(col("total_amount") < 50, "Low Fare")
    .when(col("total_amount") < 100, "Medium Fare")
    .otherwise("High Fare")
)
result = lazy_filtered_data.groupBy("fare_category").agg(
    avg("trip_duration_minutes").alias("avg_duration"),
    avg("total_amount").alias("avg_fare")
)

# COMMAND ----------

result.show()

# COMMAND ----------

from pyspark.sql.functions import col
repartitioned_df = df.repartition(4)
num_partitions = repartitioned_df.rdd.getNumPartitions()
print(f"Number of partitions after repartitioning: {num_partitions}")


# COMMAND ----------

df.createOrReplaceTempView("taxi_data")
sql_query = """
SELECT 
    fare_category, 
    AVG(trip_duration_minutes) AS avg_duration, 
    AVG(total_amount) AS avg_fare
FROM (
    SELECT 
        trip_distance, 
        total_amount, 
        trip_duration_minutes, 
        CASE 
            WHEN total_amount < 50 THEN 'Low Fare'
            WHEN total_amount < 100 THEN 'Medium Fare'
            ELSE 'High Fare'
        END AS fare_category
    FROM taxi_data
    WHERE trip_distance > 10 AND total_amount > 0
) GROUP BY fare_category
"""
sql_result = spark.sql(sql_query)
sql_result.show()


# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# Feature Engineering: Combine features into a single vector
assembler = VectorAssembler(
    inputCols=["trip_distance", "trip_duration_minutes"],
    outputCol="features"
)
gbt = GBTRegressor(featuresCol="features", labelCol="total_amount", maxIter=10)

# Pipeline: Feature Transformation and Model Training , ensures that all steps are executed in sequence
pipeline = Pipeline(stages=[assembler, gbt])

train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
pipeline_model = pipeline.fit(train_data)

# Predictions on test data
predictions = pipeline_model.transform(test_data) # predict the total amount for test data

# Evaluate the model
evaluator_rmse = RegressionEvaluator(
    labelCol="total_amount", predictionCol="prediction", metricName="rmse"
)
evaluator_r2 = RegressionEvaluator(
    labelCol="total_amount", predictionCol="prediction", metricName="r2"
)

rmse = evaluator_rmse.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-Squared: {r2}")

# Display sample predictions
predictions.select("features", "total_amount", "prediction").show(5)


# COMMAND ----------

print("Pipeline and transformations are defined but not yet executed.")
pipeline_model = pipeline.fit(train_data)
predictions = pipeline_model.transform(test_data)


# COMMAND ----------

from pyspark.sql import Row

# Create a new DataFrame with the feature values for prediction
new_data = spark.createDataFrame([
    Row(trip_distance=15.0, trip_duration_minutes=30.0)  # Replace with your input values
])

# Transform the input data to include the feature vector
new_data_with_features = assembler.transform(new_data)

# Use the trained pipeline model to make predictions
predicted_value = pipeline_model.transform(new_data_with_features)

# Display the prediction
predicted_value.select("trip_distance", "trip_duration_minutes", "prediction").show()

