''' Rating Prediction using LSH '''
# Importing libraries
import time
from pyspark.sql import Row
from pyspark.sql.functions import monotonically_increasing_id 
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, BucketedRandomProjectionLSH
from pyspark.sql.functions import when
import pyspark.sql.functions as F


spark = SparkSession.builder.appName('LSH').getOrCreate()

# Loading the training dataset
lines = spark.read.text('train.dat').rdd
parts = lines.map(lambda row: row.value.split("\t"))
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), itemId=int(p[1]), rating=int(p[2])))

ratings = spark.createDataFrame(ratingsRDD)
ratings = ratings.select("*").withColumn("id", monotonically_increasing_id())
(training, test) = ratings.randomSplit([0.8, 0.2])
# ratings.count() # 85724

# Loading the Test dataset
lines = spark.read.text('test.dat').rdd
parts = lines.map(lambda row: row.value.split("\t"))
testRatingsRDD = parts.map(lambda p: Row(userId=int(p[0]), itemId=int(p[1])))

testRatings = spark.createDataFrame(testRatingsRDD)
testRatings = testRatings.select("*").withColumn("id", monotonically_increasing_id())
# testRatings.count() # 2154

# Data Cleaning

# item_replacement_for_user819 contains a single row, chosen at random from the 90% of ratings data for user "819".
item_replacement_for_user819 = ratings.filter(ratings.userId == "819").sample(False, 0.9).limit(1)
# Extracting the value of the itemId column from item_replacement_for_user819 DataFrame, and assigning it to a new variable called iId_819.
iId_819 = item_replacement_for_user819.first()['itemId']

# Repeating the extraction process to extract two ItemId's for user "218"
item_replacement_for_user218 = ratings.filter(ratings.userId == "218").sample(False, 0.9).limit(2)
iId_218_1 = item_replacement_for_user218.first()['itemId']
iId_218_2 = item_replacement_for_user218.select('itemId').collect()[1][0]

# Replacing the found Item Id above so we have valid items in our dataset
testRatings1 = testRatings.withColumn("itemId", when(testRatings.itemId == "375",iId_819)
                                 .when(testRatings.itemId == "35",iId_218_1)
                                 .when(testRatings.itemId == "30" ,iId_218_2)
                                 .otherwise(testRatings.itemId))

# Matched the user and item so there are no missing information for our matrix
missing_user_inTraining = testRatings1.select('userId').subtract(ratings.select('userId')).count()
missing_item_inTraining = testRatings1.select('itemId').subtract(ratings.select('itemId')).count()
missing_user_inTraining, missing_item_inTraining

# Preparing the Data

# Using Vector Assembler
# VectorAssembler is a feature transformer in Apache Spark that combines multiple columns of data into a single vector column. 
# It is commonly used for preparing data for machine learning models that require input features in vector format.
def vector_assembler(dataframe, indep_cols):    
    assembler = VectorAssembler(inputCols = indep_cols,
                                outputCol = 'features')
    output = assembler.transform(dataframe)    
    return output

df = vector_assembler(ratings, indep_cols = ratings.drop('rating', 'id').columns)
testdf = vector_assembler(testRatings1, indep_cols = testRatings1.drop('id').columns)

start_train_time = time.time()

# Training the LSH model

# BucketedRandomProjectionLSH is a technique for finding similar items in a large dataset based on their feature vectors. 
# It works by projecting the vectors onto a lower-dimensional space and then dividing the space into buckets. 
# Items with similar projections are placed in the same bucket, allowing for efficient similarity search. 
brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", bucketLength=5.0, numHashTables=8.)
model = brp.fit(df)
end_train_time = time.time()

start_neighbor_time = time.time()
# Getting the neighbors in a dataframe
similar = model.approxSimilarityJoin(df, testdf, 1.5, distCol='EuclideanDistance').select(
    F.col("datasetA.rating").alias("ratings"),
    # F.col("datasetA.features").alias("user_itemA"),
    F.col("datasetB.id").alias("id"),
    F.col("datasetB.userId").alias("userId"),
    F.col("datasetB.itemId").alias("itemId"),
    F.col("EuclideanDistance")
).orderBy(F.col('EuclideanDistance'))
end_neighbor_time = time.time()

# Converting the neighbor dataframe and test dataframe to Pandas
test_set_df = testRatings1.toPandas()
similar_set_df = similar.toPandas()

# Function to predict rating
def predict_ratings(df):
  rating_file=open('lsh_ratings.dat', 'w')

  def get_rating(u, i):
    k = similar_set_df.loc[(similar_set_df['userId'] == u) & (similar_set_df['itemId'] == i)]["ratings"] 
    k = k.tolist()[0] if bool(k.tolist()) else 0
    if k == 0: 
      k = similar_set_df.loc[(similar_set_df['userId'] == u)]["ratings"]
      k = k.tolist()[0] if bool(k.tolist()) else 0
    return k if k else 1

  for ind in df.index:
      userId = df['userId'][ind]
      itemId = df['itemId'][ind]
      rating = get_rating(userId, itemId)
      rating_file.write(str(rating))
      if ind < len(df.index) - 1:
        rating_file.write('\n')
  
  rating_file.close()

# Predicting the ratings
start_pred_time = time.time()
predict_ratings(test_set_df)
end_pred_time = time.time()

# Evaluating our model
evaluation_file=open('lsh_rating_evaluation.txt', 'w')
evaluation_file.write(f'Time taken for training the model : {end_train_time - start_train_time} seconds\n')
evaluation_file.write(f'Time taken to find the closest neighbors : {end_neighbor_time - start_neighbor_time} seconds\n')
evaluation_file.write(f'Time taken for making predictions : {end_pred_time - start_pred_time} seconds\n')

spark.stop()