# Collaborative Filtering
This project aims to develop a recommender system using Apache Spark to predict user ratings. 
The project compares two collaborative filtering methods: item-item or user-user CF using LSH to find the similarity between items (or users) and CF using latent factors (matrix decomposition). 
The root-mean-squared-error (RMSE) score will be used to test the submissions, the training time, and the time it takes to predict ratings on the test data for both implementations. 
The dataset, inside the folder [Data_for_PythonCode_using_PySpark](Data_for_PythonCode_using_PySpark) consists of 85,724 ratings for the training set and 2,154 for the testing set.

### Overview and Project Goals
The objectives of this project are as follows:

1. Use Apache Spark to build a Recommender system and predict user ratings
2. Compare CF using LSH with a matrix decomposition method

### Detailed Description
The project aims to develop a recommendation system that predicts user-item ratings as accurately as possible. Collaborative Filtering (CF) systems measure the similarity of users by their item preferences and the similarity of items by the users who like them. For these CF systems, we extract item and user profiles and then compute the similarity of rows and columns in the Utility Matrix.

### Deliverables
Leader Board website: http://coe-clp.sjsu.edu/
The report for this project can be found in [Report_Collaborative_Filtering.pdf](Report_Collaborative_Filtering.pdf)
The working code is available in the folder [Python_Code](Python_Code) along with the [Model Evaluation](Model_Evaluations) result.

### There are two tracks:
1. The LSH track
2. The Latent factors track

### Below are the main tasks for this project:

* Implement item-item or user-user CF using LSH to find the similarity between items (or users)
* Implement CF using latent factors (matrix decomposition)
* Compare the RMSE score on the test data for the two implementations
* Report the training time for both implementations (LSH index building time for LSH)
* Report the time it takes to predict ratings on the test data for both methods
* Use various similarity measures to find the most similar items or users.

# Additional

This repository also consists of Hands-On Experience in Creating Recommender Systems during the academic courses in the folder [Hands_On_Experience_Recommendation_Systems](Hands_On_Experience_Recommendation_Systems)