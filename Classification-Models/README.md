# Classifications on 20k-Dimension TF matrix

## Description

Train a Logistic Regression and SVM classifiers to classify documents based on the Term Frequency (TF). Test and compare the models performance. 


## Scripts

#### logistic_regression_with_gradient_descent.py: 
* Generate the 20k-dimension TF matrix for both the training and testing set
* Train a Logistic Regression model using Gradient Descent algorithm
* Test and evaluate the model

#### logistic_regression_with_MLlib.py: 
* Generate the 20k-dimension TF matrix for both the training and testing set
* Train a Logistic Regression model using Gradient Descent algorithm
* Test and evaluate the model

#### logistic_regression_with_MLlib.py: 
* Generate the 20k-dimension TF matrix for both the training and testing set
* Use 'LinearSVC' module from the spark ML library to train a Support Vector Machine classifier
* Test and evaluate the model


## Obtaining the Dataset

* Small Dataset (~37 MB of text): see 'Data' folder, can be run in a local machine for model testing

* Training Dataset (~1.9 GB of text): see link in the 'Data' folder

* Testing Dataset (~200 MB of text): see link in the 'Data' folder


## Instructions to run

Make sure that you have Download and configured Apache Spark on your machine. 

### Running locally on an IDE (e.g Pycharm) - Not recommended
Download the dataset to your local disk. Clone the script and paste it to your IDE, substitute all arugument fields (sys.argv[?]) with corresponding path.

### Running on a Cloud Service (e.g Google Cloud or Amzon AWS)
Upload the script to your Cloud Drive. When submitting a job, supply the internal path of the script, pass in the corresponding interal URLs for the required arguments. 
