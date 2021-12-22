# Classification on 10k-Dimension TF matrix

## Description

Train a Logistic Regression and SVM classifiers to classify documents based on a subset of the 20k-dimension Term Frequency (TF) matrix. Compare run time and performance with the 20k models. 


## Scripts

#### logistic_regression_with_MLlib_(reduced_TF_matrix).py: 
* Generate the 20k-dimension TF matrix
* Select a 10k subset based on feature variance 
* Use 'LogisticRegression' module from the spark ML library to train a Support Vector Machine classifier
* Test and evaluate the model


#### svm_with_MLlib_(reduced_TF_matrix).py: 
* Generate the 20k-dimension TF matrix
* * Select a 10k subset based on feature variance 
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
