"""
Description: This script contains a PySpark program that implement
             Gradient Descent algorithm to find the optimal coefficients
             for a Logistic Regression classification model. 
"""

from __future__ import print_function
import re
import sys
import numpy as np
from operator import add
import psutil
from pyspark import SparkContext


# ========================== User-Defined Functions  ===========================


# Build an array of size 20,000 with each position tells how many occurance of 
# the word in that position of the top-20-frequent-words dictionary
def buildArray(listOfIndices):

    returnVal = np.zeros(20000)
    
    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1
    
    mysum = np.sum(returnVal)
    
    returnVal = np.divide(returnVal, mysum)
    
    return returnVal


# Check if the predicted label is a TP, FN, FP or TN
# Returns an array of len(4) containing binary values indicating the notation
# Ex: [0,1,0,0] means the prediction is a False Negative (FN)
def predictionNotation(true, pred):
    TP, FN, FP, TN = 0, 0, 0, 0
    
    if true == 1:
        if pred == 1: 
            TP = 1
        else:
            FN = 1
    elif true == 0:
        if pred == 1:
            FP = 1
        else:
            TN = 1
    
    return np.array([TP, FN, FP, TN])

# calcuate the f1-score
def f_score(TP, FP, P):
    # f1 = NA if there is no predicted positive
    if TP == 0:
        f_score = 'NaN (zero TP)'
    else:
        recall = TP/P
        precision = TP/(TP+FP)
        f_score = 2*precision*recall / (precision+recall)
    
    return f_score

# ============================= End of Functions  =============================


if __name__ == "__main__":

    # System Arguments: 
    # [1]: Training Data
    # [2]: Testing Data
    # [3]: Learning rate
    # [4]: Number of iterations
    # [5]: Regularization factor
    # [6]: Path for the output file

    if len(sys.argv) != 7:
        print("Usage: wordcount <file> <output> ", file=sys.stderr)
        exit(-1)

    sc = SparkContext.getOrCreate()

    # ============================= Data Preprocessing  =============================

    # use a regular expression here to check for non-letter characters
    regex = re.compile('[^a-zA-Z]')

    '''---- Training Set ----'''
    # Load file into an RDD
    d_corpus = sc.textFile(sys.argv[1], 1)

    # Transform both into a set of (docID, text) pairs
    d_keyAndText = d_corpus\
                .map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))


    # remove all non letter characters
    # Split the text in each (docID, text) pair into a list of words
    # Resulting RDD is a dataset with (docID, ["word1", "word2", "word3", ...])
    d_keyAndListOfWords = d_keyAndText\
                .map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))


    '''---- Testing Set ----'''
    # Load file into an RDD
    test_corpus = sc.textFile(sys.argv[2], 1)

    # Transform it into a set of (docID, text) pairs
    test_keyAndText = test_corpus\
                .map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))

    # remove all non letter characters
    # Split the text in each (docID, text) pair into a list of words
    # Resulting RDD is a dataset with (docID, ["word1", "word2", "word3", ...])
    test_keyAndListOfWords = test_keyAndText\
                .map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

    # ========================== End of Data Preprocessing  ==========================

    
    # ============= Data Preparation =============

    # - STEP 1: Build Top-20 Word Dictionary from the training set
    # map (docID, ["word1", "word2", "word3", ...])
    # to ("word1", 1) ("word2", 1)...
    allWords = d_keyAndListOfWords\
                .map(lambda x: (x[1]))\
                .flatMap(lambda x: [w for w in x])\
                .map(lambda x: (x, 1))

    # count all of the words, giving --> ("word1", 1433), ("word2", 3423423), etc.
    allCounts = allWords.reduceByKey(lambda x, y: x+y)

    # Get the top 20,000 words in a local array in a sorted format based on frequency
    topWords = allCounts.top(20000, key=lambda x: x[1])
    topWords = np.array(topWords)

    # Create a RDD that has a set of (word, dictNum) pairs
    # start by creating an RDD that has the number 0 through 19999
    # 20000 is the number of words that will be in our dictionary
    topWordsK = sc.parallelize(range(20000))

    # Then, transform (0), (1), (2), ... to ("MostCommonWord", 1) ("NextMostCommon", 2), ...
    # the number will be the spot in the dictionary used to tell us where the word is located
    dictionary = topWordsK.map(lambda x: (topWords[x][0], x))

    # Will be using Map-Side Join Operation
    # Collect the small RDD as Map (a dict in python)
    dictionaryAsMap = dictionary.collectAsMap()

    # broad cast this to all worker nodes. 
    sc.broadcast(dictionaryAsMap)


    # - STEP 2: Build TF matrix for the Training Set 

    # Create a RDD that has, for each (docID, ["word1", "word2", "word3", ...]),
    # ("word1", docID), ("word2", docId), ...
    allWordsWithDocID = d_keyAndListOfWords\
                        .flatMap(lambda x: ((j, x[0]) for j in x[1]))


    # Then do a simple map on it to get a set of (word, (dictionaryPos, docID)) pairs
    allDictionaryWords = allWordsWithDocID\
                            .map(lambda x: (x[0], (x[1], dictionaryAsMap.get(x[0]))) 
                             if x[0] in dictionaryAsMap.keys() else None)\
                            .filter(lambda x: x!=None)\
                            .map(lambda x: (x[0], (x[1][1], x[1][0])))

    # Drop the actual word itself to get a set of (docID, dictionaryPos) pairs
    justDocAndPos = allDictionaryWords.map(lambda x: (x[1][1], x[1][0]))

    # Create a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
    allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()

    # Converts the dictionary positions to a bag-of-words numpy array...
    # use the buildArray function to build the feature array
    # this gives a set of (docID, featureArray)
    allDocsAsNumpyArrays = allDictionaryWordsInEachDoc\
                            .map(lambda x: (x[0], buildArray(x[1])))



    # ============= Learning the Model =============

    # Use Gradient Descent to learn Logistic Regression Model

    # Create an RDD that has (label, TFArray) pairs for all documents
    # labeling: 1 if docID starts with 'AU' (Australia court case) and 0 if others
    docLabelAndTFArray = allDocsAsNumpyArrays\
                        .map(lambda x: (1, x[1]) if x[0][:2] == 'AU' else (0, x[1]))
    # cache this RDD
    docLabelAndTFArray.cache()

    # Variables initiation
    learningRate = float(sys.argv[3])
    num_iteration = int(sys.argv[4])
    precision = 0.1
    reg_factor = float(sys.argv[5])  # regularization factor
    r = np.zeros(20000)              # set initial coefficients to 0

    oldCost = float('inf')  # set the initial cost to a very big value
    costHist = []           # Emplty list to store cost history
    iterHist = []


    '''Main iterative part of Gradient Descent Algorithm '''
    for i in range(num_iteration):
    
        # Calculate Gradient
        # y: label; x: TFArray
        # 1st Map --> produces RDD of set (-y, x, theta) for all docs
        # 2nd Map --> produces RDD of set (-y*theta, -y, x, exp(theta))
        # 3rd Map --> produces RDD of set (-y*theta, -y, x, log(1+exp(theta)), (exp(theta)/(1+exp(theta))))
        # 4th Map --> produces RDD of set (gradiant, cost) for each doc before regularization
        # Reduce --> summing up gieves (gradient, cost) before regularization
        gradientCost = docLabelAndTFArray\
                        .map(lambda x: (-x[0], x[1], np.dot(x[1], r)))\
                        .map(lambda x: (x[0]*x[2], x[0], x[1], np.exp(x[2])))\
                        .map(lambda x: (x[0], x[1], x[2], np.log(1+x[3]), x[3]/(1+x[3])))\
                        .map(lambda x: (x[1]*x[2]+x[2]*x[4], x[0]+x[3]))\
                        .reduce(lambda x, y: (x[0]+y[0], x[1]+y[1]))
        
        # calculate l2 regularization for both cost and gradient
        costReg = reg_factor * sum(r**2)  # a scalar
        gradientReg = 2 * reg_factor * r  # a 20k-length vector
        
        # Store cost for monitoring purpose
        cost = gradientCost[1] + costReg
        costHist.append(cost)
        
        # calculate new weights (coefficients)
        new_r = r - (gradientCost[0] + gradientReg) * learningRate
        
        # Stop if l2 norm of the difference in the coefficient vector 
        # across iterations is very small.
        l2 = np.linalg.norm((new_r-r), ord=2)
        if l2 <= precision:
            print("stoped at iteration", i)
            break
        
        # otherwise update the oldCost and r to the current cost and weights
        oldCost = cost
        r = new_r

        print("Iteration No.", i, cost, l2)
        iterHist.append([i, cost, l2])

    # store cost history in a single file on the cluster.
    dataToASingleFile = sc.parallelize(iterHist).coalesce(1)
    dataToASingleFile.saveAsTextFile(sys.argv[6])


    # Check the 5 Words with Largest Coefficients

    # dic position of the 5 words with largest coefficients
    topFiveR = (-r).argsort()[:5]

    # find the word in the dicitionary
    topFiveRWords = [(v, k) for k, v in dictionaryAsMap.items() if v in topFiveR]

    # Print the results
    print("\nTASK 2 - Top Five Words with Largest Coefficients: ")
    
    print("Dictionary positions and coefficients:")
    for i in topFiveR:
        print(i, r[i])

    print("Corresponding Words:")
    print(topFiveRWords)


    # ============= Model Evaluation ============= 

    # - STEP 1: Build TF matrix for Testing Sets (using the same Dictionary)

    # Create a RDD that has, for each (docID, ["word1", "word2", "word3", ...]),
    # ("word1", docID), ("word2", docId), ...
    test_allWordsWithDocID = test_keyAndListOfWords\
                                .flatMap(lambda x: ((j, x[0]) for j in x[1]))

    # Then do a simple map on it to get a set of (word, (dictionaryPos, docID)) pairs
    test_allDictionaryWords = test_allWordsWithDocID\
                                .map(lambda x: (x[0], (x[1], dictionaryAsMap.get(x[0]))) 
                                 if x[0] in dictionaryAsMap.keys() else None)\
                                .filter(lambda x: x!=None)\
                                .map(lambda x: (x[0], (x[1][1], x[1][0])))

    # Drop the actual word itself to get a set of (docID, dictionaryPos) pairs
    test_justDocAndPos = test_allDictionaryWords.map(lambda x: (x[1][1], x[1][0]))

    # Create a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
    test_allDictionaryWordsInEachDoc = test_justDocAndPos.groupByKey()

    # Converts the dictionary positions to a bag-of-words numpy array...
    # use the buildArray function to build the feature array
    # this gives a set of (docID, featureArray)
    test_allDocsAsNumpyArrays = test_allDictionaryWordsInEachDoc\
                                    .map(lambda x: (x[0], buildArray(x[1])))


    # - STEP 2: Make label prediction using the learned Logistic Regresssion model

    # Create an RDD that has (docID, label, TFArray) for all docs
    # Then, create an RDD that has (docID, label, predictedLabel) for all docs
    # **Note: Predict label = 1 if theta (x*r) > 0 else label = 0
    test_labelAndPrediction = test_allDocsAsNumpyArrays\
                            .map(lambda x: (x[0], 1, x[1]) if x[0][:2] == 'AU' else (x[0], 0, x[1]))\
                            .map(lambda x: (x[0], x[1], 1) if np.dot(x[2], r) > 0 else (x[0], x[1], 0))

    # Build an RDD that has for each doc a set of (docID, label, predictionNotation)
    # Ex: notation [0,0,1,0] means the prediction is a False Positive (FP)
    test_labelPredNotations = test_labelAndPrediction\
                                    .map(lambda x: (x[0], x[1], predictionNotation(x[1], x[2])))


    # Then, build a confusion matrix that has (P, N, [TP, FN, FP, TN])
    test_confusionMatrix = test_labelPredNotations\
                            .map(lambda x: (1, 0, x[2]) if x[1] == 1 else (0, 1, x[2]))\
                            .reduce(lambda x, y: (x[0]+y[0], x[1]+y[1], x[2]+y[2]))

    # Store the metrics
    P, N = test_confusionMatrix[0], test_confusionMatrix[1]
    TP, FN, FP, TN = test_confusionMatrix[2].ravel()

    # Calculate F-score
    test_fscore = f_score(TP, FP, P)

    print("\nTASK 3 - F1 Score: ")
    print("Number of positives (1-AU) docs:", P)
    print("Number of negatives (0-General) docs:", N)
    print("Confusion Matrix (TP, FN, FP, TN):", TP, FN, FP, TN)
    print("F1-score:", test_fscore)

    # check and store the ID for general documents (0) that have been classified as AU document (1), if any
    if FP == 0:
        print("There is no False Positive resulted from the model.")
    else:
        FP_docID = test_labelPredNotations\
                    .filter(lambda x: x[2][2] == 1)\
                    .map(lambda x: x[0])\
                    .collect()
        if FP <= 3:
            FP_text = test_keyAndText.filter(lambda x: x[0] in FP_docID).collect()
            print("There are", FP, "false positive classifications.")
            print("Doc ID:", FP_docID)
            print(FP_text)
            
        else:
            FP_text = test_keyAndText.filter(lambda x: x[0] in FP_docID[:3]).collect()
            print("There are", FP, "false positive classifications.")
            print("Doc ID (3 of them):", FP_docID[:3])
            print(FP_text)

    sc.stop()
