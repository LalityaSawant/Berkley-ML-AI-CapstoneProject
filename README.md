# Repository: Berkley-ML-AI-Assignment-3

## This repository contains an assignment completed for comparing the performance of the classifiers (k-nearest neighbors, logistic regression, decision trees, and support vector machines) on the marketing of bank products data collected over the telephone.

### **Overview**

In this application, we will explore a dataset from the UCI Machine Learning repository. The data is from a Portuguese banking institution and is a collection of the results of multiple marketing campaigns. Our goal is to compare the performance of 4 different classifiers (k-nearest neighbors, logistic regression, decision trees, and support vector machines) using this dataset.

### **Problem Statement:**
The data is related to direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe to a term deposit (variable y). 

**Solution notebook link:** [Practical_Application_Assignment_17_1.ipynb: ](https://github.com/LalityaSawant/Berkley-ML-AI-Assignments-3/blob/master/Practical_Application_Assignment_17_1.ipynb)

To achieve this, a comprehensive analysis of the dataset is imperative. It includes below steps:

**Data Cleaning:** A meticulous review of the dataset to identify and rectify any inconsistencies or inaccuracies.

**Outlier Detection:** Identifying potential outliers and evaluating whether they should be excluded from the analysis.

**Bias Assessment:** Scrutinizing the dataset for any biases and implementing appropriate measures to address them.

**Text Data Transformation:** Converting textual data into a format understandable by the predictive model.

Once these preprocessing steps are accomplished, the subsequent task is to distribute the data in the training and testing set and then apply 4 different classifiers and compare their performance. I validated the performance by comparing the accuracy and time to train these models.

### **Some insights on Data:**
Original shape of data: (4521, 17)

RangeIndex: 4521 entries, 0 to 4520
Data columns (total 17 columns):
| #   |Column     |Non-Null Count  |Dtype |
|---  |------     |--------------  |----- |
| 0   |age        |4521 non-null   |int64 |
| 1   |job        |4521 non-null   |object|
| 2   |marital    |4521 non-null   |object|
| 3   |education  |4521 non-null   |object|
| 4   |default    |4521 non-null   |object|
| 5   |balance    |4521 non-null   |int64 |
| 6   |housing    |4521 non-null   |object|
| 7   |loan       |4521 non-null   |object|
| 8   |contact    |4521 non-null   |object|
| 9   |day        |4521 non-null   |int64 |
| 10  |month      |4521 non-null   |object|
| 11  |duration   |4521 non-null   |int64 |
| 12  |campaign   |4521 non-null   |int64 |
| 13  |pdays      |4521 non-null   |int64 |
| 14  |previous   |4521 non-null   |int64 |
| 15  |poutcome   |4521 non-null   |object|
| 16  |y          |4521 non-null   |object|
dtypes: int64(7), object(10)
memory usage: 600.6+ KB


## **Analysis Report:**                         
#####                                                  Author - Lalitya Sawant
## Which classifier performed better on this dataset?
#### Accuracy
| SMV Accuracy | LogisticReg Accuracy | DecisionTree Accuracy | KNN Accuracy |
|--------------|-----------------------|-----------------------|--------------|
| 0.890936     | 0.882093              | 0.866618              | 0.883567     |

#### Training time taken:
| SMV_train_time | LogisticReg_train_time | DecisionTree_train_time | KNN_train_time |
|----------------|-------------------------|-------------------------|----------------|
| 0.138083       | 0.009874                | 0.013814                | 0.002116       |

### **Details:**
#### KNN:
Training time KNN: 0.0021157264709472656
KNN accuracy: 0.8835666912306559
              precision    recall  f1-score   support

           0       0.89      0.99      0.94      1201
           1       0.46      0.08      0.14       156

    accuracy                           0.88      1357
   macro avg       0.68      0.54      0.54      1357
weighted avg       0.84      0.88      0.85      1357

#### SVM:
Training time SVM: 0.13808298110961914
SVM accuracy is: 0.8909358879882093
              precision    recall  f1-score   support

           0       0.90      0.99      0.94      1201
           1       0.65      0.11      0.19       156

    accuracy                           0.89      1357
   macro avg       0.77      0.55      0.56      1357
weighted avg       0.87      0.89      0.85      1357


#### Logistic Regression:
Training time Logistic Regression: 0.00987386703491211
Logistic Regression accuracy is: 0.8820928518791452
              precision    recall  f1-score   support

           0       0.90      0.98      0.94      1201
           1       0.46      0.13      0.21       156

    accuracy                           0.88      1357
   macro avg       0.68      0.56      0.57      1357
weighted avg       0.85      0.88      0.85      1357


#### Decision tree:
Training time Decision tree: 0.013813972473144531
Decision Tree accuracy is: 0.866617538688283
              precision    recall  f1-score   support

           0       0.92      0.93      0.92      1201
           1       0.42      0.40      0.41       156

    accuracy                           0.87      1357
   macro avg       0.67      0.67      0.67      1357
weighted avg       0.86      0.87      0.87      1357


### **Conclusion:**
#### Comparing different models we can see that SVM gives a good accuracy over others but also the traning time taken in the most out of 4 models. On the other hand, KNN model takes the lowest time to train the model but the accuracy is low.
