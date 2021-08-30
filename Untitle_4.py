# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/Users/Pang/Downloads/Reddit_Classification.csv')
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13:14].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

#Predicting the Test set results
y_pred_nb = nb_classifier.predict(X_test)

# Making the Confusion Matrix and generating the classification report  
from sklearn.metrics import confusion_matrix
nb_cm = confusion_matrix(y_test, y_pred_nb)
nb_cm


# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'euclidean')
knn_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_knn = knn_classifier.predict(X_test)

#Printing the confusion matrix for KNN
knn_cm = confusion_matrix(y_test, y_pred_knn)
knn_cm


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
lg_classifier= LogisticRegression(random_state = 0)
lg_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_lg= lg_classifier.predict(X_test)

#Printing the confusion matrix for Logistic regression
lg_cm = confusion_matrix(y_test, y_pred_lg)
lg_cm


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, criterion= 'entropy', random_state = 0)
rf_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_rf= rf_classifier.predict(X_test)

# Confusion matrix for random forest
rf_cm = confusion_matrix(y_test, y_pred_rf)
rf_cm


# Getting the confusion matrix metrics

def confusion_metrics(conf_matrix):
    # save confusion matrix and slice into four pieces
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    print('True Positives:', TP)
    print('True Negatives:', TN)
    print('False Positives:', FP)
    print('False Negatives:', FN)

    # calculate accuracy    
    conf_accuracy = (float (TP+TN) / float(TP + TN + FP + FN))    
    
    # calculate mis-classification    
    conf_misclassification = 1- conf_accuracy    
    
    # calculate the sensitivity    
    conf_sensitivity = (TP / float(TP + FN))    
    
    # calculate the specificity    
    conf_specificity = (TN / float(TN + FP))    
    
    # calculate precision    
    conf_precision = (TN / float(TN + FP))    
    
    # calculate f_1 score    
    conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))    
    print('-'*50)    
    print(f'Accuracy: {round(conf_accuracy,2)}') 
    print(f'Mis-Classification: {round(conf_misclassification,2)}')     
    print(f'Sensitivity: {round(conf_sensitivity,2)}')     
    print(f'Specificity: {round(conf_specificity,2)}')     
    print(f'Precision: {round(conf_precision,2)}')    
    print(f'f_1 Score: {round(conf_f1,2)}')


'''Printing the metrics for each of the 4 algorithms'''
#Confusion matrix metrics for naive bayes 
confusion_metrics (nb_cm)

#Confusion matrix metrics for knn
confusion_metrics (knn_cm)

#Confusion matrix metrics for Logistic Regression
confusion_metrics (lg_cm)

#Confusion matrix metrics for Random Forest
confusion_metrics (rf_cm)