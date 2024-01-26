from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#Train and predict with the method of SVM
def svm_classification(X_train, y_train, X_test, y_test):
       
    clf = svm.SVC(kernel='rbf')# Kernel choose from 'linear', 'rbf', 'poly', 'sigmoid'#'rbf' has the highest accuracy
    clf.fit(X_train, y_train)
    y_pred_svm = clf.predict(X_test)

    #Calculate accurracy
    accuracy = accuracy_score(y_test, y_pred_svm)
    report = classification_report(y_test, y_pred_svm)
    
    return accuracy, report, y_pred_svm

#Train and predict with the method of KNN
def KNN_classification(X_train, y_train, X_test, y_test):
   
    knn = KNeighborsClassifier(n_neighbors=7)# Get the best n_neighbor from the pre-trained model
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred_knn)
    report = classification_report(y_test, y_pred_knn)
    
    return accuracy, report, y_pred_knn

#Train and predict with the method of RandomForest
def RandomForest_classification(X_train, y_train, X_test, y_test):
   
    rf_classifier = RandomForestClassifier(n_estimators=100)  
    rf_classifier.fit(X_train, y_train)
    y_pred_rf = rf_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred_rf)
    report = classification_report(y_test, y_pred_rf)

    return accuracy, report, y_pred_rf