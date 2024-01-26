from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from A.function import svm_classification, RandomForest_classification, KNN_classification
from A.DataReshape import datareshape
from B.function import CNN_classification
from keras.utils import to_categorical

                            ####################    TaskA     ####################     
            # Use  3 machine learning methods to solve the task: SVM, KNN, and Random Forest
                            ####################     TaskA     ####################

# Load data
# Replace with .../Datasets/pneumoniamnist.npz
dataA = np.load('C:/Users/Lu34/OneDrive/Desktop/AMLS_assignment23_24--main/Datasets/pneumoniamnist.npz')
X_trainA = dataA['train_images']  
y_trainA = dataA['train_labels']
X_testA = dataA['test_images']
y_testA = dataA['test_labels']
X_valA = dataA['val_images']
y_valA = dataA['val_labels']

# Reshape data
X_trainAreshaped, y_trainAreshaped = datareshape(X_trainA, y_trainA)
X_testAreshaped, y_testAreshaped = datareshape(X_testA, y_testA)


# Method1: SVM
accuracy, report, y_pred_svm = svm_classification(X_trainAreshaped, y_trainAreshaped, X_testAreshaped, y_testAreshaped)
print(f"Accuracy Training of SVM: {accuracy * 100:.2f}%")
print("SVM Classification Report:")
print(report)

# Confusion Matrix for SVM
cm_svm = confusion_matrix(y_testAreshaped, y_pred_svm)

fig = plt.figure()
plt.subplot(2, 3, 3)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - SVM')
# Save the figure to the specified directory
file_path = 'C:/Users/Lu34/OneDrive/Desktop/AMLS_assignment23_24--main/results_and_analysis/Task_A/Confusion Matrix - SVM.pdf'
fig.savefig(file_path)

# Plot Learning Curve for SVM
svm_clf = svm.SVC(kernel='rbf')

fig = plt.figure()
train_sizes_svm, train_scores_svm, test_scores_svm = learning_curve(svm_clf, X_trainAreshaped, y_trainAreshaped, cv=3)
plt.subplot(2, 3, 1)
plt.plot(train_sizes_svm, np.mean(train_scores_svm, axis=1), label='Training Score')
plt.plot(train_sizes_svm, np.mean(test_scores_svm, axis=1), label='Validation Score')
plt.title('SVM Learning Curve')
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.legend()
# Save the figure to the specified directory
file_path = 'C:/Users/Lu34/OneDrive/Desktop/AMLS_assignment23_24--main/results_and_analysis/Task_A/Learning Curve - SVM.pdf'
fig.savefig(file_path)



# Method2: Random Forest
accuracy, report, y_pred_rf = RandomForest_classification(X_trainAreshaped, y_trainAreshaped, X_testAreshaped, y_testAreshaped)
print(f"Accuracy Training of Random Forest: {accuracy * 100:.2f}%")
print("Random Forest Classification Report:")
print(report)

# Confusion Matrix for Random Forest
cm_rf = confusion_matrix(y_testAreshaped, y_pred_rf)

fig = plt.figure()
plt.subplot(2, 3, 5)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Random Forest')
# Save the figure to the specified directory
file_path = 'C:/Users/Lu34/OneDrive/Desktop/AMLS_assignment23_24--main/results_and_analysis/Task_A/Confusion Matrix - RF.pdf'
fig.savefig(file_path)

# Plot Learning Curve for Random Forest
rf_clf = RandomForestClassifier(n_estimators=100) 

fig = plt.figure()
train_sizes_rf, train_scores_rf, test_scores_rf = learning_curve(rf_clf, X_trainAreshaped, y_trainAreshaped, cv=3)
plt.subplot(2, 3, 2)
plt.plot(train_sizes_rf, np.mean(train_scores_rf, axis=1), label='Training Score')
plt.plot(train_sizes_rf, np.mean(test_scores_rf, axis=1), label='Validation Score')
plt.title('Random Forest Learning Curve')
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.legend()
# Save the figure to the specified directory
file_path = 'C:/Users/Lu34/OneDrive/Desktop/AMLS_assignment23_24--main/results_and_analysis/Task_A/Learning Curve - RF.pdf'
fig.savefig(file_path)



# Method3: KNN
accuracy, report, y_pred_knn = KNN_classification(X_trainAreshaped, y_trainAreshaped, X_testAreshaped, y_testAreshaped)
print(f"Accuracy Training of KNN: {accuracy * 100:.2f}%")
print("KNN Classification Report:")
print(report)

# Confusion Matrix for KNN
cm_knn = confusion_matrix(y_testAreshaped, y_pred_knn)

fig = plt.figure()
plt.subplot(2, 3, 3)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - KNN')
# Save the figure to the specified directory
file_path = 'C:/Users/Lu34/OneDrive/Desktop/AMLS_assignment23_24--main/results_and_analysis/Task_A/Confusion Matrix - KNN.pdf'
fig.savefig(file_path)

KNN_clf = KNeighborsClassifier(n_neighbors=7)
train_sizes, train_scores, test_scores = learning_curve(KNN_clf, X_trainAreshaped, y_trainAreshaped, cv=5)

# Calculate mean and standard deviation across folds
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot learning curve
fig = plt.figure()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.legend(loc="best")
plt.title("Learning Curve - KNN")
# Save the figure to the specified directory
file_path = 'C:/Users/Lu34/OneDrive/Desktop/AMLS_assignment23_24--main/results_and_analysis/Task_A/Learning Curve - KNN.pdf'
fig.savefig(file_path)



                                        #############    TaskB     ###############
                                                # Use CNN to solve the task
                                        ############     TaskB     ###############


# Load data
# Replace with .../Datasets/pathmnist.npz
dataB = np.load('C:/Users/Lu34/OneDrive/Desktop/AMLS_assignment23_24--main/Datasets/pathmnist.npz')
X_trainB = dataB['train_images']  
y_trainB = dataB['train_labels']
X_testB = dataB['test_images']
y_testB = dataB['test_labels']
X_valB = dataB['val_images']
y_valB = dataB['val_labels']

width, height, channels = X_trainB.shape[1], X_trainB.shape[2], X_trainB.shape[3]
num_classes = 9
y_trainB = to_categorical(y_trainB, num_classes)
y_testB = to_categorical(y_testB, num_classes)

# Method: CNN
test_acc_cnn, test_loss_cnn, history_cnn, model_cnn = CNN_classification(X_trainB, y_trainB, X_testB, y_testB, width, height, channels, num_classes)
print(f"Accuracy Training of CNN: {test_acc_cnn * 100:.2f}%")

# Confusion Matrix for CNN
#y_pred_cnn = model_cnn.predict_classes(X_testB)

y_pred_cnn_prob = model_cnn.predict(X_testB)
y_pred_cnn = np.argmax(y_pred_cnn_prob, axis=1)
cm_cnn = confusion_matrix(np.argmax(y_testB, axis=1), y_pred_cnn)

fig = plt.figure()
plt.subplot(2, 3, 6)
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - CNN')
# Save the figure to the specified directory
file_path = 'C:/Users/Lu34/OneDrive/Desktop/AMLS_assignment23_24--main/results_and_analysis/Task_B/Confusion Matrix - CNN.pdf'
fig.savefig(file_path)


# CNN Training and Validation Accuracy Curve
fig = plt.figure()
plt.subplot(2, 3, 4)
plt.plot(history_cnn.history['accuracy'], label='Training Accuracy')
plt.plot(history_cnn.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
# Save the figure to the specified directory
file_path = 'C:/Users/Lu34/OneDrive/Desktop/AMLS_assignment23_24--main/results_and_analysis/Task_B/Training and Validation Accuracy Curve - CNN.pdf'
fig.savefig(file_path)
