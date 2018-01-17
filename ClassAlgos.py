import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "C:\Users\Akshat\Downloads\iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names) #laod dataset

print(dataset.shape) # 150 X 5

print(dataset.groupby('class').size())

dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show() #visualizing data

array = dataset.values #splitting dataset into training and testing set
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7 #psuedo random seed
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
seed = 7 #reset random seed
scoring = 'accuracy' # metric used to evaluate diff models  

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg+"\n")

print("Support Vector Machine:\n")	
svm= SVC() #most accurate model
svm.fit(X_train,Y_train)
predict=svm.predict(X_validation)  
print(accuracy_score(Y_validation, predict))
print(confusion_matrix(Y_validation, predict))
print(classification_report(Y_validation, predict))

print("K Nearest Neighbors :\n")
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


print("Logistic Regression :\n")
lr = LogisticRegression() # least accurate
lr.fit(X_train, Y_train)
prediction3 = lr.predict(X_validation)
print(accuracy_score(Y_validation, prediction3))
print(confusion_matrix(Y_validation, prediction3))
print(classification_report(Y_validation, prediction3))
