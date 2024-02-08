import pandas as pd
import matplotlib .pyplot as plt
import seaborn as sns
import numpy as np
import sklearn

data = pd.read_csv("/content/apple_quality.csv")
datac = data.copy()
data.info()
datac.shape
datac.duplicated().sum()
datac.drop(4000,axis=0,inplace=True)
datac["Quality"].value_counts()

fv=data.iloc[:,1:-1]
cv=data.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(fv,cv,test_size=0.2,random_state=3,stratify=cv)

import scipy.stats as ss
for y in x_train.co.columns:
     plt.subplot(111)
     ss.probplot(x_train[y],dist="norm",fit=True,plot=plt)
     print(y)
     plt.show()

sns.kdeplot(x_train["Crunchiness"])

from sklearn.naive_bayes import GaussianNB
gb=GaussianNB()
model=gb.fit(x_train,y_train)

from sklearn.metrics import accuracy_score
predicted_yi=model.predict(X_test)
accuracy_score(y_test,predicted_yi)
cv.value_counts()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score,f1_score,classification_report,roc_curve
cm=confusion_matrix(y_test,model.predict(X_test))
cm
gb.classes_
predicted_y = model.predict(X_test)
precision_score(y_test,predicted_yi,pos_label=1)
precision_score(y_test,predicted_yi,pos_label="bad")
recall_score(y_test,predicted_yi,pos_label="good")
recall_score(y_test,predicted_yi,pos_label="bad")
f1_score(y_test,predicted_yi,pos_label="good")
print(classification_report(y_test,predicted_yi))
pr=model.predict_proba(x_test)[:,1]
fpr,tpr,_=roc_curve(y_test,pr) # prob coming from yor model
fpr1,tpr1,_=roc_curve(y_test,[0 for y in range(len(y_test))]) # prob coming from not learned model

plt.plot(fpr,tpr,marker=" .",label="gaussianbayes")
plt.plot(fpr,tpr,marker=" o ",label="random")
plt.title("roc curve")
plt.legend
plt.show()

import pickle
fm=pickle.load(open(r"/content/apple_quality.csv"))
fm.predict(x_test)