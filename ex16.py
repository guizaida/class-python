from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import train_test_split as tts
from sklearn import neighbors as ngs
import pandas as pd


iris= datasets.load_iris()
x=pd.DataFrame(iris.data)
target = pd.DataFrame(iris.target,columns=["target"])
y = target["target"]
k=50

knn=ngs.KNeighborsClassifier(n_neighbors=k)
knn.fit(x,y)
# print("knn準確率:",knn.score(x, y))
trees=tree.DecisionTreeClassifier(max_depth=4)
trees.fit(x,y)
print("knn準確率:",knn.score(x, y),"\n決策樹準確率:",trees.score(x,y))
print("-"*30)
xtrain,xtest,ytrain,ytest =  tts(x,y,test_size=0.4,random_state=50)
a={}
for k in range(1,91):
    knn=ngs.KNeighborsClassifier(n_neighbors=k)
    knn.fit(xtrain,ytrain)
    print("當K為",k,'時的準確率',knn.score(xtest,ytest))
    a[k]=knn.score(xtest,ytest)
print("-"*30)
print("準確率最大值為:",max(a.values()))
print("-"*30)
#有0.9833333333333333的數值為3,4,10,11,12
c=[3,4,10,11,12]
k=min(c)
print('相同準確率下最小值為',k)
print("-"*30)
knn=ngs.KNeighborsClassifier(n_neighbors=k)
knn.fit(xtrain,ytrain)
print("當K值為",k,"的準確率",knn.score(xtest,ytest))
