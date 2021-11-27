import pandas as pd 
import numpy  as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
import seaborn as sns
import statsmodels.api as sm 
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
# from sklearn import metrics
# from sklearn.cross_validation import cross_val_score


#Importing Data
affairs = pd.read_csv("C:\\Users\\Desktop\\Logistic Regression\\affairs.csv" )
print(affairs.head())

affairs['affairs'] = (affairs.affairs>0).astype(int)
affairs['children'] = (affairs.children == 'yes').astype(int)

print(affairs.head())

print(affairs.groupby('affairs').mean())

print(affairs.groupby('gender').mean())

print(affairs.groupby('rating').mean())

# affairs  gender   age  yearsmarried  children  religiousness  education  occupation  rating

affairs.gender.hist()
plt.title(" Histogram of gender ")
plt.xlabel(" Gender ")
plt.ylabel(" Frequency ")
plt.show()

affairs.education.hist()
plt.title(" Histogram of Education  ")
plt.xlabel(" Education Leevel ")
plt.ylabel(" Frequency ")
plt.show()

affairs.rating.hist()
plt.title(" Histogram of Marriage Rating ")
plt.xlabel(" Marriage Rating ")
plt.ylabel(" Frequency ")
plt.show()

pd.crosstab(affairs.rating, affairs.affairs.astype(bool)).plot(kind='bar')
plt.title(" Marriage rating affair vs affair ")
plt.xlabel(" Marriage Rating ")
plt.ylabel("Frequency ")
plt.show()


affairs_yrs_married = pd.crosstab(affairs.yearsmarried, affairs. affairs.astype(bool))
affairs_yrs_married.div(affairs_yrs_married.sum(1).astype(float),axis=0).plot(kind='bar', stacked= True)
plt.title(' Affair % by yearsmarried')
plt.xlabel('Years Married ')
plt.ylabel('Percentage')
plt.show()

pd.crosstab(affairs.yearsmarried, affairs.affairs.astype(bool)).plot(kind='bar')
plt.title(" Years Married vs affair ")
plt.xlabel(" Years Married ")
plt.ylabel("Percentage ")
plt.show()

print(affairs.head())
y,x = dmatrices('affairs ~ rating + age + yearsmarried + children +education + religiousness + C(occupation)' , affairs , return_type = "dataframe")
print(x.columns)

x=x.rename (columns = {'C(occupation)[T.2]' : 'occ_2', 'C(occupation)[T.3]': 'occ_3','C(occupation)[T.4]' : 'occ_4', 'C(occupation)[T.5]' : 'occ_5', 'C(occupation)[T.6]' : 'occ_6','C(occupation)[T.7]' : 'occ_7'})
print(x.head())

y = np.ravel(y)

model = LogisticRegression(max_iter = 300)
model = model.fit(x,y)
print(y.mean())

print(pd.DataFrame(list(zip(x.columns,np.transpose(model.coef_)))))

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state = 0 )
model2= LogisticRegression(max_iter = 300)
model2.fit(x_train, y_train)

y_pred = model2.predict_proba(x_test)
print(y_test)
print(y_pred)

prob = model2.predict_proba(x_test)
print(prob)

print(' length of X_Train ', len(x_train), 'length of Y_train ', len(y_train))

print(' length of X_Test ', len(x_test), 'length of Y_test ', len(y_test))
input()

#Noemalize the data 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
input()

# print(np.any(np.isnan(bank)))
print(pd.isnull(np.array([np.nan, 0], dtype=float)))
# print(np.all(np.isfinite(bank)))
from sklearn.linear_model import LogisticRegression 
clf= LogisticRegression()
clf.fit(x_train,y_train)
print(" Train Accuracy ", clf.score(x_train,y_train))
print(" Test Accuracy ", clf.score(x_test,y_test))
input()

y_pred = clf.predict(x_test)
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test , y_pred)
print(cm )
# print(metrics.accuracy_score(y_test.round(), y_pred.round(), normalize = False))
# input()
# print(metrics.roc_auc_score(y_test, probs [:,1]))
# input()
print(metrics.classification_report(y_test,y_pred))

