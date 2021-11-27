import pandas as pd 
import numpy  as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
import seaborn as sns

#Importing Data
bank = pd.read_csv("C:\\Users\\Desktop\\Logistic Regression\\bank-full.csv" , sep=';')

print(bank.head(4))
print(bank.columns)

input()
# usage lambda and apply function
# apply function => we use to apply custom function operation on 
# each column
# lambda just an another syntax to apply a function on each value 
# without using for loop 
print("Is null " )
print(bank.isnull().sum())
print()

#  y (response)
# convert the response to numeric values and store as a new column
cat_col = [n for n in bank.columns if bank[n].dtypes == 'object']

for col in cat_col: 
	print(col, "\n\n")
	print(bank[col].value_counts())
	print(" ================ " * 4)

#To check the class Imbalance : 
No_sub = len ( bank['y'] == 0 )
print(No_sub)
Sub = len(bank['y'] == 1)
print(Sub)
percent_no_sub = ( No_sub/len(bank['y']))*100
percent_sub = ( Sub/len(bank['y']))*100

print( " Percent of non sub ", percent_no_sub)
print( " Percent of sub ", percent_sub)
input()

# bank['y'].value_counts.plt.plot(kind='bar')
pd.crosstab(bank['y'], bank.y).plot(kind='bar')
# data = bank['y'].value_counts
# plt.plot(kind = 'bar')
plt.show()
# points = data.index 
# freq = data.values

input()
#To display data with high influence 

for col in cat_col: 
	pd.crosstab(bank[col], bank.y).plot(kind='bar')
	plt.title('col')
plt.show()
input()

plt.figure(figsize =(10,6))
sns.distplot(a= bank['age'], kde= False)


print(bank['pdays'].value_counts)
bank ['pdays_no_contact'] = (bank['pdays'] == 999)*1
print(bank ['pdays_no_contact'].head(20))

# contact = ({'cellular':0, 'telephone':1})
# bank['contact'] = bank['contact'].map(contact)
# print(bank.columns)


bank = pd.get_dummies(bank, columns = ['job','marital','education','default','housing','loan','month','poutcome','contact'], drop_first = True)
print(bank.shape)
print(bank.columns)
print(bank.head())
input()

from sklearn.model_selection import train_test_split
x= bank.iloc[:, bank.columns != 'y' ]
y= bank.iloc[:, bank.columns == 'y' ]


x_train, x_test , y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)

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