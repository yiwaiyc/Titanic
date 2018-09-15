import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import  seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold,cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier



train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')
ntrain = train_data.shape[0]
ntest = test_data.shape[0]
train_id = train_data['PassengerId']
test_id = test_data['PassengerId']
train_data.drop('PassengerId',axis=1,inplace=True)
test_data.drop('PassengerId',axis=1,inplace=True)
y_train = train_data['Survived'].values
train_data.drop('Survived',axis=1,inplace=True)


all_data = pd.concat((train_data, test_data)).reset_index(drop=True)
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
#print(missing_data.head(20))

all_data.drop('Cabin',axis=1,inplace=True)
all_data.drop('Ticket',axis=1,inplace=True)
all_data.drop('Name',axis=1,inplace=True)

all_data['Age'] = all_data['Age'].fillna(all_data['Age'].mean())
all_data['Embarked'] = all_data['Embarked'].fillna(all_data['Embarked'].mode()[0])
all_data['Fare'] = all_data['Fare'].fillna(all_data['Fare'].mean())
all_data.drop('Fare',axis=1,inplace=True)
#print(all_data.head())
all_data = pd.DataFrame(data=all_data)
#print(type(all_data))
all_data = pd.get_dummies(all_data,columns=['Sex','Embarked'])
#print(all_data.head())

sc = StandardScaler()
all_data = sc.fit_transform(all_data)

#print('all_data**********************************%s'%all_data)
#all_data.to_csv('../data/all_data.csv')
train = all_data[:ntrain]
kf = KFold(n_splits=5)

lg = make_pipeline(StandardScaler(),LogisticRegression())#7889962965287804
clf  = make_pipeline(svm.SVC())#8260247316552635
dtree = make_pipeline(StandardScaler(),tree.DecisionTreeClassifier())#7946644906157806
nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)#8204255853367648
rf = make_pipeline(StandardScaler(),RandomForestClassifier(n_estimators=10)) #8002699140041429

model = rf
print(cross_val_score(model, train, y_train, scoring="accuracy", cv = kf).mean())

test = all_data[ntrain:]
model.fit(train,y_train)
result = model.predict(test)
sub = pd.DataFrame()
sub['PassengerId'] = range(892,1310)
sub['Survived'] = result
#sub.to_csv('../data/submission_4.csv',index=False)
"""
test = all_data[ntrain:]


logistic = LogisticRegression()
lo = logistic.fit(train,y_train)
result = lo.predict(test)
print(result)




sub = pd.DataFrame()
sub['PassengerId'] = range(892,1310)
sub['Survived'] = result
sub.to_csv('../data/submission_2.csv',index=False)
"""