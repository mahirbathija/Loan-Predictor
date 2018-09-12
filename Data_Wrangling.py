#Initializing import statements
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

#Reading in data
train=pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
test=pd.read_csv("test_Y3wMUE5_7gLdaTN.csv")

#Making copies of the files
train_original=train.copy()
test_original=test.copy()

#Understanding the data
print("\nTraining file columns:")
print(train.columns)
print("\nTesting file columns:")
print(test.columns)

#Understanding the data types
print("\nTraining file data types:")
print(train.dtypes)
print("\nTesting file data types:")
print(test.dtypes)

#Understanding shape(dimensions) of the data
print("\nTraining file dimensions:")
print(train.shape)
print("\nTesting file dimensions:")
print(test.shape)

#Target Variable (Loan Status) proportions
print("\nLoan Status Frequency:")
print(train['Loan_Status'].value_counts(normalize=False))

#Target Variable (Loan Status) proportions
print("\nLoan Status Proportion:")
print(train['Loan_Status'].value_counts(normalize=True))

#Plotting Loan Status Frequency and Proportion
plt.figure(1)
plt.subplot(121)
train['Loan_Status'].value_counts().plot.bar(title = 'Loan Status Frequency')
plt.subplot(122)
train['Loan_Status'].value_counts(normalize=True).plot.bar(title = 'Loan Status Proportion')


#Plotting Gender, Marital Status, Employment Status and Credit History Proportions
plt.figure(1)
plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar(title= 'Gender')
plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(title= 'Married')
plt.subplot(223)
train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self Employed')
plt.subplot(224)
train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit History')

#Plotting Dependents, Education and Property Area
plt.figure(1)
plt.subplot(131)
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,24), title= 'Dependents')
plt.subplot(132)
train['Education'].value_counts(normalize=True).plot.bar(title= 'Education', fontsize = 6)
plt.subplot(133)
train['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property Area', fontsize = 9)

#Plotting Applicant Income
plt.figure(1)
plt.subplot(121)
sns.distplot(train['ApplicantIncome']);
plt.subplot(122)
train['ApplicantIncome'].plot.box()

#Plotting Applicant Income by Education Level
#train.boxplot(column = 'ApplicantIncome', by = 'Education')


#Plotting Coapplicant Income
plt.figure(1)
plt.subplot(121)
sns.distplot(train['CoapplicantIncome']);
plt.subplot(122)
train['CoapplicantIncome'].plot.box(figsize=(16,5))


#Plotting Loan Amount
plt.figure(1)
plt.subplot(121)
df = train.dropna() #Removes midding values
sns.distplot(df['LoanAmount']);
plt.subplot(122)
train['LoanAmount'].plot.box(figsize=(16,5))

#Plotting Gender vs Loan Status
Gender=pd.crosstab(train['Gender'],train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, fontsize = 8)


#Plotting Marital Status vs Loan Status
Married=pd.crosstab(train['Married'],train['Loan_Status'])
Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(8,8))


#Plotting Dependents vs Loan Status
Dependents=pd.crosstab(train['Dependents'],train['Loan_Status'])
Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)


#Plotting Education vs Loan Status
Education=pd.crosstab(train['Education'],train['Loan_Status'])
Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(8,8), fontsize = 8)


#Plotting Self Employment vs Loan Status
Self_Employed=pd.crosstab(train['Self_Employed'],train['Loan_Status'])
Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(8,8))

#Plotting Credit History vs  Loan Status
Credit_History=pd.crosstab(train['Credit_History'],train['Loan_Status'])
Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(8,8))


#Plotting Property Area vs  Loan Status
Property_Area=pd.crosstab(train['Property_Area'],train['Loan_Status'])
Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, fontsize = 8)

#Plotting Applicant Income vs Loan Status
bins=[0,2500,4000,6000,81000]
group=['Low','Average','High', 'Very high']
train['Income_bin']=pd.cut(df['ApplicantIncome'],bins,labels=group)
Income_bin=pd.crosstab(train['Income_bin'],train['Loan_Status'])
Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, fontsize = 6, figsize = (8,8))
plt.xlabel('ApplicantIncome')
plt.ylabel('Percentage')

#Plotting Co-applicant Income vs Loan Status
bins=[0,1000,3000,42000]
group=['Low','Average','High']
train['Coapplicant_Income_bin']=pd.cut(df['CoapplicantIncome'],bins,labels=group)
Coapplicant_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status'])
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, fontsize = 6, figsize = (8,8))
plt.xlabel('CoapplicantIncome')
plt.ylabel('Percentage')

#Plotting Total Income vs Loan Status
train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
bins=[0,2500,4000,6000,81000]
group=['Low','Average','High', 'Very high']
train['Total_Income_bin']=pd.cut(train['Total_Income'],bins,labels=group)
Total_Income_bin=pd.crosstab(train['Total_Income_bin'],train['Loan_Status'])
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, fontsize = 6, figsize = (8,8))
plt.xlabel('Total_Income')
plt.ylabel('Percentage')

#Plotting Loan Amount vs Loan Status
bins=[0,100,200,700]
group=['Low','Average','High']
train['LoanAmount_bin']=pd.cut(df['LoanAmount'],bins,labels=group)
LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status'])
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, fontsize = 6, figsize = (8,8))
plt.xlabel('LoanAmount')
plt.ylabel('Percentage')


#Plotting corelation between data fields and Loan Status
train=train.drop(['Income_bin', 'Coapplicant_Income_bin', 'LoanAmount_bin', 'Total_Income_bin', 'Total_Income'], axis=1)
train['Dependents'].replace('3+', 3,inplace=True)
test['Dependents'].replace('3+', 3,inplace=True)
train['Loan_Status'].replace('N', 0,inplace=True)
train['Loan_Status'].replace('Y', 1,inplace=True)
matrix = train.corr()
f, ax = plt.subplots(figsize=(9, 12))
sns.set(font_scale=0.5)
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu")

#Printing missing values from training dataset
print('\n')
print(train.isnull().sum())

#Filling in missing values of categorical variables with mode (training dataset)
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)

#Filling in numerical variable with median
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

#Filling in missing values of all variables
test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

#Treating outliers/making the distribution more normal
train['LoanAmount_log'] = np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins=20)
test['LoanAmount_log'] = np.log(test['LoanAmount'])

#Model Making
train=train.drop('Loan_ID',axis=1)
test=test.drop('Loan_ID',axis=1)
X = train.drop('Loan_Status',1)
y = train.Loan_Status
X=pd.get_dummies(X)
train=pd.get_dummies(train)
test=pd.get_dummies(test)
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3)
model = LogisticRegression()
model.fit(x_train, y_train)
pred_cv = model.predict(x_cv)
print("Accuracy Score for training data: ")
print(accuracy_score(y_cv, pred_cv))
pred_test = model.predict(test)
submission=pd.read_csv("Sample_Submission_ZAuTl8O_FK3zQHh.csv")
submission['Loan_Status']=pred_test
submission['Loan_ID']=test_original['Loan_ID']
submission['Loan_Status'].replace(0, 'N',inplace=True)
submission['Loan_Status'].replace(1, 'Y',inplace=True)
pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('to_submit.csv')

#Using Stratified K-Fold Cross Validation to Improve Model
i = 1
kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
for train_index, test_index in kf.split(X, y):
    print('\n{} of kfold {}'.format(i, kf.n_splits))
    xtr, xvl = X.loc[train_index], X.loc[test_index]
    ytr, yvl = y[train_index], y[test_index]

    model = LogisticRegression(random_state=1)
    model.fit(xtr, ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    print('accuracy_score', score)
    i += 1
pred_test = model.predict(test)
pred = model.predict_proba(xvl)[:, 1]

#Visualizing ROC curve
fpr, tpr, _ = metrics.roc_curve(yvl,  pred)
auc = metrics.roc_auc_score(yvl, pred)
plt.figure(figsize=(12,8))
plt.plot(fpr,tpr,label="validation, auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)

#Adding varibles I think might affect Loan Status
train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
test['Total_Income']=test['ApplicantIncome']+test['CoapplicantIncome']
train['Total_Income_log'] = np.log(train['Total_Income'])
test['Total_Income_log'] = np.log(test['Total_Income'])
train['EMI']=train['LoanAmount']/train['Loan_Amount_Term']
test['EMI']=test['LoanAmount']/test['Loan_Amount_Term']
train['Balance Income']=train['Total_Income']-(train['EMI']*1000)
test['Balance Income']=test['Total_Income']-(test['EMI']*1000)
sns.distplot(train['Total_Income_log']);
sns.distplot(train['EMI']);
sns.distplot(train['Balance Income']);

#Removing Irrelevant Variables(used to make the new ones)
train=train.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)
test=test.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)

