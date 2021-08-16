#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skopt.plots import plot_convergence

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score, precision_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from skopt.searchcv import BayesSearchCV #pip install scikit-optimize
import warnings 

warnings.filterwarnings("ignore") #Ignore the numerous warnings by the bayesian search
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Read the dataset
df=pd.read_csv('adult.csv')
#The number of samples is 48842 and features is 15.
df.shape


# In[3]:


#Show the first five samples
df.head()


# ## Data Pre-Processing

# ### Handling Null Values
#         As we can see from the data, any unknown values are listed with a '?'. First we will replace the '?' with a NaN null value. Then we will handle the null values by removing all the samples which have null values.

# In[4]:


#Null values before replacing '?'
df.isnull().sum()


# In[5]:


#Loop through the dataframe and replace '?'
df.replace('?',value=np.NaN, inplace=True)
#Show the first five samples
df.head()


# In[6]:


#Null values after replacing '?'
df.isnull().sum()


# In[7]:


#Visualize null values in the dataset. 
#Heatmap before removing null values
sns.heatmap(df.isnull())


# In[8]:


#Drop all the rows which contain null values.
#We are dropping only the samples as its the minimum data we will have to remove to remove any null values in our dataset,
#whereas dropping by column would make us loose significant amount of information here
df.dropna(subset=['workclass'], how='any',inplace=True) 
df.dropna(subset=['occupation'], how='any',inplace=True)
df.dropna(subset=['native-country'], how='any',inplace=True)
df.shape #The number of smaples reduces from 48842 to 45222


# In[9]:


#Visualize null values in the dataset.
#Heatmap afetr removing null values.
sns.heatmap(df.isnull())


# ### Handling non-numerical and categorical values.
#           We will first look at all the dtypes in the dataset. If there exists any non-numerical or categorical values, then we will convert them by using LabelEncoder(). Another option would be to use pd.get_dummies() and then dropping and concataneting as nessecary.

# In[10]:


#Find the dtypes of the features
df.info()


#      As we can see, the features 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', and the target 'income' are categorical values. Therefore before applying a label encoder, we will look at all the unique values and the number of times a unique value occurs, in each column

# In[11]:


#No. of unique values in 'workclass' and the number of times each unique value occurs.
df['workclass'].value_counts()


# In[12]:


#No. of unique values in 'education' and the number of times each unique value occurs.
df['education'].value_counts()


# In[13]:


#No. of unique values in 'marital-status' and the number of times each unique value occurs.
df['marital-status'].value_counts()


# In[14]:


#No. of unique values in 'occupation' and the number of times each unique value occurs.
df['occupation'].value_counts()


# In[15]:


#No. of unique values in 'relationship' and the number of times each unique value occurs.
df['relationship'].value_counts()


# In[16]:


#No. of unique values in 'race' and the number of times each unique value occurs.
df['race'].value_counts()


# In[17]:


#No. of unique values in 'gender' and the number of times each unique value occurs.
df['gender'].value_counts()


# In[18]:


#No. of unique values in 'native-country' and the number of times each unique value occurs.
df['native-country'].value_counts()


# In[19]:


#No. of unique values in the target 'income' and the number of times each unique value occurs.
df['income'].value_counts()


# In[20]:


#Apply the label encoder to transfrom categorical values
df = df.apply(LabelEncoder().fit_transform) 
#Dataset after transformation                                                      
df.head()


# ### Splitting Dataset

# In[21]:


#Split dataset into dataframes with the target and without the target.
X=df.drop('income',axis=1)
y=df['income']
print(X.shape)
print(y.shape)


# ## Logistic Regression Without Feature Selection or Feature Scaling

# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
#create training models: split data innto training and testing sets
#test percenatge is 25%
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[23]:


#Fit the training test 
#warning as default iteration is 200 but our model is not converging in 200 iterations
#change maximum number of iterations
LR=LogisticRegression(max_iter=800)
LR.fit(X_train,y_train)


# #### Accuracy of Logistic Regression without feature scaling or feature selection

# In[24]:


print("Training score = {:.3f}\n Testing score = {:.3f}".format(LR.score(X_train,y_train), LR.score(X_test,y_test)))


# ## With Feature Selection and Feature Scaling

# ### Feature Selection

# In[25]:


#Heatmap to visualize correlation between the columns 
plt.figure(figsize=(10,10))
correlation_matrix = df.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)


#                  As we can see from the heatmap, the target income'ss correlation to other columns in a descending order is: 'capital_gain (0.34)' , 'educational-num (0.33)', 'age (0.24)', 'hours-per-week (0.23)', 'gender (0.22)', 'capital-loss (0.16)', 'education (0.08)', 'race (0.07)', 'occupation (0.05)', 'workclass (0.02)', 'native-country (0.02)','fnlwgt (-0.01)',    'martial-status (-0.19)', 'relationship (-0.25)'.
#                 Based on the above let us split data without the last three features which have the least correlation, that is correlation in negative value. We also remove the least impacting workclass and native-country features. Therefore we select 9 out of 15 features.

# In[26]:


X1=df[['capital-gain','educational-num','age','hours-per-week','gender','capital-loss','education','race','occupation']]
y1=df['income']
print(X1.shape)
print(y1.shape)


# In[27]:


X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.25, random_state=0)
#create training models: split data innto training and testing sets
#test percenatge is 25%
print(X1_train.shape)
print(y1_train.shape)
print(X1_test.shape)
print(y1_test.shape)


# ### Accuracy of Logistic Regression with feature selection

# In[28]:


#Fit the training test 
#warning as default iteration is 200 but our model is not converging in 200 iterations
#change maximum number of iterations
LR1=LogisticRegression(max_iter=800)
LR1.fit(X1_train,y1_train)
print("Training score = {:.3f}\n Testing score = {:.3f}".format(LR1.score(X1_train,y1_train), LR1.score(X1_test,y1_test)))


# ### Feature Scaling- Standard, MinMax, Robust Scaler

# In[29]:


#scaling data using Standard Scaler
sc=StandardScaler()
XS_train= sc.fit_transform(X1_train)
XS_test= sc.fit_transform(X1_test)


# In[30]:


#Fit the training test 
#warning as default iteration is 200 but our model is not converging in 200 iterations
#change maximum number of iterations
LR_FS1=LogisticRegression(max_iter=800)
LR_FS1.fit(XS_train,y1_train)


# In[31]:


#scaling data using MinMax Scaler
mc=MinMaxScaler()
XM_train= mc.fit_transform(X1_train)
XM_test= mc.fit_transform(X1_test)


# In[32]:


#Fit the training test 
#warning as default iteration is 200 but our model is not converging in 200 iterations
#change maximum number of iterations
LR_FS2=LogisticRegression(max_iter=800)
LR_FS2.fit(XM_train,y1_train)


# In[33]:


#scaling data using Robust Scaler
rc=RobustScaler()
XR_train= rc.fit_transform(X1_train)
XR_test= rc.fit_transform(X1_test)


# In[34]:


#Fit the training test 
#warning as default iteration is 200 but our model is not converging in 200 iterations
#change maximum number of iterations
LR_FS3=LogisticRegression(max_iter=800)
LR_FS3.fit(XR_train,y1_train)


# #### Accuracy of Logistic Regression with feature scaling and standard scaler

# In[35]:


print("Training score = {:.3f}\n Testing score = {:.3f}".format(LR_FS1.score(XS_train,y1_train), LR_FS1.score(XS_test,y1_test)))


# #### Accuracy of Logistic Regression with feature scaling and minmax scaler

# In[36]:


print("Training score = {:.3f}\n Testing score = {:.3f}".format(LR_FS2.score(XM_train,y1_train), LR_FS2.score(XM_test,y1_test)))


# #### Accuracy of Logistic Regression with feature scaling and robust scaler

# In[37]:


print("Training score = {:.3f}\n Testing score = {:.3f}".format(LR_FS3.score(XR_train,y1_train), LR_FS3.score(XR_test,y1_test)))


#        As we can see from above, model with feature scaling and all three types of scalers has the same training and testing score. Therefore, for further use we will use either one of the scaled training and testing sets. For ease of understanding, we will use the Robust scaled sets XR_train, y1_train, XR_test and y1_test.

# ## Confusion Matrix

# In[38]:


#Predicting values of the testing set
#As mentioned above, we are using the feature selected dataframes with the robust scaled sets and the model fitted with the robust scaled data. 
y1_test_pred=LR_FS3.predict(XR_test)
#Compare predicted values with actual values using a confusion matrix
cmat= confusion_matrix(y1_test,y1_test_pred)
print(cmat)


# In[39]:


titles_options = [('Confusion matrix without normalization', None), ('Confusion matrix with normalization', 'true')]
#Measure the normalised and non-normlised confusion matrices
for title, normalization in titles_options:
    cm_plt = plot_confusion_matrix(LR_FS3, XR_test, y1_test, display_labels = ['<=50K','>50K'], cmap=plt.cm.Reds, normalize = normalization)
    cm_plt.ax_.set_title(title)
    print(title)
    print(cm_plt.confusion_matrix)
#Plot the confusion matrices  
plt.show()


# ## Evaluation Metrics

# In[40]:


#Measure the evaluation metrics for the actual and predicted values
#The values have been predicted against the model fitted with feature selected and robust scaled traning sets
pd.DataFrame(data=[[accuracy_score(y1_test,y1_test_pred), recall_score(y1_test,y1_test_pred),
                   precision_score(y1_test,y1_test_pred), roc_auc_score(y1_test, y1_test_pred)]],
            columns=["accuracy","recall","precision","roc_auc_score"],
            index=['Score'])


# ## Classification Report

# In[41]:


#Use a classification report to print the evaluation metrics for each classification label(<=50K,>50K)
cReport = classification_report(y1_test,y1_test_pred,target_names=['<=50K','>50K'])
print(cReport)


# ## Cross Validation

# In[42]:


#Using the X1, y1 dataframes which have features selected (features=9)
n_range=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
n_scores=[]
for i in n_range:
    LR_CV=LogisticRegression(solver=i,max_iter=800)
    scores=cross_val_score(LR_CV, X1,y1,cv=5,scoring='accuracy')
    n_scores.append(scores.mean())
print(n_scores)


# In[43]:


#Mean of scores measured 
print("{:.3f}".format(scores.mean()))


# In[44]:


#Plot the measure CV scores to easily see which solver value has highest accuracy
plt.figure(figsize=(20,8))
plt.title("Cross Validation Scores")
plt.xlabel("Logistic Regression")
plt.ylabel("Accuracy")
plt.grid()
plt.plot(n_range,n_scores,c='Red')


# In[45]:


#Run the model with the CV measured best parameters
LR_CV=LogisticRegression(solver='liblinear',max_iter=800)
scores=cross_val_score(LR_CV, X1,y1,cv=5,scoring='accuracy')
print("{:.3f}".format(scores.mean()))


# ## Grid Search

# In[46]:


s_range=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
#p_range= ['none', 'l1', 'l2', 'elasticnet'] 
#Specific Penalty values are only supported by specific solvers
#SPecifci Dual values are only supported by some solvers
#d_range=[True,False]
#iter_range=range(800,1200) #800 is lower bound as we alrady know the model converges at 800 
#remove C from hyper-parameter tuning as accuracy with or without it is same
#c_range= [100, 10, 1.0, 0.1, 0.01]
para_grid=dict(solver=s_range)
#,C=c_range,max_iter=iter_range)
print(para_grid)


# In[47]:


#Use grid search to find the best hyperparameter
LR_GS=LogisticRegression(max_iter=800)
grid = GridSearchCV(LR_GS,para_grid,cv=5,scoring="accuracy",return_train_score=False)
grid.fit(X1,y1)


# In[48]:


#Print the grid search measured best hyperparameters
print(grid.best_score_) 
print(grid.best_params_) 
print(grid.best_estimator_) 


# ## Random Search

# In[49]:


s_range=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
c_range= [100, 10, 1.0, 0.1, 0.01]
para_rand=dict(solver=s_range,C=c_range)
#Run random search on both solver and C as this makes sure that the parameter space is not smaller than the n_iter=10
print(para_rand)


# In[50]:


#Use random search to find the best hyperparemeters
LR_RS=LogisticRegression(max_iter=800)
rand = RandomizedSearchCV(LR_RS,para_rand,cv=5,scoring="accuracy", random_state=0,return_train_score=False)
rand.fit(X1,y1)


# In[51]:


#Print the random search measured best hyperparameters
print(rand.best_score_) 
print(rand.best_params_) 
print(rand.best_estimator_) 


# ## Bayesian Search

# In[52]:


s_range=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
c_range= [100, 10, 1.0, 0.1, 0.01]
para_bayes=dict(solver=s_range,C=c_range)
print(para_bayes)


# In[53]:


#Use bayesian serach to find the best hyperparameter
LR_BS=LogisticRegression(max_iter=800)
bayes = BayesSearchCV(LR_BS,para_bayes,cv=5,scoring="accuracy", random_state=0,return_train_score=False)
bayes.fit(X1,y1)


# In[54]:


#Print the bayesian search measured best hyperparameters
print(bayes.best_score_) 
print(bayes.best_params_) 
print(bayes.best_estimator_) 

