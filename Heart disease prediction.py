#!/usr/bin/env python
# coding: utf-8

# # END TO END HEART DISEASE PREDICTION
#     This end to end machine learning project predicts if a person has heart disease or not based 
#     on their medical attributes, 
#     various data science and machine learning libraries

# ..............................................................................................................................................................................................................................................................

# ## Preparing the tools
#         We are going to use pandas, numpy, matplotlib for data analysis and classification
#         and scikit-learn for the logistic regression model to create a binary model for 
#         prediction.

# ## Data Dictionary
# 
# age - age in years<br>
# sex - (1 = male; 0 = female)<br>
# cp - chest pain type<br>
# 0: Typical angina: chest pain related decrease blood supply to the heart<br>
# 1: Atypical angina: chest pain not related to heart<br>
# 2: Non-anginal pain: typically esophageal spasms (non heart related)<br>
# 3: Asymptomatic: chest pain not showing signs of disease<br>
# trestbps - resting blood pressure (in mm Hg on admission to the hospital) anything above 130-140 is typically cause for concern<br>
# chol - serum cholestoral in mg/dl<br>
# serum = LDL + HDL + .2 * triglycerides<br>
# above 200 is cause for concern<br>
# fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)<br>
# '>126' mg/dL signals diabetes<br>
# restecg - resting electrocardiographic results<br>
# 0: Nothing to note<br>
# 1: ST-T Wave abnormality<br>
# can range from mild symptoms to severe problems<br>
# signals non-normal heart beat<br>
# 2: Possible or definite left ventricular hypertrophy<br>
# Enlarged heart's main pumping chamber<br>
# thalach - maximum heart rate achieved<br>
# exang - exercise induced angina (1 = yes; 0 = no)<br>
# oldpeak - ST depression induced by exercise relative to rest looks at stress of heart during excercise unhealthy heart will stress more<br>
# slope - the slope of the peak exercise ST segment<br>
# 0: Upsloping: better heart rate with excercise (uncommon)<br>
# 1: Flatsloping: minimal change (typical healthy heart)<br>
# 2: Downslopins: signs of unhealthy heart<br>
# ca - number of major vessels (0-3) colored by flourosopy<br>
# colored vessel means the doctor can see the blood passing through<br>
# the more blood movement the better (no clots)<br>
# thal - thalium stress result<br>
# 1,3: normal<br>
# 6: fixed defect: used to be defect but ok now<br>
# 7: reversable defect: no proper blood movement when excercising<br>
# target - have disease or not (1=yes, 0=no) (= the predicted attribute)<br>

# In[1]:


#import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
# To view the plots inside the notebook

# Scikit-Learn models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import plot_roc_curve


# ## loading the dataset

# In[3]:


df=pd.read_csv("heart-disease.csv")
df


# In[4]:


# lets find out how many of each class is present in the dataset
df.target.value_counts()


# In[5]:


# lets visualize the two class in a bar chart
df.target.value_counts().plot(kind="bar",color=["red","blue"]);


# In[6]:


# lets see if there's any missing value in the dataset
df.isna().sum()


# In[7]:


df.describe()


# In[8]:


# lets see the trend of heart disease with gender
pd.crosstab(df.target,df.sex)


# In[9]:


#visualizing the trend
pd.crosstab(df.target,df.sex).plot(kind="bar",
                                   figsize=(10,6),
                                   color=["red","blue"]);
plt.legend(["Female","Male"])
plt.title("heart disease frequency for gender")
plt.xlabel("0 -> No Disease , 1-> Disease")
plt.ylabel("NUmber of persons")
plt.xticks(rotation=0);


# In[10]:


# lets visualize for cholestrol with age and heart rate with age in two subplots
fig,(ax1,ax2)=plt.subplots(nrows=1,ncols=2,figsize=(20,6))

#cholestrol vs age
# plotting for disease
ax1.scatter(df.age[df.target==1],df.chol[df.target==1],c="red")
# plotting for no disease
ax1.scatter(df.age[df.target==0],df.chol[df.target==0],c="lightgreen")
ax1.set(title="age vs cholestrol",xlabel="age",ylabel="cholestrol")
ax1.legend(["disease","no disease"])

#Heart Rate vs age
# plotting for disease
ax2.scatter(df.age[df.target==1],df.thalach[df.target==1],c="red")
# plotting for no disease
ax2.scatter(df.age[df.target==0],df.thalach[df.target==0],c="lightgreen")
ax2.set(title="age vs Heart Rate",xlabel="age",ylabel="Heart Rate")
ax2.legend(["disease","no disease"]);


# In[11]:


# distribution of the age in the data
df.age.plot.hist()


# ### Heart disease frequency per chest pain type
# cp - chest pain type<br>
# 0: Typical angina: chest pain related decrease blood supply to the heart<br>
# 1: Atypical angina: chest pain not related to heart<br>
# 2: Non-anginal pain: typically esophageal spasms (non heart related)<br>
# 3: Asymptomatic: chest pain not showing signs of disease<br>

# In[12]:


pd.crosstab(df.cp,df.target).plot(kind="bar",color=["green","red"]);
plt.xlabel("type")
plt.ylabel("number of persons")
plt.legend(["no disease","disease"])
plt.title("Heart disease frequency per chest pain type")
plt.xticks(rotation=0);


# ## Checking the dependencies between the colums using the correlation matrix

# In[13]:


df.corr()


# In[14]:


#visualizing the correlation matrix
corr_matrix=df.corr()
fig,ax=plt.subplots(figsize=(15,10))
ax=sns.heatmap(corr_matrix,
               annot=True,
               fmt=".2f",
               cmap="YlGnBu");


# In[15]:


df.head()


# ### now as we have understood the dataset, lets split it into input and output, since it is a classification problem
# x -> input
# <br>
# y -> output

# In[16]:


x=df.drop("target",axis=1)
y=df.target


# In[17]:


x


# In[18]:


y


# ### lets split the input and output data for training and testing

# In[19]:


np.random.seed(42)
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2)


# In[20]:


len(x_train), len(y_train)


# In[21]:


len(x_test),len(y_test)


# In[22]:


x_test


# In[23]:


y_test


# ## Applying Machine Learning Models
# we will be applying various classification models and see which have better accuracy<br>
# 
# we will be using : <br>
# 1. Logistic Regression<br>
# 2. K Nearest Neighbour<br>
# 3. Random Forest

# In[24]:


# making a dictionary to put all the models
models = { "logistic regression":LogisticRegression(),
           "KNN":KNeighborsClassifier(),
           " Random Forest":RandomForestClassifier()}

# making a function which will train and test our data on all models

def train_test_check(models,x_train,x_test,y_train,y_test):
    
    #setting random seed
    np.random.seed(42)
    
    #making a empty dictionary which will score all the model score
    model_score={}
    
    #Iterating through all the models to train and test
    
    for name,model in models.items():
        model.fit(x_train,y_train)
        model_score[name]=model.score(x_test,y_test)
    return model_score


# In[25]:


#calling the function and displaying the scores
model_score=train_test_check(models,
                             x_train,
                             x_test,
                             y_train,
                             y_test)
model_score


# ### Visualizing the accuracy of the models

# In[26]:


model_compare=pd.DataFrame(model_score,index=["Accuracy"])
model_compare


# In[27]:


model_compare.T.plot.bar()
plt.xticks(rotation=0);


# ## lets make some changes in our models and see if there's is an increase in accuracy
# 
# we will be seeing:<br>
# * Hyperparameter tuning
# * Feature importance
# * Confusion Matrix
# * Cross-Validation
# * Precision
# * Recall
# * F1 score
# * Classification report
# * Roc curve
# * Area under the curve (AUC)
# 
# ### Hyperparameter tuning

# In[28]:


#Lets tune KNN

train_scores=[]
test_scores=[]

#create a list of different values for n_neighbours
neighbors=range(1,21)

#setup knn instance
knn=KNeighborsClassifier()

#iterate trough different neighbors
for i in neighbors:
    knn.set_params(n_neighbors=i)
    
    #fit the model
    knn.fit(x_train,y_train)
    
    #update training score list
    train_scores.append(knn.score(x_train,y_train))
    
    #updationg test score
    test_scores.append(knn.score(x_test,y_test))


# In[29]:


train_scores


# In[30]:


test_scores


# In[30]:


#lets visulaze the scores
plt.plot(neighbors,train_scores,label="Train score")
plt.plot(neighbors,test_scores,label="Test score")
plt.xlabel("Number of neighbors")
plt.ylabel("Model Score")
plt.legend()
plt.xticks(np.arange(1,21,1))
print(f"Maximum KNN score on the test data : {max(test_scores)*100:.2f}%")


# therefore there is increase in accuracy in KNN model from 68.85% to 75.41%, But it still less than the logistic regression model

# ## hyperparameter tuning of Logistic Regression and Random Forest Model using Randomized  Search cv (cross validation)

# In[31]:


# creating a hyperparameter grid for logistic Regression
log_reg_grid= {"C": np.logspace(-4,4,30),
               "solver": ["liblinear"]}

# Creating a hyperparameter grid for Random Forest Classifier
rf_grid={"n_estimators":np.arange(10,1000,50),
         "max_depth": [None, 3, 5, 10],
         "min_samples_split": np.arange(2,20,2),
         "min_samples_leaf":np.arange(1,20,2)}


# In[32]:


# tuning LOgistic Regression

np.random.seed(42)

#setting up random hyperparameter search for logistic regression
rs_log_reg=RandomizedSearchCV(LogisticRegression(),
                              param_distributions=log_reg_grid,
                              cv=5,
                              n_iter=20,
                             verbose=True)
#fit random hyperparameter
rs_log_reg.fit(x_train,y_train)


# In[33]:


rs_log_reg.best_params_


# In[34]:


#checking the new score
rs_log_reg.score(x_test,y_test)


# ### its nearly same as we have got in first time
# 
# 

# In[35]:


# tuning Random Forest

np.random.seed(42)

#setting up random hyperparameter search for Random Forest
rs_rf=RandomizedSearchCV(RandomForestClassifier(),
                              param_distributions=rf_grid,
                              cv=5,
                              n_iter=20,
                             verbose=True)
#fit random hyperparameter
rs_rf.fit(x_train,y_train)


# In[36]:


# checking the best parameter
rs_rf.best_params_


# In[37]:


#checking the score
rs_rf.score(x_test,y_test)


# ## old vs new accuracy

# In[38]:


#Old score
model_score


# In[39]:


#new score
print(f"logistic regression: {rs_log_reg.score(x_test,y_test)}, KNN {max(test_scores)},Random Forest {rs_rf.score(x_test,y_test)}")


# ## so as the logistic regression model is giving the higher accuracy, we are going for Logistic Regression

# 
# 
# 
# ## Evaluating our tuned model using 
# * ROC curve and AUC score
# * Confusion Matrix
# * Classification Report
# * Precision 
# * Recall
# * F1-Score
# 

# In[40]:


#Make predictions with tuned model
y_preds = rs_log_reg.predict(x_test)


# In[41]:


y_preds


# In[42]:


y_test


# In[43]:


#ploting ROC Curve and calculate Auc metric
plot_roc_curve(rs_log_reg,x_test,y_test);


# In[44]:


#confusion matrix
print(confusion_matrix(y_test,y_preds))


# In[45]:


# plotting the confusion matrix using heatmap

sns.set(font_scale=1.5)

def plot_conf_mat(y_test,y_preds):
    fig,ax=plt.subplots(figsize=(3,3))
    ax=sns.heatmap(confusion_matrix(y_test,y_preds),
                   annot=True,
                   cbar=False)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    
plot_conf_mat(y_test,y_preds)


#   ## calculating evaluation metrics using cross validation data

# In[46]:


#checking best hyperparameters
rs_log_reg.best_params_


# In[47]:


# making a model with best hyperparameter
clf = LogisticRegression(C=0.20433597178569418,
                         solver="liblinear")


# In[50]:


# cross validated accuracy
cv_acc = cross_val_score(clf,
                         x,
                         y,
                         cv=5,
                         scoring="accuracy")
cv_acc = np.mean(cv_acc)
cv_acc


# In[52]:


# cross validated precision
cv_precision = cross_val_score(clf,
                         x,
                         y,
                         cv=5,
                         scoring="precision")
cv_precision = np.mean(cv_precision)
cv_precision


# In[54]:


# cross validated recall
cv_recall = cross_val_score(clf,
                         x,
                         y,
                         cv=5,
                         scoring="recall")
cv_recall = np.mean(cv_recall)
cv_recall


# In[55]:


# cross validated f1
cv_f1 = cross_val_score(clf,
                         x,
                         y,
                         cv=5,
                         scoring="f1")
cv_f1 = np.mean(cv_f1)
cv_f1


# In[60]:


# visualizing creoss-validated metrics

cv_metrics = pd.DataFrame({"Accuracy":cv_acc,
                           "Precision":cv_precision,
                           "Recall":cv_recall,
                           "f1-score":cv_f1},
                           index=[0])
cv_metrics.T.plot.bar(title="Cross-Validated Classification metrics",
                      legend=False);


# ## Feature Importance
# 

# In[62]:


#checking coef list of all features
clf.fit(x_train,y_train)
clf.coef_


# In[65]:


# creating feature dictionary
feature_dict= dict(zip(df.columns,list(clf.coef_[0])))
feature_dict


# In[66]:


#visualizing feature importance
feature_df = pd.DataFrame(feature_dict,index=[0])
feature_df.T.plot.bar(title="Feature Importance", legend=False);


# ## so the most important feature in our dataset is cp which denotes 'chest pain'
# 
