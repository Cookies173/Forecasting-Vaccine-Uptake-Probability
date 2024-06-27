#!/usr/bin/env python
# coding: utf-8

# In[1]:


# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np
from scipy.stats import randint

# to make this notebook's output identical at every run
np.random.seed(42)

# Working with data
import pandas as pd

# Plotting library
from pandas.plotting import scatter_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces

# tells matplotlib to embed plots within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

#Sk-Learn 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier


# In[2]:


features=pd.read_csv('training_set_features.csv')
labels=pd.read_csv('training_set_labels.csv')


# In[3]:


features.head()


# In[4]:


labels.head()


# In[5]:


features.shape


# In[6]:


labels.shape


# In[7]:


features.info()


# In[8]:


labels.info()


# In[9]:


features.describe()


# In[10]:


features.describe(include='object')


# In[11]:


features.isnull().sum(axis=0)


# In[12]:


features.hist(bins=10, figsize=(14, 12))
plt.show()


# In[13]:


features.duplicated().sum()


# In[14]:


features['health_insurance'].fillna(value=0.0, inplace=True)


# In[15]:


X=features.drop(['respondent_id', 'employment_industry', 'employment_occupation'], axis=1)
y1=labels['xyz_vaccine']
y2=labels['seasonal_vaccine']


# In[16]:


X1_train, X1_test, y1_train, y1_test=train_test_split(X, y1, test_size=0.2, random_state=42)
X2_train, X2_test, y2_train, y2_test=train_test_split(X, y2, test_size=0.2, random_state=42)


# In[17]:


col_names=X1_train.columns


# In[18]:


X1_train.head()


# In[19]:


X1_train.shape


# In[20]:


y1_train.head()


# In[21]:


X2_test


# In[22]:


y2_test


# In[23]:


data=X1_test.copy()


# In[24]:


numeric=data.select_dtypes(exclude='object')
corr_matrix=numeric.corr()


# In[25]:


corr_matrix['chronic_med_condition'].sort_values(ascending=False)


# Some of the relations(Positive) which we can use later to impute the missing values in the data using them:
# 1. opinion_xyz_risk(Respondent's opinion about risk of getting sick with xyz flu without vaccine) & opinion_seas_risk(Respondent's opinion about risk of getting sick with 
# seasonal flu without vaccin)
# 2. opinion_xyz_vacc_effective(Respondent's opinion about xyz vaccine 
# effectivenes) & opinion_seas_vacc_effective(Respondent's opinion about seasonal flu 
# vaccine effectivenes)
# 3. doctor_recc_xyz(xyz flu vaccine was recommended by doctor) & doctor_recc_seasonal(Seasonal flu vaccine was recommended by doctor)
# 4. behavioral_outside_home(Has reduced contact with people outside of own
# househol) & behavioral_large_gatherings(Has reduced time at large gatherings)
# 5. opinion_xyz_sick_from_vacc(Respondent's worry of getting sick from 
# taking xyz vaccin) & opinion_seas_sick_from_vacc(Respondent's worry of getting sick from 
# taking seasonal flu vaccin)eedsse

# In[26]:


sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[27]:


sns.set_style('whitegrid')
temp = data['health_insurance'].astype(str)
sns.countplot(x='health_worker', hue=temp, data=data)


# In[28]:


impute_most_freq = data.drop(['health_insurance'], axis=1).columns
health_insurance=['health_insurance']


# In[29]:


impute_missing = ColumnTransformer([
                                    ('imp_mf', SimpleImputer(strategy='most_frequent'), impute_most_freq),
                                    ('imp_hi', SimpleImputer(strategy='constant', fill_value=0.0), health_insurance)
], remainder='passthrough')


# In[30]:


filled_data=data.copy()
filled_data=impute_missing.fit_transform(filled_data)
filled_data


# In[31]:


df=pd.DataFrame(filled_data, columns=col_names)
df.head()


# In[32]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[33]:


data=X1_train.copy()


# In[34]:


object_cols=data.select_dtypes(include='object').columns
object_cols


# In[35]:


data[object_cols].nunique()


# In[36]:


num_cols=data.select_dtypes(exclude='object').columns
num_cols


# In[37]:


encoding = ColumnTransformer([
                            ('hot', OneHotEncoder(handle_unknown='ignore'), object_cols), 
                            ('scaler', StandardScaler(), num_cols)
], remainder='passthrough')


# In[38]:


encoded_data=data.copy()
encoded_data=encoding.fit_transform(encoded_data)
encoded_data


# In[39]:


df=pd.DataFrame(encoded_data)


# In[40]:


df.head()


# In[41]:


categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('hot', OneHotEncoder(handle_unknown='ignore'))
])
continuous_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaling', StandardScaler())
])


# In[42]:


preprocessor = ColumnTransformer(transformers=[
    ('num', continuous_pipeline, num_cols),
    ('cat', categorical_pipeline, object_cols)
], remainder='passthrough')


# In[43]:


data=X1_train.copy()


# In[44]:


scaled_data=preprocessor.fit_transform(data)


# In[45]:


scaled_data.shape


# In[46]:


df=pd.DataFrame(scaled_data)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[47]:


sgd_clf = SGDClassifier(max_iter=500, random_state=42)
sgd_clf_model = make_pipeline(preprocessor, sgd_clf)
sgd_clf_model.fit(X1_train, y1_train)


# In[48]:


sgd_clf_model.score(X1_train, y1_train)


# In[49]:


sgd_clf_model.score(X1_test, y1_test)


# In[50]:


cross_val_score(sgd_clf_model, X1_train, y1_train, cv=5, scoring="accuracy")


# In[51]:


y1_pred = cross_val_predict(sgd_clf_model, X1_train, y1_train, cv=3)


# In[52]:


confusion_matrix(y1_train, y1_pred)


# In[53]:


precision_score(y1_train, y1_pred)


# In[54]:


recall_score(y1_train, y1_pred)


# In[55]:


f1_score(y1_train, y1_pred)


# In[56]:


y1_scores = cross_val_predict(sgd_clf_model, X1_train, y1_train, cv=3,
                             method="decision_function")


# In[57]:


y1_scores.shape


# In[58]:


precisions, recalls, thresholds = precision_recall_curve(y1_train, y1_scores)
plt.figure(figsize=(8, 4))
plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
plt.xlabel("Threshold", fontsize=16)
plt.legend(loc="upper left", fontsize=16)
plt.ylim([0, 1])
plt.show()


# In[59]:


fpr, tpr, thresholds = roc_curve(y1_train, y1_scores)
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.axis([0, 1, 0, 1])
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.figure(figsize=(8, 6))
plt.show()


# In[60]:


roc_auc_score(y1_train, y1_scores)


# In[61]:


svc = SVC()
svc_model = make_pipeline(preprocessor, svc)
svc_model.fit(X1_train, y1_train)


# In[62]:


svc_model.score(X1_train, y1_train)


# In[63]:


svc_model.score(X1_test, y1_test)


# In[64]:


rand = RandomForestClassifier(random_state=42)
rand_model = make_pipeline(preprocessor, rand)
rand_model.fit(X1_train, y1_train)


# In[65]:


rand_model.score(X1_train, y1_train)


# In[66]:


rand_model.score(X1_test, y1_test)


# In[67]:


y1_pred=rand_model.predict(X1_test)


# In[68]:


confusion_matrix(y1_test, y1_pred)


# In[69]:


y1_pred_prob = rand_model.predict_proba(X1_test)


# In[70]:


y1_pred_prob=y1_pred_prob[:, 1]


# In[71]:


roc_auc_score(y1_test, y1_pred_prob)


# In[72]:


cross_val_score(rand_model, X1_train, y1_train, cv=3, scoring="accuracy")


# In[73]:


# Hyperparameters
parameters = {
                'classifier__n_estimators': [30, 50, 100],
                'classifier__criterion' : ['gini', 'entropy', 'log_loss']
                }
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
grid_search = GridSearchCV(pipeline, param_grid=parameters, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(X1_train, y1_train)


# In[74]:


grid_search.best_params_


# In[75]:


pd.DataFrame(grid_search.cv_results_)


# In[76]:


param_distribs = {
        'classifier__n_estimators': randint(low=200, high=300),
        'classifier__criterion' : ['gini']
    }
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
rnd_search = RandomizedSearchCV(pipeline, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(X1_train, y1_train)


# In[77]:


rnd_search.best_params_


# In[78]:


pd.DataFrame(rnd_search.cv_results_)


# In[79]:


tuned_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, criterion='gini', n_estimators=288))
])


# In[80]:


tuned_pipeline.fit(X1_train, y1_train)


# In[81]:


tuned_pipeline.score(X1_train, y1_train)


# In[82]:


tuned_pipeline.score(X1_test, y1_test)


# In[83]:


y1_pred=tuned_pipeline.predict(X1_test)


# In[84]:


confusion_matrix(y1_test, y1_pred)


# In[85]:


y1_pred_prob = tuned_pipeline.predict_proba(X1_test)


# In[86]:


y1_pred_prob=y1_pred_prob[:, 1]


# In[87]:


roc_auc_score(y1_test, y1_pred_prob)


# In[88]:


tuned_pipeline.fit(X2_train, y2_train)
tuned_pipeline.score(X2_test, y2_test)


# In[89]:


# Hyperparameters
parameters = {
                'classifier__n_estimators': [200, 250, 200],
                'classifier__criterion' : ['gini', 'entropy', 'log_loss']
                }
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
grid_search = GridSearchCV(pipeline, param_grid=parameters, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(X2_train, y2_train)


# In[90]:


grid_search.best_params_


# In[91]:


pd.DataFrame(grid_search.cv_results_)


# In[92]:


param_distribs = {
        'classifier__n_estimators': randint(low=200, high=300),
        'classifier__criterion' : ['gini']
    }
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
rnd_search = RandomizedSearchCV(pipeline, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(X2_train, y2_train)


# In[93]:


rnd_search.best_params_


# In[94]:


pd.DataFrame(rnd_search.cv_results_)


# In[95]:


tuned_pipeline2 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, criterion='gini', n_estimators=274))
])


# In[96]:


tuned_pipeline2.fit(X2_train, y2_train)


# In[97]:


tuned_pipeline2.score(X2_train, y2_train)


# In[98]:


tuned_pipeline2.score(X2_test, y2_test)


# In[99]:


y2_pred_prob = tuned_pipeline.predict_proba(X2_test)
y2_pred_prob=y2_pred_prob[:, 1]


# In[100]:


roc_auc_score(y2_test, y2_pred_prob)


# In[101]:


# X_train, X_test, y_train, y_test =train_test_split(X, labels, test_size=0.2, random_state=42)


# In[102]:


# knn_clf = KNeighborsClassifier()
# knn_clf_model=make_pipeline(preprocessor, knn_clf)
# multi = MultiOutputClassifier(knn_clf_model, n_jobs=-1)
# knn_clf_model.fit(X_train, y_train)


# In[103]:


# y_train_pred = cross_val_predict(multi, X_train, y_train, cv=3, n_jobs=-1)
# y_train_pred = multi.predict(X_train)


# In[104]:


# from sklearn.metrics import classification_report
# knn_clf_model.score(X_train, y_train)
# multi.score(X_test, y_test)


# In[105]:


# round(f1_score(y_train, y_train_pred), 2)


# In[106]:


# f1_score(y_train, y_train_pred, average="macro")


# In[107]:


FINAL_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, criterion='gini', n_estimators=288))
])


# In[108]:


FINAL_pipeline.fit(X, y1)


# In[109]:


FINAL_pipeline.score(X, y1)


# In[110]:


submission = pd.read_csv('test_set_features.csv')


# In[111]:


submission.head()


# In[112]:


id=submission['respondent_id']


# In[113]:


submission.drop(['respondent_id', 'employment_industry', 'employment_occupation'], axis=1, inplace=True)


# In[114]:


y_proba_first = FINAL_pipeline.predict_proba(submission)


# In[115]:


y_proba_first


# In[116]:


y_proba_first=y_proba_first[:, 1]


# In[117]:


FINAL_pipeline2 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, criterion='gini', n_estimators=274))
])


# In[118]:


FINAL_pipeline2.fit(X, y2)


# In[119]:


FINAL_pipeline2.score(X, y2)


# In[120]:


y_proba_second = FINAL_pipeline2.predict_proba(submission)


# In[121]:


y_proba_second=y_proba_second[:, 1]


# In[122]:


first=pd.DataFrame(y_proba_first, columns=['h1n1_vaccine'])
first.head()


# In[123]:


second=pd.DataFrame(y_proba_second, columns=['seasonal_vaccine'])
second.head()


# In[124]:


submit=pd.concat([id, first, second], axis=1)


# In[125]:


submit.head()


# In[126]:


submit.shape


# In[127]:


submit.tail()


# In[129]:


submit.to_csv('Meet_Kotadiya_Datahack.csv', index=False)


# In[ ]:




