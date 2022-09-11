import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import os
import time
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics

data = pd.read_csv(os.path.join('data', 'OnlineNewsPopularity', 'OnlineNewsPopularity.csv'), sep="," )

df = pd.DataFrame(data)
df2 = df.drop(['url',' timedelta'], axis=1)

# pd.set_option('display.max_rows', 50)
# pd.set_option('display.max_columns',999)
# pd.set_option('display.width', 1000)
# pd.set_option('display.colheader_justify', 'center')
# pd.set_option('display.precision', 3)
#
#
# ### Find 10 most popular articles, sorting by shares
# df3=df.sort_values(by=[' n_tokens_title'], ascending=True)
# print(df3.head(10))
# top10=[]
# top10 = df3['url'], df3[' shares'][:10]
# print(top10)
#
#
#
# df.sort_values(by=[' shares'], ignore_index=True, inplace=True, ascending=False)
# print(df.head(20))
#
#
# # # #
#
#
# monday = df2[' weekday_is_monday'].sum()
# tuesday = df2[' weekday_is_tuesday'].sum()
# wednesday = df2[' weekday_is_wednesday'].sum()
# thursday = df2[' weekday_is_thursday'].sum()
# friday = df2[' weekday_is_friday'].sum()
# saturday = df2[' weekday_is_saturday'].sum()
# sunday = df2[' weekday_is_sunday'].sum()
# weekendmean=(saturday+sunday)/2
# print(weekendmean)
#
# print(monday, tuesday, wednesday, thursday, friday, saturday, sunday)
# weekdaymean=(monday+tuesday+wednesday+thursday+friday)/5
# print(weekdaymean)
#
# #
# # #### Dataset exploration - Getting an overview ##########
#
# print(df2.describe())
#
#
# ### plotting number of shares for articles published on each weekday
# weekday_vals = (df2.filter(like=" weekday_is")     # get only " weekday_is*" columns
#                   .idxmax(axis="columns")         # take the index-maximum to reverse 1-hot
#                   .str.split("_")                 # split over _ to help get the actual weekday
#                   .str[-1])                       # like "monday", "tuesday" etc.
#
# ax = plt.subplot()
# df2[" shares"].groupby(weekday_vals, sort=False).sum().plot.bar(ax=ax, edgecolor="black", linewidth=1, align='center', width=0.5)
# ax.set(xlim=(-1,5),xticks=np.arange(0,8), ylim=(5, 25000000))
# plt.ylabel('number of shares in million')
# plt.show()
#
# ### plotting number of shares against number of words for each article
#
# plt.scatter(df2[' shares'], df2[' n_tokens_content'])
# plt.title("Number of words in an article compared to number of shares")
# plt.xlabel("Number of shares")
# plt.ylabel("Number of words in content")
# plt.show()
# #
# plt.scatter(df2[' n_tokens_title' ], df2[' shares'])
# plt.title("Number of words in headline compared to number of shares")
# plt.xlabel("Number of words in headline")
# plt.ylabel("Number of shares")
# plt.show()



# ax.bar(np.where(df2[' weekday_is_monday'==1]), df2[' shares'])





# adding column to DataFrame for classification
df2 = df.drop(['url',' timedelta', ' n_tokens_title'], axis=1)
df2.loc[:, 'popular'] = '0'
df2['popular'] = np.where(df2[' shares']>=2800, 1, 0)









################## Train test split and normalizing Data ###########
## dropping irrelevant features

df2 = df.drop(['url',' timedelta', ' n_non_stop_words'], axis=1)

### creating new column to classify popularity >75% quantil
df2.loc[:, 'popular'] = '0'
df2['popular'] = np.where(df2[' shares']>=2800, 1, 0)


x_train, x_test, y_train, y_test = train_test_split(df2.drop(columns=['popular',' shares'] ), df2['popular'], test_size=0.3)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#
# models = {"kNN": KNeighborsClassifier(n_neighbors=10),
#           "Decision Tree": DecisionTreeClassifier(min_samples_leaf=30),
#           "Random Forest": RandomForestClassifier(min_samples_leaf=30),
#           "Adaboost": AdaBoostClassifier()}
#

# for name, model in models.items():
#         start_time = time.time()
#         model.fit(x_train,y_train)
#         print("Model {} scored with an accuracy of {:.2f}%".format(name, model.score(x_test, y_test)*100), "in {:.2f} seconds".format(time.time() - start_time))

# model1=DecisionTreeClassifier()
# model1.fit(x_train, y_train)
# importance = model1.feature_importances_
# # summarize feature importance
# for i,v in enumerate(importance):
# 	print('Feature: %0d, Score: %.5f' % (i,v))
# # plot feature importance
# plt.bar([x for x in range(len(importance))], importance)
# plt.show()


model2 = AdaBoostClassifier()
model2.fit(x_train, y_train)
y_pred = model2.predict(x_test)
ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
plt.show()
#
model3 = RandomForestClassifier()
model3.fit(x_train, y_train)
y_pred = model3.predict(x_test)
ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
plt.show()


ax=plt.subplot()
#
plt.figure(figsize=(8,6))
plt.title("ROC-Kurve")
fpr, tpr, thresholds = metrics.roc_curve(y_test.values, model2.predict_proba(x_test)[:,1:])
fpr2, tpr2, thresholds = metrics.roc_curve(y_test.values, model3.predict_proba(x_test)[:,1:])
plt.plot([0,1],[0,1],ls="--",c="black",alpha=0.2)
plt.plot(fpr,tpr,label="ROC-Graph AdaBoost",c="#1ACC94")
plt.plot(fpr2,tpr2,label="ROC-Graph RandomForest",c="#CC1B58")


plt.xlabel("False-Positive Rate (FPR)")
plt.ylabel("True-Positive Rate (TPR)")
plt.legend()
plt.show()

auroc = metrics.auc(fpr, tpr)
auroc2 = metrics.auc(fpr2,tpr2)
print(auroc, auroc2)
#

#
############
#
#
#
