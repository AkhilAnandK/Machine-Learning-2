print("Beginning of Assignment 2\n")

#Loading the dataset

from sklearn.preprocessing import OneHotEncoder
from sklearn import datasets
lis = datasets.fetch_openml(data_id=688)
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("encoder", OneHotEncoder(sparse=False), ['isns'])], remainder="passthrough")
new_data = ct.fit_transform(lis.data)
import pandas as pd
lis_new_data = pd.DataFrame(new_data, columns = ct.get_feature_names(), index = lis.data.index)

#Decision Tree
print("Decision Tree Regressor\n")

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import learning_curve
dtr=DecisionTreeRegressor()
parameters = [{"min_samples_leaf": [2,4,6,8,10]}]
from sklearn import model_selection
dt_tuned=model_selection.GridSearchCV(dtr, parameters, scoring="neg_root_mean_squared_error", cv = 10)
dt_score=model_selection.cross_validate(dt_tuned, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")
rmse_dt = 0-dt_score["test_score"]
print("The RMSE mean value of Decision tree Regressor is: ",rmse_dt.mean())
dt_train_sizes, dt_train_scores, dt_test_scores, dt_fit_times, dt_score_times = learning_curve(dtr, lis_new_data, lis.target, train_sizes=[0.2, 0.4, 0.6, 0.8, 1], cv=10,return_times=True,scoring="neg_root_mean_squared_error",shuffle=True,random_state=0)
dtlc_rmse=0 - dt_test_scores
print("\nThe training computational time for Decision Tree Regressor is:",dt_fit_times[4].mean())
print("\nThe testing computational time for Decision Tree Regressor is:",dt_score_times[4].mean())

#KNN Regressor
print("\nKNearestNeighborsRegressor\n")
from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor(n_neighbors=14)
knn_score = model_selection.cross_validate(knn, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")
rmse_knn = 0-knn_score["test_score"]
print("The RMSE mean value of K Nearest Neighbor Regressor is: ",rmse_knn.mean())
knn_train_sizes, knn_train_scores, knn_test_scores, knn_fit_times, knn_score_times = learning_curve(knn, lis_new_data, lis.target, train_sizes=[0.2, 0.4, 0.6, 0.8, 1], cv=10,return_times=True,scoring="neg_root_mean_squared_error",shuffle=True,random_state=0)
knn_rmse=0 - knn_test_scores
print("\nThe training computational time for KNN Regressor is:",knn_fit_times[4].mean())
print("\nThe testing computational time for KNN Regressor is:",knn_score_times[4].mean())

#Linear Regression
print("\nLinear Regression\n")
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
lr = LinearRegression()
scores_lr= cross_validate(lr, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")
rmse_lr = 0-scores_lr["test_score"]
print("The RMSE mean value of Linear Regressor is: ",rmse_lr.mean())
lr_train_sizes, lr_train_scores, lr_test_scores, lr_fit_times, lr_score_times = learning_curve(lr, lis_new_data, lis.target, train_sizes=[0.2, 0.4, 0.6, 0.8, 1], cv=10,return_times=True,scoring="neg_root_mean_squared_error",shuffle=True,random_state=0)
lr_rmse=0 - lr_test_scores
print("\nThe training computational time for Linear Regressor is:",lr_fit_times[4].mean())
print("\nThe testing computational time for Linear Regressor is:",lr_score_times[4].mean())

#Support Vector Machine
print("\nSupport Vector Machine")
from sklearn.svm import SVR
sv=SVR()
scores_sv= cross_validate(sv, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")
rmse_sv = 0-scores_sv["test_score"]
print("The RMSE mean value of Support Vector Machine Regressor is: ",rmse_sv.mean())
sv_train_sizes, sv_train_scores, sv_test_scores, sv_fit_times, sv_score_times = learning_curve(sv, lis_new_data, lis.target, train_sizes=[0.2, 0.4, 0.6, 0.8, 1], cv=10,return_times=True,scoring="neg_root_mean_squared_error",shuffle=True,random_state=0)
sv_rmse=0 - sv_test_scores
print("\nThe training computational time for Linear Regressor is:",sv_fit_times[4].mean())
print("\nThe testing computational time for Linear Regressor is:",sv_score_times[4].mean())

#Bagged Decision Tree
print("\nBagged Decision Tree\n")
from sklearn.ensemble import BaggingRegressor
br = BaggingRegressor()
scores_bdt= cross_validate(br, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")
rmse_bdt = 0-scores_bdt["test_score"]
print("The RMSE mean value of Bagged Decision Tree Regressor is:",rmse_bdt.mean())
bdt_train_sizes, bdt_train_scores, bdt_test_scores, bdt_fit_times, bdt_score_times = learning_curve(br, lis_new_data, lis.target, train_sizes=[0.2, 0.4, 0.6, 0.8, 1], cv=10,return_times=True,scoring="neg_root_mean_squared_error",shuffle=True,random_state=0)
bdt_rmse=0 - bdt_test_scores
print("\nThe training computational time for Linear Regressor is:",bdt_fit_times[4].mean())
print("\nThe testing computational time for Linear Regressor is:",bdt_score_times[4].mean())

#Dummy Regressor
print("\nDummy Regressor\n")
from sklearn.dummy import DummyRegressor
dr=DummyRegressor()
scores_dr= cross_validate(dr, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")
rmse_dr = 0-scores_dr["test_score"]
print("The RMSE mean value of Dummy Regressor is: ",rmse_dr.mean())
dr_train_sizes, dr_train_scores, dr_test_scores, dr_fit_times, dr_score_times = learning_curve(dr, lis_new_data, lis.target, train_sizes=[0.2, 0.4, 0.6, 0.8, 1], cv=10,return_times=True,scoring="neg_root_mean_squared_error",shuffle=True,random_state=0)
dr_rmse=0 - dr_test_scores
print("\nThe training computational time for Linear Regressor is:",dr_fit_times[4].mean())
print("\nThe testing computational time for Linear Regressor is:",dr_score_times[4].mean())

#Plotting the Learning Curve
print("\nPlotting the Learning Curve\n")
from matplotlib import pyplot
pyplot.plot(dt_train_sizes, dtlc_rmse.mean(axis=1), label = "DecisionTreeRegressor")
pyplot.plot(knn_train_sizes, knn_rmse.mean(axis=1), label = "KNearestNeighborsRegressorighborsRegressor")
pyplot.plot(lr_train_sizes, lr_rmse.mean(axis=1), label = "Linear Regression")
pyplot.plot(sv_train_sizes, sv_rmse.mean(axis=1), label = "Support Vector Machine")
pyplot.plot(bdt_train_sizes, bdt_rmse.mean(axis=1), label = "Bagging Tree Regressor")
pyplot.plot(dr_train_sizes, dr_rmse.mean(axis=1), label = "Dummy Regressor")
pyplot.xlabel("Number of training examples")
pyplot.ylabel("RMSE value")
pyplot.legend()
pyplot.show()

#Statistical Significance
print("\nStatistical Significance\n")
import scipy
print("Statistical signifcance between Bagged Tree and Decision Tree:",scipy.stats.ttest_rel(rmse_bdt,rmse_dt))
print("\nStatistical signifcance between Bagged Tree and KNearestNeighbors:",scipy.stats.ttest_rel(rmse_bdt,rmse_knn))
print("\nStatistical signifcance between Bagged Tree and Linear Regression:",scipy.stats.ttest_rel(rmse_bdt,rmse_lr))
print("\nStatistical signifcance between Bagged Tree and Support Vector Machine:",scipy.stats.ttest_rel(rmse_bdt,rmse_sv))
print("\nStatistical signifcance between Bagged Tree and Dummy Regressor:",scipy.stats.ttest_rel(rmse_bdt,rmse_dr))

print("\nEnd of Assignment 2")
