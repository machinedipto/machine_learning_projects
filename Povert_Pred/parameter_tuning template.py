import os
import re
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import classification_report,confusion_matrix, log_loss
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")


## Helper Functions 

#Read Data - Passing ID col makes lot of tasks easier
def read_data(output_path,filename,idcols=[]):
    df = pd.read_csv(os.path.join(output_path,filename))
    if idcols.__len__() > 0:
        df.set_index(idcols,inplace=True)
    return df

#Ipuatation Function - Not used in this code, as preprocessing done in different file 
def clean_df(df,id_cols=[]):
    from sklearn_pandas import gen_features,DataFrameMapper,CategoricalImputer

    print("Imputation of numeric Columns")
    if id_cols.__len__() > 0:
        df.set_index(keys=id_cols,inplace=True)

    df_numeric = df.select_dtypes(include=[int,float])
    df_non_num = df.select_dtypes(exclude=[int,float])

    num_imp_train = gen_features(
        columns = df_numeric.columns,
        classes = [CategoricalImputer]
    )

    # num_std_scale = gen_features(
    #     columns= df_numeric.columns,
    #     classes = [StandardScaler]
    # )

    num_map_train = DataFrameMapper(num_imp_train,df_out=True,input_df=True)
    # num_scale_map = DataFrameMapper(num_std_scale,df_out=True,input_df=True)
    print("Train Dataset numeric Impute")
    df_new = num_map_train.fit_transform(df_numeric)
    # print("Scaling Data")
    # df_new = num_scale_map.fit_transform(df_new)
    df_new = df_new.merge(df_non_num,left_index=True,right_index=True)
    print("Imputation with Mode Complete")
    return df_new

#Splitting of training to train, and validation. Train is used for GridSearch and Tuning. 
def prepare_and_split_data(df_train,target_col):
        x_data = df_train.drop(columns=target_col)
        y_data = df_train[target_col].astype(int)
        # x_main_test = df_test.drop(columns=target_col)
        # print("Classification Balance")
        # print(df_train.value_counts())
        x_train,x_val,y_train,y_val = train_test_split(x_data,y_data,test_size=0.20,random_state=45)
        return x_train,x_val,y_train,y_val


# Function to dynamically print, current best parameter being tuned
def print_best_tune_prams(model,param_name):
    modelparams = model.get_params()
    if type(param_name) == str:
        print(param_name)
        print(modelparams[param_name])
    if type(param_name) == list:
        for param_name in param_name:
            print(param_name)
            print(modelparams[param_name])

#Printing metrics of model, pass model, x variables and target. datype = to print if it is Training or Validation Metrics
def conf_metrics(model,xvals,yvals,datype):
    # print("Classification Report for {0}".format(datype))
    # print(classification_report(act,pred))
    print("Confusion Matrix for {0}".format(datype))
    conf_mat = confusion_matrix(yvals,model.predict(xvals))
    print(conf_mat)
    over_acc = ((conf_mat[1,1] + conf_mat[0,0])/ sum(sum(conf_mat)) * 100)
    print("Overall accuracy of {0} is {1}".format(datype,over_acc))
    print("Log Loss of {0}".format(log_loss(yvals,model.predict_proba(xvals))))

# Function to update grid search parameter grid with required list of value. 
# param_name can take 'str' or ['str','str']
# Param_value can take single list or lists of lists, if two strings are passed in param_name
def update_grid_list(param_grid,param_name,param_value):
    if type(param_name) == str:
        param_grid[param_name] = param_value

    if type(param_name) == list:
        for i,j in zip(param_name,param_value):
            param_grid[i] = j
            
    return param_grid


#Function to update model parameter, (not Used)
def update_model_param(model,param_name,param_value):
    if type(param_name) == str:
        model.param_name = param_value

    if type(param_name) == list:
        for i,j in zip(param_name,param_value):
            model.i = j
    return model

# Dynamincally get model parameters in dict format, and convert it to param_grid format
def conv_param_dict_to_grid(model):
    pdict = model.get_xgb_params()
    for i in pdict:
        lst = []
        x = pdict[i]
        lst.append(x)
        pdict[i] = lst
    return pdict

# Dynamic print of grid Search results, based on parameter being tuned
def print_rank(grid_results,param_name,h=10):
    other_cols = ['rank_test_score','mean_test_score']
    if type(param_name) == list:
        param_cols = ['param_'+str(i) for i in param_name]

        for j in param_cols:
            other_cols.append(j)
    if type(param_name) == str:
        other_cols.append('param_'+str(param_name))
    x = grid_results.sort_values('rank_test_score')[other_cols]
    print(x.head(h))


# Function, to tune parameter based on conditions.
def tune_grid_search(model,tobetunedparam,tunevals,trainx,trainy,score,verbose=3):
    model_params = conv_param_dict_to_grid(model)
    model_param_grid = update_grid_list(model_params,tobetunedparam,tunevals)
    gsearch = GridSearchCV(model,param_grid=model_param_grid,cv=3,scoring=score,n_jobs=2,verbose=verbose)
    gsearch.fit(trainx,trainy)
    result_df = gsearch.cv_results_
    return result_df, gsearch.best_estimator_


# Function to append, current grid search results to master grid search table
def update_master_results(newdf,new_val,olddf):
    import pandas as pd
    newdf['gridsearch'] = str(new_val)
    master_score_df = pd.concat([olddf,newdf])
    return master_score_df



## Output path, is the folder where your files are
output_path = "/home/dananjay/Desktop/Workstation/AllProjects/poverty2/enc_raw_data"
pred_folder = "/home/dananjay/Desktop/Workstation/AllProjects/poverty2/grid_pred_folder"
files = os.listdir(output_path)

recomple = re.compile('.*train.*')
testcomple = re.compile('.*test.*')
hh_compile = re.compile('.*hhold.*')
in_compile = re.compile('.*indiv.*')
file_list = list(filter(recomple.match,files))
file_list.sort()
file_list_test = list(filter(testcomple.match,files))
file_list_test.sort()
file_hh_train = list(filter(hh_compile.match,file_list))
file_hh_test = list(filter(hh_compile.match,file_list_test))
file_in_train = list(filter(in_compile.match,file_list))
file_in_test = list(filter(in_compile.match,file_list_test))

# Read big file and Split
df = read_data(output_path=output_path,filename=file_in_train[0],idcols=['id','iid'])
df['poor'].value_counts()
x_train,x_val,y_train,y_val = prepare_and_split_data(df,['poor'])

#Base Model - Create Base model, and Score Base Model
xgb_base = XGBClassifier(seed=100,nthread=2)
xgb_base.fit(x_train,y_train)
conf_metrics(xgb_base,x_train,y_train,"Train")
conf_metrics(xgb_base,x_val,y_val,"Validation")


#Line 171 to 188 is a manual iterative process
# We first tune, for n_estimators and learning_rate
# Then we tune, max_depth and min_child_weight
# Next - we tune Gamma
# Next - We tune subsample, colsample_bytree, colsample_bylevel
# Next - Regularization if required. 
# After above steps, obtain best model 
# To this new model,start new grid search where
# Learning rate is halved, and n_estimators is doubled
# For egs, if best model has 0.1 learning rate and 100 n_estimators
# do a grid search as follows learning rate [0.1,0.05,0.025m...] and n_estimators[100,200,400,..]


## Select parameter name to be tuned, pass the gridlist to current_param_list
current_param = ['n_estimators','learning_rate']
n_estimators = [100,50,25,200,400]
eta = [0.1,0.2,0.3]
current_param_list = [n_estimators,eta]

# Run below function
grid_results,grid_best_model = tune_grid_search(model=xgb_base,tobetunedparam=current_param,tunevals=current_param_list,trainx=x_train,trainy=y_train,score="log_loss",verbose=1)
grid_results = pd.DataFrame(grid_results)

# Print grid Results, keep note of the socre of the best model 
# PS.--- If the score has not improved from the previous iteration, then repeat gridsearch with different values
print_rank(grid_results=grid_results,param_name=current_param,h=5)
print(grid_best_model)

# Run the best model obtained, and note metrics on train and validation datasets. 
# The validation set, was not used in Gridsearch, so we will get fair idea how model will perform
# in actual test Dataset
conf_metrics(grid_best_model,x_train,y_train, "Train")
conf_metrics(grid_best_model,x_val,y_val,"Validation")

# If Satisfied with model performance on validation, run below code to save the current grid model, as the base model
xgb_base = grid_best_model
# Repeat process, and go back to iterative start to tune next parameter


# Final Prediction Stage - After retuning of n_estimators and learning rate

df = read_data(output_path=output_path,filename=file_hh_train[1],idcols=['id'])
dftest = read_data(output_path=output_path,filename=file_hh_test[1],idcols=["id"])

x_train = df.drop(columns=['poor'])
y_train = df['poor']
x_test = dftest.drop(columns='poor')

final_model = xgb_base

final_model.fit(x_train,y_train)
conf_metrics(final_model,x_train,y_train,"Train")
predictions = final_model.predict_proba(x_test)
preds = predictions[:,1]

x_test['poor'] = preds
x_test['country'] = 'B'
new_preds = x_test[['country','poor']]
new_preds.head(1)
new_preds.to_csv(os.path.join(output_path,"predictions","b_country.csv"))

pred_folder = os.path.join(output_path,"predictions")

# After parameter tunings for all countries 
df1 = read_data(pred_folder,'a_country.csv')
df2 = read_data(pred_folder,'b_country.csv')
df3 = read_data(pred_folder,'c_country.csv')
df4 = pd.concat([df1,df2,df3])
df4.to_csv(os.path.join(pred_folder,"final.csv"),index=False)

df4.head(1)

""""
# Best Models So Far - Houselhold
Country A 
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.4, gamma=0.3, learning_rate=0.003125,
       max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
       n_estimators=9600, n_jobs=1, nthread=1, objective='binary:logistic',
       random_state=0, reg_alpha=1, reg_lambda=1, scale_pos_weight=1,
       seed=100, silent=0, subsample=0.6)


Country B
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1.0,
       colsample_bytree=0.5, gamma=0.3, learning_rate=0.1,
       max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
       n_estimators=100, n_jobs=1, nthread=1, objective='binary:logistic',
       random_state=0, reg_alpha=0, reg_lambda=1.0, scale_pos_weight=1,
       seed=100, silent=0, subsample=0.9)

Country C
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=1, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=500, silent=0,
       subsample=0.8)

"""




