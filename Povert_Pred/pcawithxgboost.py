import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import warnings
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
warnings.filterwarnings("ignore")

DATAPATH = '/media/light/UbuntuDrive/Python_Code/Propython/Povert/combined_csv/'


def load_povert_data(dpath=DATAPATH):
    csv_path = os.path.join(DATAPATH, "country_a_combined_train.csv")
    return pd.read_csv(csv_path)


train_a = load_povert_data()
test_a = load_povert_data()

train_a_label = train_a['poor']
train_a = train_a.drop('poor', axis=1)
train_a_cleaned = train_a.drop('id', axis=1)
test_a_cleaned = test_a.drop('id', axis=1)

# creating the pipeline for scaling and label encoding
train_a_cat = train_a_cleaned.select_dtypes(include='object')
cat_columns_a = list(train_a_cat.columns)

train_a_num = train_a_cleaned.select_dtypes(include=['int64', 'float64'])
num_columns_a = list(train_a_num.columns)


class LabelEncoder_new(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):
        self.encoder = None
        return self

    def transform(self, X, y=None):
        if(self.encoder is None):
            print("Initializing encoder")
            self.encoder = LabelEncoder()
            result = self.encoder.fit_transform(X)
        else:
            result = self.encoder.transform(X)

        return result


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.attribute_names].values


class CustomOneHotEnocoder(TransformerMixin, BaseEstimator):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
        self.encoder = None
        self.encoder2 = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        if self.encoder is None:
            print("Initializing encoder")
            d = defaultdict(LabelEncoder)
            p = defaultdict(OneHotEncoder)
            df = pd.DataFrame(X)
            lencoded = df.apply(lambda x: d[x.name].fit_transform(x))
            lencoded.columns = self.attribute_names
            cat_list = []
            for i in range(len(self.attribute_names)):
                cat_list.append(i)
            hotcoded = OneHotEncoder(categorical_features=cat_list)
            hotcodedfit = hotcoded.fit_transform(lencoded)
            result = hotcodedfit
            return result.toarray()


num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_columns_a)),
    ('scaling', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_columns_a)),
    ('onehotencosing', CustomOneHotEnocoder(cat_columns_a)),
])

full_pipeline = FeatureUnion([
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline),
])


train_a_prepared = full_pipeline.fit_transform(train_a_cleaned)

train_a_prepared.shape
pca = PCA()

pca.fit(train_a_prepared)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.90) + 1
d

pca = PCA(n_components=d)
train_a_prepared_reduced = pca.fit_transform(train_a_prepared)

test_prepared = full_pipeline.fit_transform(test_a_cleaned)
test_prepared.shape
test_a_prepared_reduced = pca.fit_transform(test_prepared)

# now tuning parameters for xgboost

encoder = LabelEncoder()
train_a_label_encoded = encoder.fit_transform(train_a_label)


# Splitting of training to train, and validation. Train is used for GridSearch and Tuning.
def prepare_and_split_data(df_train, target_col):
    x_data = df_train.drop(columns=target_col)
    y_data = df_train[target_col].astype(int)
    # x_main_test = df_test.drop(columns=target_col)
    # print("Classification Balance")
    # print(df_train.value_counts())
    x_train, x_val, y_train, y_val = train_test_split(
        x_data, y_data, test_size=0.20, random_state=45)
    return x_train, x_val, y_train, y_val


# Function to dynamically print, current best parameter being tuned
def print_best_tune_prams(model, param_name):
    modelparams = model.get_params()
    if type(param_name) == str:
        print(param_name)
        print(modelparams[param_name])
    if type(param_name) == list:
        for param_name in param_name:
            print(param_name)
            print(modelparams[param_name])

# Printing metrics of model, pass model, x variables and target. datype = to print if it is Training or Validation Metrics


def conf_metrics(model, xvals, yvals, datype):
    # print("Classification Report for {0}".format(datype))
    # print(classification_report(act,pred))
    print("Confusion Matrix for {0}".format(datype))
    conf_mat = confusion_matrix(yvals, model.predict(xvals))
    print(conf_mat)
    over_acc = ((conf_mat[1, 1] + conf_mat[0, 0]) / sum(sum(conf_mat)) * 100)
    print("Overall accuracy of {0} is {1}".format(datype, over_acc))
    print("Log Loss of {0}".format(
        log_loss(yvals, model.predict_proba(xvals))))

# Function to update grid search parameter grid with required list of value.
# param_name can take 'str' or ['str','str']
# Param_value can take single list or lists of lists, if two strings are passed in param_name


def update_grid_list(param_grid, param_name, param_value):
    if type(param_name) == str:
        param_grid[param_name] = param_value

    if type(param_name) == list:
        for i, j in zip(param_name, param_value):
            param_grid[i] = j

    return param_grid


# Function to update model parameter, (not Used)
def update_model_param(model, param_name, param_value):
    if type(param_name) == str:
        model.param_name = param_value

    if type(param_name) == list:
        for i, j in zip(param_name, param_value):
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


def print_rank(grid_results, param_name, h=10):
    other_cols = ['rank_test_score', 'mean_test_score']
    if type(param_name) == list:
        param_cols = ['param_' + str(i) for i in param_name]

        for j in param_cols:
            other_cols.append(j)
    if type(param_name) == str:
        other_cols.append('param_' + str(param_name))
    x = grid_results.sort_values('rank_test_score')[other_cols]
    print(x.head(h))


# Function, to tune parameter based on conditions.
def tune_grid_search(model, tobetunedparam, tunevals, trainx, trainy, score, verbose=3):
    model_params = conv_param_dict_to_grid(model)
    model_param_grid = update_grid_list(model_params, tobetunedparam, tunevals)
    gsearch = GridSearchCV(model, param_grid=model_param_grid,
                           cv=3, scoring=score, n_jobs=1, verbose=verbose)
    gsearch.fit(trainx, trainy)
    result_df = pd.DataFrame(gsearch.cv_results_)
    return result_df, gsearch.best_estimator_


# Function to append, current grid search results to master grid search table
def update_master_results(newdf, new_val, olddf):
    import pandas as pd
    newdf['gridsearch'] = str(new_val)
    master_score_df = pd.concat([olddf, newdf])
    return master_score_df


x_train, x_val, y_train, y_val = train_test_split(
    train_a_prepared_reduced, train_a_label_encoded, test_size=0.20, random_state=45)
x_train.shape
y_train.shape
x_val.shape
y_val.shape
xgb_base = XGBClassifier(seed=100, nthread=2)
xgb_base.fit(x_train, y_train)
conf_metrics(xgb_base, x_train, y_train, "Train")
conf_metrics(xgb_base, x_val, y_val, "Validation")

current_param = ['n_estimators', 'learning_rate']
n_estimators = [100, 200, 300, 400]
learning_rate = [0.1, 0.05, 0.2, 0.3]
current_param_list = [n_estimators, learning_rate]

grid_results, grid_best_model = tune_grid_search(model=xgb_base, tobetunedparam=current_param, tunevals=current_param_list, trainx=x_train,
                                                 trainy=y_train, score="log_loss", verbose=3)

# Print grid Results, keep note of the socre of the best model
# PS.--- If the score has not improved from the previous iteration, then repeat gridsearch with different values
print_rank(grid_results=grid_results, param_name=current_param, h=5)
print(grid_best_model)

# Run the best model obtained, and note metrics on train and validation datasets.
# The validation set, was not used in Gridsearch, so we will get fair idea how model will perform
# in actual test Dataset
xgbtrial = XGBClassifier(base_score=0.5, colsample_bylevel=0.5, colsample_bytree=0.6,
                         gamma=6, learning_rate=0.0071428571428571435, max_delta_step=0,
                         max_depth=3, min_child_weight=9, missing=None, n_estimators=7000,
                         nthread=2, objective='binary:logistic', reg_alpha=0, reg_lambda=1,
                         scale_pos_weight=1, seed=100, silent=1, subsample=0.9)

xgbtrial.fit(x_train, y_train)
conf_metrics(xgbtrial, x_train, y_train, "Train")
conf_metrics(xgbtrial, x_val, y_val, "Validation")
print_best_tune_prams(model=grid_best_model, param_name=current_param)
conf_metrics(grid_best_model, x_train, y_train, "Train")
conf_metrics(grid_best_model, x_val, y_val, "Validation")
cross_val_score(xgbtrial, train_a_prepared_reduced,
                train_a_label_encoded, scoring="log_loss", cv=3)

current_param = ['max_depth', 'min_child_weight', 'n_estimators',
                 'learning_rate', 'gamma', 'subsample', 'colsample_bytree', 'colsample_bylevel']
max_depth = [3]
min_child_weight = [9]
n_estimators = [500, 500 * 2, 500 * 4, 500 *
                6, 500 * 8, 500 * 10, 500 * 12, 500 * 14]
learning_rate = [0.1, 0.1 / 2, 0.1 / 4, 0.1 / 8, 0.1 / 10, 0.1 / 12, 0.1 / 14]
gamma = [0]
subsample = [0.9]
colsample_bytree = [0.6]
colsample_bylevel = [0.5]
current_param_list = [max_depth, min_child_weight, n_estimators,
                      learning_rate, gamma, subsample, colsample_bytree, colsample_bylevel]
