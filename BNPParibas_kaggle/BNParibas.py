import os
import numpy as np
import pandas as pd
from fancyimpute import MICE
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from collections import defaultdict
from scipy import stats
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb


BNPPATH = '/media/light/UbuntuDrive/Python_Code/Propython/BNPParibas/'


def load_BNP_data(bnppath=BNPPATH):
    csv_path = os.path.join(BNPPATH, "test.csv")
    return pd.read_csv(csv_path)


traindf = load_BNP_data()
testdf = load_BNP_data()

traindf.describe()
testdf.describe()

testdf['traget'] = 0
traindf['flag'] = 'Train'
testdf['flag'] = 'Test'


train_num = traindf.select_dtypes(include=['float64', 'int64'])

train_cat = traindf.select_dtypes(include=['object'])

test_cat = testdf.select_dtypes(include=['object'])


np.unique(train_cat['v47'].values)
np.unique(test_cat['v47'].values)

np.unique(train_cat['v79'])
np.unique(test_cat['v79'])

type(train_cat)

traindf.drop(traindf.index[59613], inplace=True)
traindf.drop(traindf.index[66130], inplace=True)

# getiing the names of columns in data frame
columns = list(train_cat)

for column in columns:
    print(column, train_cat[column].isnull().sum())
train_cat.describe()

train_cat = train_cat.drop(
    ['v22', 'v30', 'v31', 'v56', 'v125', 'v113', 'flag'], axis=1)
traindf = traindf.drop(
    ['v22', 'v30', 'v31', 'v56', 'v125', 'v113', 'flag'], axis=1)
test_cat = test_cat.drop(
    ['v22', 'v30', 'v31', 'v56', 'v125', 'v113', 'flag'], axis=1)
testdf = testdf.drop(
    ['v22', 'v30', 'v31', 'v56', 'v125', 'v113', 'flag'], axis=1)

train_cat = train_cat.drop('flag', axis=1)

train_num_predictors = train_num.drop(['target', 'ID'], axis=1)

train_labels = traindf['target']
cat_attributes = list(train_cat)
num_attributes = list(train_num_predictors)


full_train_predictors = num_attributes + cat_attributes
full_train_predictors.remove('flag')

train_predictors = traindf[full_train_predictors]
test_predictors = testdf[full_train_predictors]
train_num = train_predictors.select_dtypes(include=['float64', 'int64'])
train_cat = train_predictors.select_dtypes(include=['object'])
cat_attributes = list(train_cat)
num_attributes = list(train_num)

try1train = train_predictors[['v10', 'v12', 'v14', 'v21',
                              'v34', 'v38', 'v40', 'v50', 'v62', 'v72', 'v114', 'v129']]
try1test = test_predictors[['v10', 'v12', 'v14', 'v21',
                            'v34', 'v38', 'v40', 'v50', 'v62', 'v72', 'v114', 'v129']]
cat_attributes


class CatImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mode', filler='NA'):
        self.strategy = strategy
        self.fill = filler

    def fit(self, X, y=None):
        if self.strategy == 'mode':
            datfr = pd.DataFrame(X)
            self.fill = datfr.mode().iloc[0]
        return self

    def transform(self, X, y=None):
        xdf = pd.DataFrame(X)
        pdf = xdf.fillna(self.fill)
        return pdf


train_cat = CatImputer(strategy='mode').fit_transform(train_cat)

test_cat = CatImputer(strategy='mode').fit_transform(test_cat)
train_predictors
test_predictors[cat_attributes] = test_cat
train_predictors[cat_attributes] = train_cat

try1train = train_predictors[['v10', 'v12', 'v14', 'v21', 'v34', 'v38', 'v40', 'v50', 'v62', 'v72', 'v114',
                              'v129', 'v3', 'v24', 'v47', 'v52', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91', 'v107', 'v110', 'v112']]
try1test = test_predictors[['v10', 'v12', 'v14', 'v21', 'v34', 'v38', 'v40', 'v50', 'v62', 'v72', 'v114',
                            'v129', 'v3', 'v24', 'v47', 'v52', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91', 'v107', 'v110', 'v112']]

trynumattributes = ['v10', 'v12', 'v14', 'v21', 'v34',
                    'v38', 'v40', 'v50', 'v62', 'v72', 'v114', 'v129']


class CustomImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if np.isnan(X).any() == True:
            return MICE().complete(X)
        else:
            return np.c_[X]

# class CatImputer(BaseEstimator,TransformerMixin):
#     def fit(self, X, y=None):
#         self.fill = pd.Series([X[c].value_counts().index[0] for c in X],index=X.columns)
#         return self
#
#     def transform(self, X, y=None):
#         return X.fillna(self.fill)


class LabelEncoder_new(TransformerMixin, BaseEstimator):
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
            df = pd.DataFrame(X)
            lencoded = df.apply(lambda x: d[x.name].fit_transform(x))
            lencoded.columns = self.attribute_names
            return lencoded


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
    ('selector', DataFrameSelector(trynumattributes)),
    ('imputer', CustomImputer()),
    ('Scaling', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attributes)),
    ('label encoder', CustomOneHotEnocoder(cat_attributes)),
])

full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline),
])

trainprepared = full_pipeline.fit_transform(try1train)


# spca = TruncatedSVD(200)
#
# spca.fit(trainprepared)
#
# cumsum = np.cumsum(spca.explained_variance_ratio_)
#
# cumsum
#
# d = np.argmax(cumsum >= 0.95) + 1
#
# tpca = TruncatedSVD(n_components=200)
#
# train_prepared_SVD = tpca.fit_transform(trainprepared)

param_grid = {'n_estimators': [600, 700, 900], 'max_features': [6, 8, 10]}

gradient_clas = GradientBoostingClassifier(verbose=True)

grid_search = GridSearchCV(gradient_clas, param_grid,
                           cv=5, scoring='accuracy', verbose=True)


grid_search.fit(trainprepared, train_labels)

#final_model = grid_search.best_estimator_

final_model = grid_search.best_estimator_

grid_search.best_score_

test_prepared = full_pipeline.fit_transform(try1test)


#test_prepared_SVD = tpca.fit_transform(test_prepared)

final_predictions = final_model.predict_proba(test_prepared)

finaldftocsv = pd.DataFrame(final_predictions)

finaldftocsv.columns = ['PredictedProb0', 'PredictedProb1']

finaldftocsv.to_csv('BNPParibasProb.csv', index=False)
