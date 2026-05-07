import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def create_preprocessor(df):
    ## separating numeric and object columns
    num_col = df.select_dtypes(include = ['number']).columns
    cat_col = df.select_dtypes(include = ['object']).columns

    ##numeric pipeline
    num_pipe = Pipeline(steps = [
        ('imputer',SimpleImputer(strategy = 'mean')),
        ('scaler', StandardScaler())
    ])

    ##category pipeline
    cat_pipe = Pipeline(steps = [
        ('imputer',SimpleImputer(strategy ='most_frequent')),
        ('encoder',OneHotEncoder(handle_unknown = 'ignore'))
    ])

    ##combining both pipelines
    preprocessor = ColumnTransformer(transformers = [
        ('num',num_pipe,num_col),
        ('cat',cat_pipe,cat_col)
    ])

    return preprocessor