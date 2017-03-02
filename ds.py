# Martin Kersner, m.kersner@gmail.com
# 2017/03/02

# Author: meikegw
# https://www.kaggle.com/meikegw/house-prices-advanced-regression-techniques/filling-up-missing-values
def show_missing(df):
  missing = df.columns[df.isnull().any()].tolist()
  return missing

def show_missing_stat(df):
  return df[show_missing(df)].isnull().sum()

def remove_missing(df, colname):
  return df[~df[colname].isnull()]
