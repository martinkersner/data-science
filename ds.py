# Martin Kersner, m.kersner@gmail.com
# 2017/03/02

# Prerequisities:
# * pandas
# * matplotlib
# * scikit-learn

# Author: meikegw
# https://www.kaggle.com/meikegw/house-prices-advanced-regression-techniques/filling-up-missing-values
def show_missing(df):
  missing = df.columns[df.isnull().any()].tolist()
  return missing

def show_missing_stat(df):
  return df[show_missing(df)].isnull().sum()

def remove_missing(df, colname):
  return df[~df[colname].isnull()]

# Author: meikegw
# https://www.kaggle.com/meikegw/house-prices-advanced-regression-techniques/filling-up-missing-values
def cat_exploration(df, column):
  ''' Looking at categorical values '''
  return df[column].value_counts()

# Author: meikegw
# https://www.kaggle.com/meikegw/house-prices-advanced-regression-techniques/filling-up-missing-values
def cat_imputation(df, column, value):
  ''' Imputing the missing values '''
  df.loc[df[column].isnull(), column] = value

def plot_roc(y_test, y_pred):
  '''
  Print ROC curve from given true labels and positive prediction probabilities.

  Example usage:
  y_pred = clf.predict_proba(X)
  plot_roc(y, y_pred[:, 1])
  '''
  from sklearn.metrics import roc_curve, auc
  false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
  roc_auc = auc(false_positive_rate, true_positive_rate)

  plt.title('Receiver Operating Characteristic')
  plt.plot(false_positive_rate, true_positive_rate, 'b',
  label='AUC = %0.2f'% roc_auc)
  plt.legend(loc='lower right')
  plt.plot([0,1],[0,1],'r--')
  plt.tight_layout()
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.show()

def hist_plot(hist_dict):
  values = list(hist_dict.values())
  labels = list(hist_dict.keys())

  x_idxs = range(len(values))

  plt.bar(x_idxs, values, align='center')
  plt.xticks(x_idxs, labels)
  plt.show()
