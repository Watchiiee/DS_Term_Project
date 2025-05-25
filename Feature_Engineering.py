import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

def data_preprocessing(is_plot=True):
  # Load the dataset
  df = pd.read_csv("DS_term/3_Preprocessing_DataSet.csv")
  df_test = pd.read_csv("DS_term/3_Preprocessing_TestDataSet.csv") # 테스트 데이터셋 로드

  # feature creation - load_ratio
  df['load_ratio'] = df['Sload'] / (df['Dload'] + 1e-6)
  df_test['load_ratio'] = df_test['Sload'] / (df_test['Dload'] + 1e-6)

  # feature creation - ttl_diff
  df['ttl_diff'] = abs(df['sttl'] - df['ct_state_ttl'])
  df_test['ttl_diff'] = abs(df_test['sttl'] - df_test['ct_state_ttl'])

  target_col = 'Label'

  # Identify numerical and categorical columns
  numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
  categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

  # Define prefixes of one-hot encoded columns to exclude
  exclude_prefixes = ('proto_', 'state_', 'service_')

  # Get all one-hot encoded columns based on prefixes
  excluded_cols = [col for col in df.columns if col.startswith(exclude_prefixes)]

  # Logistic Regression-based feature selection using L1 regularization
  X = df[excluded_cols].copy()
  target = df[target_col]

  model = LogisticRegression(penalty='l1', solver='liblinear', C=0.05, class_weight='balanced', max_iter=1000, random_state=42)
  model.fit(X, target)
  selected_features = np.array(X.columns)[model.coef_[0] != 0]

  print(f"Number of selected features: {len(selected_features)}")
  print("Selected feature list:", selected_features)

  # Create DataFrame with selected features + target
  df_LogisticRegression_selected = df[list(selected_features) + [target_col]]

  # Correlation-based feature selection for numerical features
  corr_target_cols = [col for col in numeric_cols if col not in excluded_cols or col == target_col]
  correlations = df[corr_target_cols].corr()[target_col].abs()
  selected_numeric = correlations[correlations > 0.3].index.tolist()

  # Plot correlation heatmap for selected numeric features
  if is_plot:
    corr_matrix = df[selected_numeric].corr().round(2)
    plt.figure(figsize=(30, 20))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True, annot_kws={"size": 8})
    plt.title("Correlation Matrix (Filtered Features Only)")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("correlation.png", dpi=300, bbox_inches='tight')
    plt.show()

  # Apply variance threshold to remove low-variance features
  df_numeric_selected = df[selected_numeric].drop(columns=[target_col])
  print("Number of features before variance filtering:", df_numeric_selected.shape[1])

  selector = VarianceThreshold(threshold=0.01)
  selected_variance_data = selector.fit_transform(df_numeric_selected)
  selected_variance_features = df_numeric_selected.columns[selector.get_support(indices=True)]
  df_numeric_selected = pd.DataFrame(selected_variance_data, columns=selected_variance_features)

  # Print variance values of selected features
  variances = df_numeric_selected.var()
  print(variances.sort_values())
  print("Number of features after variance filtering:", df_numeric_selected.shape[1])

  # Combine numeric features and Logistic Regression-selected features
  df_filtered = pd.concat([df_numeric_selected, df_LogisticRegression_selected], axis=1)

  # Save the final feature-engineered dataset
  df_filtered.to_csv('DS_term/4_Feature_engineering_DataSet.csv', index=False)

  print(df_filtered.columns.tolist())

  # 테스트 데이터셋 피쳐 추출
  df_test_filtered = df_test[df_filtered.columns]
  df_filtered.to_csv('DS_term/4_Feature_engineering_TestDataSet.csv', index=False)
  
if __name__ == '__main__':
  data_preprocessing()