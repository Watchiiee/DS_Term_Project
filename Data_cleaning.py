import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def remove_outliers_iqr(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include='number').columns

    df_clean = df.copy()

    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    return df_clean

def data_cleaning(df, is_test=False, is_plot=True):
  # 해당 컬럼의 결측치 0으로 대체
  df['ct_flw_http_mthd'] = df['ct_flw_http_mthd'].fillna(0).astype(np.uint8)
  df['is_ftp_login'] = df['is_ftp_login'].fillna(0).astype(np.uint8)
  df['ct_ftp_cmd'].replace(' ', np.nan, regex=True, inplace=True)
  df['ct_ftp_cmd'] = pd.to_numeric(df['ct_ftp_cmd'], errors='coerce')
  df['ct_ftp_cmd'] = df['ct_ftp_cmd'].fillna(0).astype('Int64')

  print("-----")
  print(df['ct_ftp_cmd'].unique())
  print(df['ct_ftp_cmd'].head(10))
  print(df['ct_ftp_cmd'].dtype)

  # 원본 데이터 행 개수 저장
  original_count = df.shape[0]

  # 중복 데이터 확인 / 제거
  if not is_test:
    duplicates_count = df.duplicated().sum()
    print(duplicates_count)
    df = df.drop_duplicates()

    print("중복 처리 이후")
    print(df.shape[0])

  # 0, 1 로 이루어진 범주형 변수 uint8 타입으로 변환
  for col in df.columns:
      unique_vals = df[col].dropna().unique()
      if set(unique_vals).issubset({0, 1}):
          df[col] = df[col].astype(np.uint8)

  df_final = None
  if not is_test:
    df_normal = df[df['Label'] == 0]
    df_attack = df[df['Label'] == 1]

    print("아웃라이어 처리 이전")
    print(df_normal.shape[0])

    before_outlier_count = df_normal.shape[0]

    df_normal_cleaned = remove_outliers_iqr(df_normal) # 아웃 라이어 처리

    after_outlier_count = df_normal_cleaned.shape[0]

    print("아웃라이어 처리 이후")
    print(df_normal_cleaned.shape[0])

    df_final = pd.concat([df_attack, df_normal_cleaned])
  else:
    df_final = df

  # 전처리 단계에서 제거할 피처
  columns_to_drop = ['srcip', 'dstip', 'Stime', 'Ltime', 'stcpb', 'dtcpb']
  df_final = df_final.drop(columns=columns_to_drop, errors='ignore')

  if is_test:
    df_final.to_csv("DS_term/2_Cleaning_TestDataSet.csv", index=False)
  else:
    df_final.to_csv("DS_term/2_Cleaning_DataSet.csv", index=False)

  if is_plot:
    outliers_removed = before_outlier_count - after_outlier_count
    final_remain = original_count - (duplicates_count + outliers_removed)

    labels = ['duplicates', 'outliers', 'remain']
    sizes = [duplicates_count, outliers_removed, final_remain]
    colors = ['lightcoral', 'lightgray', 'lightskyblue']

    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title('Data Cleaning')
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
  raw_dataset = pd.read_csv("DS_term/1_Raw_DataSet.csv")

  # 결측치 개수 확인
  null_counts = raw_dataset.isnull().sum()
  print(null_counts[null_counts > 0])

  df_train, processed_test = train_test_split(
      raw_dataset,
      test_size = 0.3,
      random_state = 128,
      stratify=raw_dataset['Label']
  )
  df_train.reset_index(drop=True, inplace=True)
  processed_test.reset_index(drop=True, inplace=True)

  # Train-Test 데이터셋 분리
  data_cleaning(df_train)
  data_cleaning(processed_test, is_test=True, is_plot=False)
