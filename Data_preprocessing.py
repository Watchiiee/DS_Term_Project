import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
import seaborn as sns

def data_scaling(is_plot=True):
  # 1. 데이터 로드
  df = pd.read_csv("DS_term/2_Cleaning_DataSet.csv")

  # 2. 정상/공격 분리
  df_normal = df[df['Label'] == 0].copy()
  df_attack = df[df['Label'] == 1].copy()

  df_normal_numeric = df_normal.select_dtypes(include=['number']).drop(columns=['Label'], errors='ignore').copy()
  if is_plot:
    df_normal_numeric.hist(figsize=(20, 15), bins=50)
    plt.tight_layout()
    plt.show()

  # 3. 로그 변환 대상
  log_transform_cols = [
      'dur', 'sbytes', 'dbytes', 'sloss', 'dloss', 'Sload', 'Dload',
      'Spkts', 'Dpkts', 'smeansz', 'dmeansz',
      'Sjit', 'Djit', 'Sintpkt', 'Dintpkt'
  ]

  df_normal[log_transform_cols] = df_normal[log_transform_cols].apply(lambda x: np.log1p(x))
  df_attack[log_transform_cols] = df_attack[log_transform_cols].apply(lambda x: np.log1p(x))
  df[log_transform_cols] = df[log_transform_cols].apply(lambda x: np.log1p(x))

  # 4. 스케일링: RobustScaler
  robust_scaler = RobustScaler()
  robust_scaler.fit(df_normal[log_transform_cols])

  df[log_transform_cols] = robust_scaler.transform(df[log_transform_cols])

  # 5. StandardScaler 대상
  standard_cols = ['tcprtt', 'synack', 'ackdat']
  standard_scaler = StandardScaler()
  standard_scaler.fit(df_normal[standard_cols])

  df[standard_cols] = standard_scaler.transform(df[standard_cols])

  # 6. MinMaxScaler 대상
  minmax_cols = ['trans_depth', 'res_bdy_len', 'ct_flw_http_mthd', 'ct_ftp_cmd']
  minmax_scaler = MinMaxScaler()
  minmax_scaler.fit(df_normal[minmax_cols])

  df[minmax_cols] = minmax_scaler.transform(df[minmax_cols])


  # 범주형 컬럼 자동 추출
  categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
  # 'Label' 컬럼 제외
  categorical_cols = [col for col in categorical_cols if col != 'Label']
  # 원-핫 인코딩
  df_final= pd.get_dummies(df, columns=categorical_cols)

  # 11. 저장
  df_final.to_csv("DS_term/3_Preprocessing_DataSet.csv", index=False)

  if is_plot:
    print(df_final.dtypes)

    scaled_cols = log_transform_cols + standard_cols + minmax_cols
    df[scaled_cols].hist(figsize=(20, 15), bins=50)
    plt.tight_layout()
    plt.show()

  # 12. 테스트 데이터 스케일링
  df_test = pd.read_csv("DS_term/2_Cleaning_TestDataSet.csv")
  df_test[log_transform_cols] = robust_scaler.transform(df_test[log_transform_cols])
  df_test[standard_cols] = standard_scaler.transform(df_test[standard_cols])
  df_test[minmax_cols] = minmax_scaler.transform(df_test[minmax_cols])
  df_test_final = pd.get_dummies(df_test, columns=categorical_cols)
  df_test_final.to_csv("DS_term/3_Preprocessing_TestDataSet.csv", index=False)

if __name__ == '__main__':
  data_scaling()