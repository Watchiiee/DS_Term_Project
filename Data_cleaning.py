import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

def data_cleaning(df_train, df_test, is_plot=False):
    # 원본 데이터 보존을 위한 복사본 생성
    df_train_clean = df_train.copy()
    df_test_clean = df_test.copy()

    # 결측값 분석 (훈련 데이터 기준)
    null_counts = df_train_clean.isnull().sum()
    print("훈련 데이터의 결측값:")
    print(null_counts[null_counts > 0])

    # 1. 결측값 처리 - 동일한 방식으로 양쪽 데이터에 적용
    def handle_missing_values(df):
        df['ct_flw_http_mthd'] = df['ct_flw_http_mthd'].fillna(0).astype(np.uint8)
        df['is_ftp_login'] = df['is_ftp_login'].fillna(0).astype(np.uint8)
        df.replace({'ct_ftp_cmd': {' ': np.nan}}, regex=True, inplace=True)
        df['ct_ftp_cmd'] = pd.to_numeric(df['ct_ftp_cmd'], errors='coerce')
        df['ct_ftp_cmd'] = df['ct_ftp_cmd'].fillna(0).astype('Int64')
        return df

    df_train_clean = handle_missing_values(df_train_clean)
    df_test_clean = handle_missing_values(df_test_clean)

    print("-----")
    print("훈련 데이터 ct_ftp_cmd 처리 결과:")
    print(df_train_clean['ct_ftp_cmd'].unique())
    print(df_train_clean['ct_ftp_cmd'].head(10))
    print(df_train_clean['ct_ftp_cmd'].dtype)

    # 2. 데이터 개수 기록
    original_train_count = df_train_clean.shape[0]
    original_test_count = df_test_clean.shape[0]
    print(f"훈련 데이터 원본: {original_train_count}")
    print(f"테스트 데이터 원본: {original_test_count}")

    # 3. 중복 데이터 제거
    train_duplicates = df_train_clean.duplicated().sum()
    test_duplicates = df_test_clean.duplicated().sum()
    print(f"훈련 데이터 중복: {train_duplicates}")
    print(f"테스트 데이터 중복: {test_duplicates}")

    df_train_clean = df_train_clean.drop_duplicates()
    df_test_clean = df_test_clean.drop_duplicates()

    # 4. 이진 컬럼 타입 최적화 - 양쪽 데이터에 적용
    def optimize_binary_columns(df):
        for col in df.columns:
            unique_vals = df[col].dropna().unique()
            if set(unique_vals).issubset({0, 1}):
                df[col] = df[col].astype(np.uint8)
        return df

    df_train_clean = optimize_binary_columns(df_train_clean)
    df_test_clean = optimize_binary_columns(df_test_clean)

    # 5. 이상치 처리 - 훈련 데이터에서 기준 계산, 양쪽에 적용
    # outlier_cols = ['tcprtt', 'synack', 'ackdat']
    outlier_cols = []
    outlier_stats = {}

    print("\n이상치 처리 기준 (훈련 데이터 기준):")
    for col in outlier_cols:
        if col in df_train_clean.columns:
            q1 = df_train_clean[col].quantile(0.25)
            q3 = df_train_clean[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outlier_stats[col] = {'lower': lower, 'upper': upper}
            print(f"{col}: [{lower:.2f}, {upper:.2f}]")

    # 이상치 제거 적용
    train_before_outlier = df_train_clean.shape[0]

    for col, bounds in outlier_stats.items():
        if col in df_train_clean.columns:
            df_train_clean = df_train_clean[
                (df_train_clean[col] >= bounds['lower']) &
                (df_train_clean[col] <= bounds['upper'])
            ]

    train_outliers_removed = train_before_outlier - df_train_clean.shape[0]

    print(f"훈련 데이터 이상치 제거: {train_outliers_removed}개")

    # 6. 불필요한 컬럼 제거
    columns_to_drop = ['srcip', 'dstip', 'Stime', 'Ltime', 'stcpb', 'dtcpb']
    df_train_clean = df_train_clean.drop(columns=columns_to_drop, errors='ignore')
    df_test_clean = df_test_clean.drop(columns=columns_to_drop, errors='ignore')

    # 7. 최종 결과 출력
    final_train_count = df_train_clean.shape[0]
    final_test_count = df_test_clean.shape[0]

    print(f"\n=== 데이터 클리닝 완료 ===")
    print(f"훈련 데이터: {original_train_count} → {final_train_count} "
          f"({final_train_count/original_train_count*100:.1f}% 유지)")
    print(f"테스트 데이터: {original_test_count} → {final_test_count} "
          f"({final_test_count/original_test_count*100:.1f}% 유지)")

    # 8. 시각화 (선택사항)
    if is_plot:
        create_cleaning_plot(original_train_count, final_train_count,
                           original_test_count, final_test_count,
                           train_duplicates + train_outliers_removed,
                           test_duplicates)

    return df_train_clean, df_test_clean


def create_cleaning_plot(orig_train, final_train, orig_test, final_test, train_removed, test_removed):
    """데이터 클리닝 결과를 시각화하는 함수"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 훈련 데이터 변화
    train_labels = ['Removed', 'Maintained']
    train_sizes = [train_removed, final_train]
    colors = ['lightcoral', 'lightgreen']

    ax1.pie(train_sizes, labels=train_labels, colors=colors,
            autopct='%1.1f%%', startangle=140)
    ax1.set_title(f'Train Set Cleaning Result\n(Origin #: {orig_train:,})')

    # 테스트 데이터 변화
    test_labels = ['Removed', 'Maintained']
    test_sizes = [test_removed, final_test]

    ax2.pie(test_sizes, labels=test_labels, colors=colors,
            autopct='%1.1f%%', startangle=140)
    ax2.set_title(f'Test Set Cleaning Result\n(Origin #: {orig_test:,})')

    plt.tight_layout()
    plt.show()


# 메인 실행 코드
if __name__ == "__main__":
    # 데이터 로드
    print("데이터 로딩 중...")
    df = pd.read_csv("/DS_term/1_Raw_DataSet.csv", low_memory=False)

    df_train, df_test = train_test_split(
        df,
        test_size = 0.3,
        shuffle = True,
        random_state = 42,
        stratify = df['Label']
    )

    print(f"원본 훈련 데이터 크기: {df_train.shape}")
    print(f"원본 테스트 데이터 크기: {df_test.shape}")
    print("-" * 50)

    # 데이터 클리닝 실행
    df_train_clean, df_test_clean = data_cleaning(df_train, df_test, is_plot=True)

    # 결과 확인
    print("\n클리닝 완료!")
    print(f"정제된 훈련 데이터 크기: {df_train_clean.shape}")
    print(f"정제된 테스트 데이터 크기: {df_test_clean.shape}")

    # 파일 저장
    df_train_clean.to_csv("/DS_term/2_Cleaning_DataSet.csv", index=False)
    df_test_clean.to_csv("/DS_term/2_Cleaning_TestDataSet.csv", index=False)
