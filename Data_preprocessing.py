import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, StandardScaler
import seaborn as sns
import pickle
import os
from typing import Tuple, Dict, Any


import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


class CustomPreprocessor:
    def __init__(self):
        self.log_cols = [
            'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss',
            'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin',
            'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit',
            'Sintpkt', 'Dintpkt', 'ct_state_ttl', 'ct_flw_http_mthd', 'ct_ftp_cmd',
            'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_dst_src_ltm'
        ]
        self.standard_cols = ['tcprtt', 'synack', 'ackdat']
        self.exclude_cols = ['Label', 'sport', 'dsport']

        self.robust_scaler = None
        self.standard_scaler = None
        self.fitted_log_cols = None
        self.fitted_standard_cols = None
        self.fitted_categorical_cols = None
        self.final_columns = None
        self.fitted = False

    def fit(self, X):
        """훈련 데이터로 변환 파라미터 학습"""

        # dsport 컬럼 형변환
        if 'dsport' in X.columns:
          X['dsport'] = pd.to_numeric(X['dsport'], errors='coerce').fillna(0)

        if 'sport' in X.columns:
          X['sport'] = pd.to_numeric(X['sport'], errors='coerce').fillna(0)

        # 실제 존재하는 컬럼만 선택
        self.fitted_log_cols = [col for col in self.log_cols if col in X.columns]
        self.fitted_standard_cols = [col for col in self.standard_cols if col in X.columns]

        # 범주형 컬럼 자동 감지
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        self.fitted_categorical_cols = [col for col in categorical_cols if col not in self.exclude_cols]

        # 로그 변환 적용
        X_log = self._apply_log_transform(X)

        # 스케일러 학습
        if self.fitted_log_cols:
            self.robust_scaler = RobustScaler()
            self.robust_scaler.fit(X_log[self.fitted_log_cols])

        if self.fitted_standard_cols:
            self.standard_scaler = StandardScaler()
            self.standard_scaler.fit(X_log[self.fitted_standard_cols])

        # 최종 컬럼 구조 파악
        self.final_columns = X.columns

        self.fitted = True
        return self

    def transform(self, X):
        """학습된 파라미터로 데이터 변환"""
        if not self.fitted:
            raise ValueError("fit() 먼저 호출 필요")

        X_transformed = X.copy()

        # 로그 변환
        X_transformed = self._apply_log_transform(X_transformed)

        # 스케일링
        if self.robust_scaler and self.fitted_log_cols:
            available_cols = [col for col in self.fitted_log_cols if col in X_transformed.columns]
            if available_cols:
                X_transformed[available_cols] = self.robust_scaler.transform(X_transformed[available_cols])

        if self.standard_scaler and self.fitted_standard_cols:
            available_cols = [col for col in self.fitted_standard_cols if col in X_transformed.columns]
            if available_cols:
                X_transformed[available_cols] = self.standard_scaler.transform(X_transformed[available_cols])

        # 원핫 인코딩
        X_transformed = self._apply_one_hot_encoding(X_transformed)

        # 컬럼 일관성 보장
        for col in self.final_columns:
            if col not in X_transformed.columns:
                X_transformed[col] = 0

        return X_transformed[self.final_columns]

    def fit_transform(self, X):
        """fit과 transform을 연속 수행"""
        return self.fit(X).transform(X)

    def _apply_log_transform(self, X):
        """로그 변환 적용"""
        X_copy = X.copy()
        for col in self.fitted_log_cols or []:
            if col in X_copy.columns:
                X_copy[col] = np.log1p(np.maximum(X_copy[col], 0))
        return X_copy

    def _apply_one_hot_encoding(self, X):
        """원핫 인코딩 적용"""
        if not self.fitted_categorical_cols:
            return X

        available_cols = [col for col in self.fitted_categorical_cols if col in X.columns]
        if available_cols:
            return pd.get_dummies(X, columns=available_cols, prefix=available_cols)
        return X



def visualize_results(train_before, train_after, test_before, test_after, sample_columns):
    """전처리 결과 시각화"""

    if not sample_columns:
        print("시각화할 컬럼이 없습니다.")
        return

    fig, axes = plt.subplots(2, len(sample_columns), figsize=(4*len(sample_columns), 8))
    if len(sample_columns) == 1:
        axes = axes.reshape(-1, 1)

    for i, col in enumerate(sample_columns):
        if col in train_before.columns and col in train_after.columns:
            # 전처리 전
            axes[0, i].hist(train_before[col].dropna(), bins=50, alpha=0.7,
                           label='Train Before', color='blue')
            axes[0, i].hist(test_before[col].dropna(), bins=50, alpha=0.7,
                           label='Test Before', color='red')
            axes[0, i].set_title(f'{col} - Before Preprocessing')
            axes[0, i].legend()
            axes[0, i].set_ylabel('Frequency')

            # 전처리 후
            axes[1, i].hist(train_after[col].dropna(), bins=50, alpha=0.7,
                           label='Train After', color='blue')
            axes[1, i].hist(test_after[col].dropna(), bins=50, alpha=0.7,
                           label='Test After', color='red')
            axes[1, i].set_title(f'{col} - After Preprocessing')
            axes[1, i].legend()
            axes[1, i].set_ylabel('Frequency')
            axes[1, i].set_xlabel('Value')

    plt.tight_layout()
    plt.show()

    # 데이터 형태 변화 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 데이터 크기 변화
    categories = ['Training', 'Test']
    before_counts = [train_before.shape[0], test_before.shape[0]]
    after_counts = [train_after.shape[0], test_after.shape[0]]

    x = np.arange(len(categories))
    width = 0.35

    ax1.bar(x - width/2, before_counts, width, label='Before', color='lightcoral')
    ax1.bar(x + width/2, after_counts, width, label='After', color='lightgreen')
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Sample Count Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()

    # 컬럼 수 변화
    before_cols = [train_before.shape[1], test_before.shape[1]]
    after_cols = [train_after.shape[1], test_after.shape[1]]

    ax2.bar(x - width/2, before_cols, width, label='Before', color='lightcoral')
    ax2.bar(x + width/2, after_cols, width, label='After', color='lightgreen')
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Number of Features')
    ax2.set_title('Feature Count Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()

    plt.tight_layout()
    plt.show()


# 실행 예제
if __name__ == "__main__":
    # 데이터 로드
    train_data = pd.read_csv("/DS_term/2_Cleaning_DataSet.csv", low_memory=False)
    test_data = pd.read_csv("/DS_term/2_Cleaning_TestDataSet.csv", low_memory=False)

    # 전처리 수행
    preprocessor = CustomPreprocessor()
    train_processed = preprocessor.fit_transform(train_data)
    test_processed = preprocessor.transform(test_data)

    # 결과 시각화
    visualize_results(train_data, train_processed, test_data, test_processed, ['sttl', 'dttl', 'Sload', 'Dload', 'ct_state_ttl'])

    print("컬럼 변동 (train) : ", end='')
    print(set(train_data) - set(train_processed))

    print("컬럼 변동 (test) : ", end='')
    print(set(test_data) - set(test_processed))

    # 결과 저장
    train_processed.to_csv("/DS_term/3_Preprocessing_DataSet.csv", index=False)
    test_processed.to_csv("/DS_term/3_Preprocessing_TestDataSet.csv", index=False)

    print("전처리 완료 및 결과 저장됨")
