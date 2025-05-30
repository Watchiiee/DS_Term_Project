import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class CustomFeatureEngineer:
    def __init__(self, top_k=10, corr_threshold=0.8, target_corr_min=0.3, show_plots=False):
        self.original_features = None
        self.selected_features = None
        self.tree_based_importances = None
        self.top_k = top_k
        self.corr_threshold = corr_threshold
        self.target_corr_min = target_corr_min
        self.show_plots = show_plots
        self.et_model = None

    def _create_features(self, X):
        """새로운 피쳐 생성"""
        X_new = X.copy()

        # 기본 피쳐 생성
        if 'Sload' in X.columns and 'Dload' in X.columns:
            X_new['load_ratio'] = X_new['Sload'] / (X_new['Dload'] + 1e-6)

        return X_new

    def _plot_feature_importance(self, importance_series, title="Tree-based Feature Importance"):
        """Tree-based 피쳐 중요도 시각화"""
        plt.figure(figsize=(12, 8))

        # 수평 막대 그래프
        importance_series.plot(kind='barh', color='lightblue')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Feature Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.grid(axis='x', alpha=0.3)

        # 각 막대에 수치 표시
        for i, v in enumerate(importance_series.values):
            plt.text(v + max(importance_series.values) * 0.01, i, f'{v:.4f}',
                    va='center', fontsize=10)

        plt.tight_layout()
        plt.show()

    def fit(self, X, y):
        """피쳐 엔지니어링 파이프라인 학습"""
        print("=== 피쳐 선택 시작 ===")

        # 원본 피쳐 저장
        self.original_features = X.columns.tolist()

        # 피쳐 생성
        X_engineered = self._create_features(X)
        print(f"피쳐 생성 후: {X_engineered.shape[1]}개")

        # 1단계: Tree-based 피쳐 중요도 계산
        self.et_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
        self.et_model.fit(X_engineered, y)

        self.tree_based_importances = pd.Series(
            self.et_model.feature_importances_,
            index=X_engineered.columns
        ).sort_values(ascending=False)

        print(f"\n=== Tree-based 피쳐 중요도 (상위 {self.top_k}개) ===")
        top_features = self.tree_based_importances.head(self.top_k)
        print(top_features)

        # Tree-based 중요도 시각화 (조건부)
        if self.show_plots:
            self._plot_feature_importance(
                top_features,
                f"Tree-based Feature Importance (Top {self.top_k})"
            )

        # 2단계: 타겟과의 상관관계 필터링 및 피쳐 간 상관관계 처리
        self.selected_features = self._simple_correlation_filtering(
            X_engineered, y, top_features.index.tolist()
        )

        print(f"\n=== 최종 선택된 피쳐 ({len(self.selected_features)}개) ===")
        for feat in self.selected_features:
            print(f"{feat}: {self.tree_based_importances[feat]:.4f}")

        return self

    def _simple_correlation_filtering(self, X, y, top_features):
        """상관관계 기반 필터링"""
        # 상관관계 매트릭스 계산
        df_corr = X[top_features].copy()
        df_corr['Label'] = y
        corr_matrix = df_corr.corr().round(2)

        # 히트맵 시각화 (조건부)
        if self.show_plots:
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".3f",
                       square=True, cbar=True)
            plt.title(f'Correlation Matrix (k = {self.top_k})', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()

        # 타겟과의 상관관계 확인
        target_correlations = corr_matrix['Label'].drop('Label').abs()

        # 타겟 상관관계 임계점 이상인 피쳐들만 유지
        qualified_features = target_correlations[
            target_correlations >= self.target_corr_min
        ].index.tolist()

        print(f"\n=== 타겟 상관관계 분석 (임계점: {self.target_corr_min}) ===")
        for feat in qualified_features:
            print(f"{feat}: {target_correlations[feat]:.3f}")

        # 피쳐 간 높은 상관관계 처리
        feature_corr_matrix = corr_matrix.drop('Label', axis=0).drop('Label', axis=1)
        selected_features = self._remove_high_correlation_features(
            qualified_features, feature_corr_matrix
        )

        return selected_features

    def _remove_high_correlation_features(self, features, corr_matrix):
        """높은 상관관계를 가진 피쳐 중 하나만 선택"""
        features_to_remove = set()
        processed_pairs = set()

        print(f"\n=== 피쳐 간 상관관계 분석 (임계점: {self.corr_threshold}) ===")

        for i, feat1 in enumerate(features):
            for j, feat2 in enumerate(features):
                if i < j and feat1 not in features_to_remove and feat2 not in features_to_remove:
                    pair = tuple(sorted([feat1, feat2]))
                    if pair not in processed_pairs:
                        corr_val = abs(corr_matrix.loc[feat1, feat2])

                        if corr_val > self.corr_threshold:
                            print(f"{feat1} - {feat2}: {corr_val:.3f}")

                            # Tree 중요도가 낮은 피쳐 제거
                            if self.tree_based_importances[feat1] < self.tree_based_importances[feat2]:
                                features_to_remove.add(feat1)
                                print(f"  → {feat1} 제거 (중요도: {self.tree_based_importances[feat1]:.4f})")
                            else:
                                features_to_remove.add(feat2)
                                print(f"  → {feat2} 제거 (중요도: {self.tree_based_importances[feat2]:.4f})")

                        processed_pairs.add(pair)

        # 최종 선택된 피쳐들
        selected_features = [f for f in features if f not in features_to_remove]

        if features_to_remove:
            print(f"\n총 제거된 피쳐: {list(features_to_remove)}")
        else:
            print(f"\n높은 상관관계를 가진 피쳐 쌍이 없음")

        return selected_features

    def transform(self, X):
        """학습된 파이프라인을 사용하여 데이터 변환"""
        if self.selected_features is None:
            raise ValueError("모델을 먼저 학습시켜야 합니다. fit() 메서드를 호출하세요.")

        # 피쳐 생성
        X_engineered = self._create_features(X)

        # 선택된 피쳐만 반환
        return X_engineered[self.selected_features]

    def fit_transform(self, X, y):
        """학습과 변환을 동시에 수행"""
        return self.fit(X, y).transform(X)

    def get_feature_names(self):
        """선택된 피쳐 이름 반환"""
        return self.selected_features

    def get_feature_importances(self):
        """선택된 피쳐의 중요도 반환"""
        if self.selected_features is None or self.tree_based_importances is None:
            return None
        return self.tree_based_importances[self.selected_features]

    def get_summary(self):
        """피쳐 선택 결과 요약"""
        if self.selected_features is None:
            return "모델이 아직 학습되지 않았습니다."

        summary = f"""
=== 피쳐 선택 요약 ===
원본 피쳐 수: {len(self.original_features) if self.original_features else 'N/A'}
최종 선택된 피쳐 수: {len(self.selected_features)}
선택 기준:
  - Tree-based 상위 K개: {self.top_k}
  - 타겟 상관관계 최소: {self.target_corr_min}
  - 피쳐 간 상관관계 임계점: {self.corr_threshold}

선택된 피쳐:
"""
        for feat in self.selected_features:
            importance = self.tree_based_importances[feat]
            summary += f"  - {feat}: {importance:.4f}\n"

        return summary

if __name__ == '__main__':
    df_train, df_test = pd.read_csv("/DS_term/3_Preprocessing_DataSet.csv", low_memory=False), pd.read_csv("/DS_term/3_Preprocessing_TestDataSet.csv", low_memory=False)
    X_train, X_test = df_train.drop(["Label"], axis=1), df_test.drop(["Label"], axis=1)
    y_train, y_test = df_train["Label"], df_test["Label"]

    fe = CustomFeatureEngineer(
        top_k=10,
        corr_threshold=0.8,
        target_corr_min=0.3,
        show_plots=True
    )

    # 학습 및 변환
    X_transformed = fe.fit_transform(X_train, y_train)
    X_transformed["Label"] = y_train

    # 테스트 데이터 변환
    X_test_transformed = fe.transform(X_test)
    X_transformed["Label"] = y_test

    # 결과 확인
    print(fe.get_summary())
    selected_features = fe.get_feature_names()

    # 결과 저장
    X_transformed.to_csv("/DS_term/4_Feature_engineering_DataSet.csv")
    X_test_transformed.to_csv("/DS_term/4_Feature_engineering_TestDataSet.csv")
