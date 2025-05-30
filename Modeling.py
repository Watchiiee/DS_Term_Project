import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# 머신러닝 라이브러리
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    KFold,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# 시각화 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# 경고 메시지 숨기기
import warnings

warnings.filterwarnings("ignore")


class ModelingPipeline:
    """
    모델링 파이프라인 클래스
    """

    def __init__(self, test_size=0.3):
        self.data_path="/DS_term/1_Raw_DataSet.csv"
        self.df = None
        self.test_size = test_size
        self.results = {}

    def load_data(self):
        """원본 데이터를 불러오기"""
        print("\n" + "=" * 50)
        print("데이터 불러오기")
        print("=" * 50)

        try:
            self.data_path = "/DS_term/1_Raw_DataSet.csv"
            self.df = pd.read_csv(self.data_path, low_memory=False)
            print(f"데이터 로드 완료: {self.df.shape[0]}행 x {self.df.shape[1]}열")
            return True
        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다: {self.data_path}")
            return False
        except Exception as e:
            print(f"데이터 로드 중 오류 발생: {e}")
            return False

    def preprocess(self, df_train=None, df_test=None):
        """데이터 전처리"""
        with open(os.devnull, "w") as fnull:
            original_stdout = sys.stdout
            sys.stdout = fnull
            try:

                # 데이터 클리닝
                df_train_clean, df_test_clean = data_cleaning(df_train, df_test, is_plot=False)

                # 데이터 전처리
                prep = CustomPreprocessor()
                prep_train = prep.fit_transform(df_train_clean)
                prep_test = prep.transform(df_test_clean)

                del df_train_clean, df_test_clean # 클리닝 데이터 캐시 제거

                X_train, X_test = prep_train.drop(["Label"], axis=1), prep_test.drop(["Label"], axis=1)
                y_train, y_test = prep_train["Label"], prep_test["Label"]

                del prep_train, prep_test # 전처리 데이터 캐시 제거

                # 특징 선택
                fe = CustomFeatureEngineer(
                    top_k=10,
                    corr_threshold=0.8,
                    target_corr_min=0.3,
                    show_plots=False
                )

                # 학습 및 변환
                X_transformed = fe.fit_transform(X_train, y_train)
                X_transformed["Label"] = y_train

                # 테스트 데이터 변환
                X_test_transformed = fe.transform(X_test)
                X_test_transformed["Label"] = y_test

                return X_transformed, X_test_transformed

            finally:
                sys.stdout = original_stdout

    # ========== 예측 함수 ==========
    def linear_regression_predict(self, X_train, X_test, y_train, y_test):
        """
        선형 회귀를 사용한 분류 예측
        Args:
            X_train, X_test: 훈련/테스트 특성 데이터
            y_train, y_test: 훈련/테스트 타겟 데이터
        Returns:
            tuple: (예측값, 모델 객체, 최적 임계값)
        """
        # 모델 생성 및 훈련
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 예측
        y_pred_continuous = model.predict(X_test)

        # 예측값 범위 확인
        pred_min, pred_max = y_pred_continuous.min(), y_pred_continuous.max()

        # 다양한 임계값에서 성능 평가
        best_metric = 0
        best_threshold = 0.5
        best_predictions = None

        # 예측값 범위 내에서 임계값 후보 생성
        threshold_candidates = np.linspace(pred_min, pred_max, 20)

        for thresh in threshold_candidates:
            # 현재 임계값으로 분류
            y_pred_temp = np.where(y_pred_continuous >= thresh, 1, 0)

            # 지표 계산
            accuracy = accuracy_score(y_test, y_pred_temp)
            precision = precision_score(y_test, y_pred_temp, zero_division=0)
            recall = recall_score(y_test, y_pred_temp, zero_division=0)
            f1 = f1_score(y_test, y_pred_temp, zero_division=0)

            # 최고 성능 임계값 업데이트
            if accuracy*0.2 + precision*0.2 + recall*0.4 + f1*0.2 > best_metric:
                best_accuracy = accuracy
                best_threshold = thresh
                best_predictions = y_pred_temp

            print(f"\n임계값: {thresh}\n가중치된 지표: Accuracy({accuracy:.4f}) + Precision({precision:.4f}) + Recall({recall:.4f}) + F1({f1:.4f})")

        return best_predictions, model, best_threshold

    def logistic_regression_predict(self, X_train, X_test, y_train, y_test):
        """
        로지스틱 회귀 예측

        Returns:
            tuple: (예측값, 확률값, 모델 객체)
        """
        model = LogisticRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        return y_pred, y_pred_proba, model

    def random_forest_predict(self, X_train, X_test, y_train, y_test):
        """
        랜덤 포레스트 예측

        Returns:
            tuple: (예측값, 확률값, 모델 객체)
        """
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        return y_pred, y_pred_proba, model

    def decision_tree_predict(self, X_train, X_test, y_train, y_test):
        """
        의사결정트리 예측

        Returns:
            tuple: (예측값, 확률값, 모델 객체)
        """
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        return y_pred, y_pred_proba, model

    def PCA_regression_predict(self, X_train, X_test, y_train, y_test, n_components=1):
        """
        PCA 기반 차원축소 후 분류 예측

        Args:
            X_train, X_test: 훈련/테스트 특성 데이터
            y_train, y_test: 훈련/테스트 타겟 데이터
            n_components: PCA 주성분 개수

        Returns:
            tuple: (예측값, PCA 객체, 최적 임계값, 성능 지표)
        """

        # 1단계: 훈련 데이터로 PCA 학습
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)  # 훈련 데이터로 학습 및 변환

        # 2단계: 훈련 데이터에서 최적 임계값 찾기
        if n_components == 1:
            train_pca_values = X_train_pca.flatten()
        else:
            train_pca_values = X_train_pca[:, 0]

        # 훈련 데이터 범위에서 임계값 후보 생성
        pca_min, pca_max = train_pca_values.min(), train_pca_values.max()
        threshold_candidates = np.linspace(pca_min, pca_max, 20)

        best_composite_score = 0
        best_threshold = 0.5

        for thresh in threshold_candidates:
            # 훈련 데이터로 분류 성능 평가
            y_pred_train = np.where(train_pca_values >= thresh, 1, 0)

            # 훈련 데이터 기준 성능 지표 계산
            accuracy = accuracy_score(y_train, y_pred_train)
            precision = precision_score(y_train, y_pred_train, zero_division=0)
            recall = recall_score(y_train, y_pred_train, zero_division=0)
            f1 = f1_score(y_train, y_pred_train, zero_division=0)

            # 가중 복합 점수 계산
            composite_score = accuracy*0.2 + precision*0.2 + recall*0.4 + f1*0.2

            if composite_score > best_composite_score:
                best_composite_score = composite_score
                best_threshold = thresh

            print(f"\n임계값: {thresh}\n가중치된 지표: Accuracy({accuracy:.4f}) + Precision({precision:.4f}) + Recall({recall:.4f}) + F1({f1:.4f})")

        # 3단계: 테스트 데이터 변환 및 예측
        X_test_pca = pca.transform(X_test)  # 학습된 PCA로 테스트 데이터 변환

        if n_components == 1:
            test_pca_values = X_test_pca.flatten()
        else:
            test_pca_values = X_test_pca[:, 0]

        # 찾은 임계값으로 테스트 데이터 예측
        y_pred = np.where(test_pca_values >= best_threshold, 1, 0)

        return y_pred, pca, best_threshold


    # ========== 지표 계산 함수 ==========
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        성능 지표 계산

        Args:
            y_true: 실제 라벨
            y_pred: 예측 라벨
            y_pred_proba: 예측 확률 (선택사항)

        Returns:
            dict: 성능 지표 딕셔너리
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
        }

        # ROC AUC는 확률값이 있을 때만 계산
        if y_pred_proba is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                metrics["roc_auc"] = None
        else:
            metrics["roc_auc"] = None

        return metrics

    def plot_confusion_matrix(self, cm, model_name, fold_num=None):
        """
        혼동 행렬 시각화

        Args:
            cm: 혼동 행렬
            model_name: 모델 이름
            fold_num: 폴드 번호 (선택사항)

        Returns:
            matplotlib.pyplot
        """
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)

        title = f"{model_name} - Confusion Matrix"
        if fold_num is not None:
            title += f" (Fold {fold_num})"

        plt.title(title)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        return plt

    def plot_roc_curve(self, y_true, y_pred_proba, model_name, fold_num=None):
        """
        ROC 곡선 시각화

        Args:
            y_true: 실제 라벨
            y_pred_proba: 예측 확률
            model_name: 모델 이름
            fold_num: 폴드 번호 (선택사항)

        Returns:
            matplotlib.pyplot
        """
        if y_pred_proba is None:
            print("ROC 곡선을 그리려면 예측 확률이 필요합니다.")
            return

        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            plt.plot(
                fpr,
                tpr,
                color="darkorange",
                lw=2,
                label=f"ROC curve (AUC = {roc_auc:.2f})",
            )
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")

            title = f"{model_name} - ROC Curve"
            if fold_num is not None:
                title += f" (Fold {fold_num})"
            plt.title(title)

            plt.legend(loc="lower right")
            plt.grid(True)

            return plt
        except ValueError as e:
            print(f"ROC 곡선 생성 중 오류: {e}")

    def plot_precision_recall_curve(
        self, y_true, y_pred_proba, model_name, fold_num=None
    ):
        """
        Precision-Recall 곡선 시각화

        Args:
            y_true: 실제 라벨
            y_pred_proba: 예측 확률
            model_name: 모델 이름
            fold_num: 폴드 번호 (선택사항)
        """
        if y_pred_proba is None:
            print("PR 곡선을 그리려면 예측 확률이 필요합니다.")
            return

        try:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = auc(recall, precision)

            plt.figure(figsize=(8, 6))
            plt.plot(
                recall,
                precision,
                color="blue",
                lw=2,
                label=f"PR curve (AUC = {pr_auc:.2f})",
            )
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])

            title = f"{model_name} - Precision-Recall Curve"
            if fold_num is not None:
                title += f" (Fold {fold_num})"
            plt.title(title)

            plt.legend(loc="lower left")
            plt.grid(True)

            return plt
        except ValueError as e:
            print(f"PR 곡선 생성 중 오류: {e}")

    def print_data_distribution(self, y_train, y_test, fold_num):
        """데이터 분포 출력"""
        print(f"\n--- Fold {fold_num} 데이터 분포 ---")
        print(f"훈련 세트 - 일반: {np.sum(y_train == 0)}, 공격: {np.sum(y_train == 1)}")
        print(f"테스트 세트 - 일반: {np.sum(y_test == 0)}, 공격: {np.sum(y_test == 1)}")

    def print_model_results(self, metrics, model_name, fold_num):
        """모델 결과 출력"""
        print(f"\n--- Fold {fold_num}: {model_name} 모델 성능 평가 ---")
        print(f"정확도 (Accuracy): {metrics['accuracy']:.4f}")
        print(f"정밀도 (Precision): {metrics['precision']:.4f}")
        print(f"재현율 (Recall): {metrics['recall']:.4f}")
        print(f"F1-점수 (F1-Score): {metrics['f1_score']:.4f}")

        if metrics["roc_auc"] is not None:
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")

        print("\n혼동 행렬 (Confusion Matrix):")
        print(metrics["confusion_matrix"])

    # ========== k-Fold Cross Validation 실행 함수 ==========
    def run_kfold_validation(
        self,
        n_splits=3,
        model_name="Random Forest",
        train_path="/DS_term/4_Feature_engineering_DataSet.csv",
        test_path="/DS_term/4_Feature_engineering_TestDataSet.csv",
    ):
        """
        K-Fold Cross Validation 실행

        Args:
            n_splits (int): 폴드 수
            model_name (str): 사용할 모델 이름
            train_path (str): 전처리된 훈련 데이터 경로
            test_path (str): 전처리된 테스트 데이터 경로
        """
        # K-Fold 설정
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # 결과 저장용 리스트
        fold_results = []

        print(f"\n{'='*60}")
        print(f"사용 모델: {model_name}")
        print(f"데이터 샘플 수: {len(self.df)}")
        print(f"K-Fold Cross Validation 시작 (k={n_splits})")
        print(f"{'='*60}")

        # 각 폴드에 대해 실행
        for round, (train_index, test_index) in enumerate(kf.split(self.df)):
            print(f"\n{'='*20} Fold {round + 1} {'='*20}")

            try:
                # 데이터 전처리
                df_train, df_test = self.df.iloc[train_index], self.df.iloc[test_index]
                processed_train, processed_test = self.preprocess(df_train, df_test)

                # 특성과 라벨 분리
                X_train = processed_train.drop(["Label"], axis=1)
                X_test = processed_test.drop(["Label"], axis=1)
                y_train = processed_train["Label"]
                y_test = processed_test["Label"]

                # 데이터 분포 출력
                self.print_data_distribution(y_train, y_test, round + 1)

                # 모델별 예측 실행
                y_pred_proba = None

                if model_name == "Linear Regression":
                    y_pred, _, _ = self.linear_regression_predict(
                        X_train, X_test, y_train, y_test
                    )

                    # 성능 지표 계산
                    metrics = self.calculate_metrics(y_test, y_pred)
                    cm = confusion_matrix(y_test, y_pred)

                    # 결과 출력
                    self.print_model_results(metrics, model_name, round + 1)

                    # 결과 저장
                    fold_results.append(
                        {
                            "metrics": metrics,
                            "cm": cm,
                            "y_true": y_test,
                            "y_pred_proba": y_pred_proba,
                        }
                    )

                elif model_name == "Logistic Regression":
                    y_pred, y_pred_proba, model = self.logistic_regression_predict(
                        X_train, X_test, y_train, y_test
                    )

                    # 성능 지표 계산
                    metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
                    cm = confusion_matrix(y_test, y_pred)

                    # 결과 출력
                    self.print_model_results(metrics, model_name, round + 1)

                    # 결과 저장
                    fold_results.append(
                        {
                            "metrics": metrics,
                            "cm": cm,
                            "y_true": y_test,
                            "y_pred_proba": y_pred_proba,
                        }
                    )

                elif model_name == "Random Forest":
                    y_pred, y_pred_proba, model = self.random_forest_predict(
                        X_train, X_test, y_train, y_test
                    )

                    # 성능 지표 계산
                    metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
                    cm = confusion_matrix(y_test, y_pred)

                    # 결과 출력
                    self.print_model_results(metrics, model_name, round + 1)

                    # 결과 저장
                    fold_results.append(
                        {
                            "metrics": metrics,
                            "cm": cm,
                            "y_true": y_test,
                            "y_pred_proba": y_pred_proba,
                        }
                    )

                elif model_name == "Decision Tree":
                    y_pred, y_pred_proba, model = self.decision_tree_predict(
                        X_train, X_test, y_train, y_test
                    )

                    # 성능 지표 계산
                    metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
                    cm = confusion_matrix(y_test, y_pred)

                    # 결과 출력
                    self.print_model_results(metrics, model_name, round + 1)

                    # 결과 저장
                    fold_results.append(
                        {
                            "metrics": metrics,
                            "cm": cm,
                            "y_true": y_test,
                            "y_pred_proba": y_pred_proba,
                        }
                    )

                elif model_name == "PCA Regression":
                    y_pred, _, _ = self.PCA_regression_predict(
                        X_train, X_test, y_train, y_test, 1
                    )

                    # 성능 지표 계산
                    metrics = self.calculate_metrics(y_test, y_pred)
                    cm = confusion_matrix(y_test, y_pred)

                    # 결과 출력
                    self.print_model_results(metrics, model_name, round + 1)

                    # 결과 저장
                    fold_results.append(
                        {
                            "metrics": metrics,
                            "cm": cm,
                            "y_true": y_test,
                            "y_pred_proba": y_pred_proba,
                        }
                    )

                else:
                    print(f"지원하지 않는 모델: {model_name}")
                    continue

            except FileNotFoundError as e:
                print(f"파일을 찾을 수 없습니다: {e}")
                continue
            except Exception as e:
                print(f"Fold {round + 1} 실행 중 오류 발생: {e}")
                continue

        # 전체 결과 요약
        self.summarize_kfold_results(fold_results, model_name)

    def summarize_kfold_results(self, fold_results, model_name, is_plot=True):
        """K-Fold 결과 요약"""
        if not fold_results:
            print("요약할 결과가 없습니다.")
            return

        print(f"\n{'='*60}")
        print(f"{model_name} K-Fold Cross Validation 결과 요약")
        print(f"{'='*60}")

        # 평균과 표준편차 계산
        metrics_summary = {}
        for metric in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            values = [
                result["metrics"][metric]
                for result in fold_results
                if result["metrics"][metric] is not None
            ]
            if values:
                metrics_summary[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }

        # 결과 출력
        for metric, stats in metrics_summary.items():
            print(f"avg {metric.upper()}: {stats['mean']:.4f}")

        if is_plot:
            n_folds = len(fold_results)
            plt.figure(figsize=(18, 12))

            # 첫 번째 서브플롯: 집계된 성능 지표
            plt.subplot(3, max(n_folds, 3), 1)
            metrics = list(metrics_summary.keys())
            means = [metrics_summary[m]["mean"] for m in metrics]
            stds = [metrics_summary[m]["std"] for m in metrics]

            bars = plt.bar(range(len(metrics)), means, yerr=stds, capsize=5,
                          color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
                          alpha=0.8)
            plt.title(f'{model_name} - Average Performance')
            plt.xticks(range(len(metrics)), [m.upper() for m in metrics], rotation=45)
            plt.ylabel('Score')
            plt.ylim(0, 1.1)

            # 평균값 표시
            for i, (bar, mean_val) in enumerate(zip(bars, means)):
                plt.text(i, mean_val + 0.02, f'{mean_val:.3f}',
                        ha='center', fontweight='bold')

            # 혼동 행렬
            for round, result in enumerate(fold_results):
                if "cm" in result:
                    plt.subplot(3, n_folds, n_folds + round + 1)
                    self.plot_confusion_matrix(result["cm"], model_name, round + 1)

            # ROC 커브
            for round, result in enumerate(fold_results):
                if ("y_true" in result) and ("y_pred_proba" in result):
                    plt.subplot(3, n_folds, 2 * n_folds + round + 1)
                    self.plot_roc_curve(
                        result["y_true"], result["y_pred_proba"], model_name, round + 1
                    )

            plt.tight_layout()
            plt.show()


def main():
    # 파이프라인 초기화
    pipeline = ModelingPipeline()

    # 데이터 로드
    if not pipeline.load_data():
        return

    # 데이터 샘플 수 조절
    sample_rate = 0.8 # 이거 조절 하시면 됩니다. 0.5 무난, 0.1은 빛의 속도! 0.3까지는 결과가 robust 합니다.
    X_other, X_sample = train_test_split(
      pipeline.df,
      test_size = sample_rate,
      random_state = 42,
      shuffle=True,
      stratify=pipeline.df['Label']
    )
    pipeline.df = X_sample.reset_index(drop=True)

    models_to_test = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Linear Regression",
        "PCA Regression",
    ]

    # K-Fold Cross Validation 실행
    for model_name in models_to_test:
        pipeline.run_kfold_validation(
            n_splits=3,
            model_name=model_name,
            train_path="/DS_term/4_Feature_engineering_DataSet.csv",
            test_path="/DS_term/4_Feature_engineering_TestDataSet.csv",
        )

    print(f"\n{'='*60}")
    print("모든 모델링 절차가 완료되었습니다!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
