import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from Data_collection_selection import load_dataset
from Data_cleaning import data_cleaning
from Data_preprocessing import data_scaling
from Feature_Engineering import data_preprocessing

# --- 데이터 불러오기 ---
load_dataset()

print("\n--- 데이터 불러오기 ---")
df = pd.read_csv("DS_term/1_Raw_DataSet.csv")
print(df)

# --- 사용할 모델 정의 ---
print("\n--- 모델 정의 ---")
models = {
    "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
}
print(f"정의된 모델: {list(models.keys())}")

# --- K-Fold Cross Validation ---
kf = KFold(n_splits=3, shuffle=True, random_state=42)

model_results = {} # 모델 별 결과 저장용 딕셔너리

for round, (train_index, test_index) in enumerate(kf.split(df)):
  
  print(f"\n--- {round}번째 Fold  ---")
  # --- 1. 데이터 분할 (학습/테스트) ---
  df_train, df_test = df.iloc[train_index], df.iloc[test_index]

  # --- 2. 데이터 클리닝 ---
  data_cleaning(df_train, is_test=False, is_plot=False)
  data_cleaning(df_test, is_test=True, is_plot=False)

  # --- 3. 데이터 스케일링 ---
  data_scaling(is_plot=False)

  # --- 4. 데이터 전처리 ---
  data_preprocessing(is_plot=False)

  # --- 5. 전처리 된 데이터 불러오기 ---
  processed_train = pd.read_csv("DS_term/4_Feature_engineering_DataSet.csv")
  processed_test = pd.read_csv("DS_term/4_Feature_engineering_TestDataSet.csv")

  X_train = processed_train.drop(["Label"], axis=1)
  X_test = processed_test.drop("Label", axis=1)
  y_train = processed_train["Label"]
  y_test = processed_test["Label"]

  # --- 6. 모델 학습, 예측 및 평가 ---
  model_results = {}

  print("\n\n--- 모델 학습, 예측 및 평가 시작 ---")
  for model_name, model_instance in models.items():
      print(f"\n--- 모델: {model_name} ---")

      # --- 6.1. 모델 학습 ---
      print("모델 학습 중...")
      model_instance.fit(X_train, y_train)
      print("모델 학습 완료.")

      # --- 6.2. 테스트 데이터 예측 ---
      y_pred = model_instance.predict(X_test)
      if hasattr(model_instance, "predict_proba"):
          y_pred_proba = model_instance.predict_proba(X_test)[:, 1] # 이진 분류 가정
      else:
          y_pred_proba = None

      # --- 6.3. 기본 평가 지표 계산 ---
      accuracy = accuracy_score(y_test, y_pred)
      precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
      recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
      f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

      # --- 6.4. 혼동 행렬 시각화 ---
      cm = confusion_matrix(y_test, y_pred)
      plt.figure(figsize=(6, 4))
      sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
      plt.title(f"{model_name} - Confusion Matrix")
      plt.xlabel("Predicted Label")
      plt.ylabel("True Label")
      plt.show()

      roc_auc = None
      if y_pred_proba is not None:
          try:
            # --- 6.5. ROC 곡선 시각화 ---
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, color='orange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{model_name} - ROC Curve')
            plt.legend(loc="lower right")
            plt.show()

            # --- 6.6. Precision-Recall 곡선 시각화 ---
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall_curve, precision_curve)
            plt.figure(figsize=(6,4))
            plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.0])
            plt.xlim([0.0, 1.0])
            plt.title(f'{model_name} - Precision-Recall Curve')
            plt.legend(loc="lower left")
            plt.show()
            
          except ValueError as e:
              print(f"ROC AUC 또는 PR 곡선 계산/시각화 중 오류: {e}")
              roc_auc = None

      # --- 6.7. 평가 지표 출력 ---
      print(f"\n--- {model_name} 모델 성능 평가 결과 ---")
      print(f"정확도 (Accuracy): {accuracy:.4f}")
      print(f"정밀도 (Precision): {precision:.4f}")
      print(f"재현율 (Recall): {recall:.4f}")
      print(f"F1-점수 (F1-Score): {f1:.4f}")
      if roc_auc is not None:
          print(f"ROC AUC: {roc_auc:.4f}")
      
      print("\n혼동 행렬 (Confusion Matrix):")
      print(cm)

# --- 스크립트 종료 메시지 ---
print("\n\n모델링 절차 완료.")