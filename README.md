# 🛡️ SIM Swapping Attack Detection and Risk Prediction Model

> **데이터과학 팀프로젝트 | 2025년 1학기**

통신사 유심 해킹(SIM 스와핑)을 통한 개인정보 탈취, 메신저 피싱 및 금융사기 사고가 증가하고 있습니다.  
SIM이 부정 재발급될 경우 공격자가 정상 사용자로 인증받아 SNS, 계좌 접근이 가능해지는 문제를 해결하기 위해  
본 프로젝트는 **네트워크 접속 로그 기반 이상행위 탐지 시스템**을 설계하였습니다.

---

## 📌 프로젝트 개요

- **목표**: 다양한 전처리 및 피처 엔지니어링을 통해 사이버 공격 탐지 정확도를 극대화
- **데이터셋**: UNSW-NB15 원본 데이터 + 전처리 가공 데이터
- **사용 기술**: pandas, scikit-learn, matplotlib, seaborn, logistic regression, random forest 등

---

## 📁 폴더/파일 구조

```
DS_Term_Project/
├── DS_term/                      # 대형 데이터셋 경로 (GitHub에는 제외됨)
│   └── README.txt                # → 다운로드 경로 안내
├── Data_cleaning.py             # 데이터 정제 스크립트
├── Data_collection_selection.py # 데이터 수집 및 선택
├── Data_preprocessing.py        # 스케일링/인코딩 처리
├── Feature_Engineering.py       # 피처 생성 및 선택
├── Modeling.py                  # 머신러닝 모델 학습
├── ZDataScience_TermProject.ipynb # 전체 과정 통합 노트북
├── ReadMe.docx                  # 보고서 제출용 Word 문서
└── README.md                    # 프로젝트 설명
```

## 📂 데이터셋 다운로드

> GitHub의 파일 크기 제한(100MB)으로 인해, 전체 데이터셋은 포함되어 있지 않습니다.

📁 `DS_term/README.txt` 파일을 참고하여 아래 Google Drive 링크에서 다운로드 후  
`DS_term/` 폴더 안에 직접 넣어주세요.

🔗 [Google Drive 다운로드 링크](https://drive.google.com/your-dataset-link)

---

## 🧪 주요 처리 과정

### 1. 데이터 수집 및 클렌징
- 여러 raw CSV 병합 및 열 이름 정리
- 결측치 처리, 중복 제거, Outlier 제거 (정상 트래픽만)

### 2. 전처리
- 로그 + RobustScaler, StandardScaler, MinMaxScaler 사용
- One-Hot Encoding 적용

### 3. 피처 엔지니어링
- Feature Selection: L1 Logistic, Correlation (> 0.3)
- Feature Creation: `load_ratio`, `ttl_diff` 등
- 총 17개 수치형 + 13개 범주형 피처 선정

### 4. 모델링
- 모델: Logistic Regression, Random Forest
- 정확도: **99% / 100%**

---

## 📊 시각화

- TTL, Sload, Dload 등의 공격/정상 분포 비교
- Confusion matrix, feature importance 시각화

📌 자세한 시각화는 `ZDataScience_TermProject.ipynb` 에서 확인 가능

---

## 👥 팀원 역할 분담

| 이름   | 역할                          
|--------|-------------------------
| 장예은 | 데이터 클리닝             
| 김나현 | 데이터 분석               
| 송영우 | 피처 엔지니어링          
| 강민재 | 모델링   
| 정명성 | 문서화  
---

## 📎 참고 자료

- [UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- Scikit-learn 공식 문서
- Kaggle 코드 참조

---

> 🚀 2025. Gachon University Data Science Team Project
