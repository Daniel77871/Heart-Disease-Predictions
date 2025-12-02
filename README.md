# Heart Disease Prediction

使用機器學習演算法預測心臟病的完整分析專案。

## 專案概述

本專案使用多種機器學習模型來預測心臟病，包括：
- 傳統機器學習模型（Logistic Regression, Random Forest, SVM 等）
- Boosting 演算法（XGBoost, LightGBM, CatBoost）
- 模型可解釋性分析（SHAP, ELI5）

### 主要功能
- 📊 完整的探索性數據分析 (EDA)
- 🔍 多種相關性分析（Pearson, Point-Biserial, Cramer's V）
- 🤖 12+ 種機器學習模型比較
- ⚙️ 自動化超參數調整
- 📈 模型效能評估與視覺化
- 🔬 特徵重要性分析與 SHAP 解釋

### 模型效能
- **最佳模型**: Logistic Regression & LightGBM
- **準確率**: 86.49%
- **ROC AUC**: 0.92

## 安裝說明

### 1. 環境需求
- Python 3.9+
- Homebrew (Mac 用戶需安裝 OpenMP)

### 2. 安裝 OpenMP

**Mac 用戶:**
```bash
brew install libomp
```

**Windows 用戶:**
```bash
# 通常 pip 安裝 XGBoost 時會自動包含，如遇到問題可安裝：
# 下載並安裝 Microsoft C++ Build Tools
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

### 3. 建立虛擬環境
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# 或
venv\Scripts\activate  # Windows
```

### 4. 安裝套件
```bash
pip install -r requirements.txt
```

## 使用方法

### 執行完整分析流程
```bash
python main.py
```

### 輸出結果
執行後會在 `plots/` 目錄生成以下檔案：

#### 探索性數據分析
- `target_distribution.png` - 目標變數分布
- `numerical_distributions.png` - 數值特徵分布
- `categorical_distributions.png` - 類別特徵分布
- `pairplot.png` - 特徵配對圖
- `regression_plots.png` - 迴歸分析圖

#### 相關性分析
- `pearson_correlation.png` - Pearson 相關係數
- `point_biserial_correlation.png` - Point-Biserial 相關係數
- `cramers_v_correlation.png` - Cramer's V 相關係數

#### 模型評估
- `baseline_model_results.csv` - 基準模型效能總表
- `boosting_model_results.csv` - Boosting 模型效能總表
- `confusion_matrices.png` - 所有模型的混淆矩陣
- `roc_curves.png` - ROC 曲線比較
- `lr_tuned_confusion_matrix.png` - 調參後 Logistic Regression
- `lgbm_tuned_confusion_matrix.png` - 調參後 LightGBM

#### 模型可解釋性
- `permutation_importance.png` - 排列重要性
- `shap_summary_bar.png` - SHAP 特徵重要性
- `shap_summary_dot.png` - SHAP 詳細分析圖

## 資料集

專案使用 `heart.csv` 資料集，包含以下特徵：
- **年齡** (age)
- **性別** (sex)
- **胸痛類型** (chest_pain_type)
- **靜息血壓** (resting_blood_pressure)
- **膽固醇** (cholesterol)
- **空腹血糖** (fasting_blood_sugar)
- **靜息心電圖** (resting_electrocardiogram)
- **最大心率** (max_heart_rate_achieved)
- **運動誘發心絞痛** (exercise_induced_angina)
- **ST 段壓低** (st_depression)
- **ST 段斜率** (st_slope)
- **主要血管數量** (num_major_vessels)
- **地中海貧血** (thalassemia)
- **目標變數** (target) - 是否有心臟病

## 專案結構

```
Heart Disease Predictions/
├── main.py                 # 主程式
├── heart.csv              # 資料集
├── requirements.txt       # 套件需求
├── README.md             # 說明文件
└── plots/                # 輸出圖表目錄
    ├── *.png            # 視覺化圖表
    └── *.csv            # 模型效能結果
```

## 技術棧

- **數據處理**: NumPy, Pandas
- **視覺化**: Matplotlib, Seaborn
- **機器學習**: Scikit-learn
- **Boosting**: XGBoost, LightGBM, CatBoost
- **可解釋性**: SHAP, ELI5

## 授權

此專案僅供學習與研究使用。
