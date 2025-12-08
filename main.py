# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import uniform

import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import recall_score, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Boosting libraries
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Explainability
import eli5
from eli5.sklearn import PermutationImportance
import shap

# ============================================================
# 2. CONFIGURATION
# ============================================================
# Color palette
MYPAL = ['#FC05FB', '#FEAEFE', '#FCD2FC', '#F3FEFA', '#B4FFE4', '#3FFEBA']
MYPAL_1 = ['#FC05FB', '#FEAEFE', '#FCD2FC', '#F3FEFA', '#B4FFE4', '#3FFEBA', '#FC05FB', '#FEAEFE', '#FCD2FC']

# Random seed
SEED = 0
TEST_SIZE = 0.25


# ============================================================
# 3. DATA LOADING AND PREPROCESSING FUNCTIONS
# ============================================================
def load_data(filepath):
    """Load the heart disease dataset."""
    data = pd.read_csv(filepath)
    print(f'Shape of the data is {data.shape}')
    return data


def clean_data(data):
    """Clean the dataset by removing invalid values."""
    data = data[data['ca'] < 4]  # drop the wrong ca values
    data = data[data['thal'] > 0]  # drop the wrong thal value
    print(f'The length of the data now is {len(data)} instead of 303!')
    return data


def rename_columns(data):
    """Rename columns for better readability."""
    data = data.rename(
        columns={
            'cp': 'chest_pain_type',
            'trestbps': 'resting_blood_pressure',
            'chol': 'cholesterol',
            'fbs': 'fasting_blood_sugar',
            'restecg': 'resting_electrocardiogram',
            'thalach': 'max_heart_rate_achieved',
            'exang': 'exercise_induced_angina',
            'oldpeak': 'st_depression',
            'slope': 'st_slope',
            'ca': 'num_major_vessels',
            'thal': 'thalassemia'
        },
        errors="raise"
    )
    return data


def encode_categorical_values(data):
    """Convert numeric codes to meaningful categorical values."""
    # Sex
    data.loc[data['sex'] == 0, 'sex'] = 'female'
    data.loc[data['sex'] == 1, 'sex'] = 'male'

    # Chest pain type
    data.loc[data['chest_pain_type'] == 0, 'chest_pain_type'] = 'typical angina'
    data.loc[data['chest_pain_type'] == 1, 'chest_pain_type'] = 'atypical angina'
    data.loc[data['chest_pain_type'] == 2, 'chest_pain_type'] = 'non-anginal pain'
    data.loc[data['chest_pain_type'] == 3, 'chest_pain_type'] = 'asymptomatic'

    # Fasting blood sugar
    data.loc[data['fasting_blood_sugar'] == 0, 'fasting_blood_sugar'] = 'lower than 120mg/ml'
    data.loc[data['fasting_blood_sugar'] == 1, 'fasting_blood_sugar'] = 'greater than 120mg/ml'

    # Resting electrocardiogram
    data.loc[data['resting_electrocardiogram'] == 0, 'resting_electrocardiogram'] = 'normal'
    data.loc[data['resting_electrocardiogram'] == 1, 'resting_electrocardiogram'] = 'ST-T wave abnormality'
    data.loc[data['resting_electrocardiogram'] == 2, 'resting_electrocardiogram'] = 'left ventricular hypertrophy'

    # Exercise induced angina
    data.loc[data['exercise_induced_angina'] == 0, 'exercise_induced_angina'] = 'no'
    data.loc[data['exercise_induced_angina'] == 1, 'exercise_induced_angina'] = 'yes'

    # ST slope
    data.loc[data['st_slope'] == 0, 'st_slope'] = 'upsloping'
    data.loc[data['st_slope'] == 1, 'st_slope'] = 'flat'
    data.loc[data['st_slope'] == 2, 'st_slope'] = 'downsloping'

    # Thalassemia
    data.loc[data['thalassemia'] == 1, 'thalassemia'] = 'fixed defect'
    data.loc[data['thalassemia'] == 2, 'thalassemia'] = 'normal'
    data.loc[data['thalassemia'] == 3, 'thalassemia'] = 'reversable defect'

    return data


def label_encode_cat_features(data, cat_features):
    """
    Given a dataframe and its categorical features, 
    this function returns label-encoded dataframe.
    """
    label_encoder = LabelEncoder()
    data_encoded = data.copy()

    for col in cat_features:
        data_encoded[col] = label_encoder.fit_transform(data[col])

    return data_encoded


# ============================================================
# 4. VISUALIZATION FUNCTIONS
# ============================================================
def plot_target_distribution(data, save_path='plots/target_distribution.png'):
    """Plot target variable distribution."""
    plt.figure(figsize=(7, 5), facecolor='#F6F5F4')
    total = float(len(data))
    ax = sns.countplot(x=data['target'], palette=MYPAL[1::4])
    ax.set_facecolor('#F6F5F4')

    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 3,
                '{:1.1f} %'.format((height / total) * 100), ha="center",
                bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.5))

    ax.set_title('Target variable distribution', fontsize=20, y=1.05)
    sns.despine(right=True)
    sns.despine(offset=5, trim=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_numerical_distributions(data, num_feats, save_path='plots/numerical_distributions.png'):
    """Plot distributions of numerical features (Fixed Layout)."""
    L = len(num_feats)
    ncol = 2
    nrow = int(np.ceil(L / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 14), facecolor='#F6F5F4')
    fig.subplots_adjust(top=0.92)
    
    axes_flat = axes.flatten()

    for i in range(L, len(axes_flat)):
        axes_flat[i].set_visible(False)

    for i, col in enumerate(num_feats):
        ax = axes_flat[i]
        ax.set_facecolor('#F6F5F4')

        if col == 'num_major_vessels':
            sns.countplot(data=data, x=col, hue="target", palette=MYPAL[1::4], ax=ax)
            for p in ax.patches:
                height = p.get_height()
                if height > 0:
                    ax.text(p.get_x() + p.get_width() / 2., height + 1, '{:1.0f}'.format((height)), ha="center",
                            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.5))
            ax.set_xlabel(col, fontsize=16)
            ax.set_ylabel("count", fontsize=16)
        
        else:
            sns.kdeplot(data=data, x=col, hue="target", multiple="stack", palette=MYPAL[1::4], ax=ax)
            ax.set_xlabel(col, fontsize=16)
            ax.set_ylabel("density", fontsize=16)

        sns.despine(right=True, ax=ax)
        sns.despine(offset=0, trim=False, ax=ax)

    plt.suptitle('Distribution of Numerical Features', fontsize=24)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_pairplot(data, save_path='plots/pairplot.png'):
    """Plot pairplot for numerical features."""
    cols = ['age', 'cholesterol', 'resting_blood_pressure', 'max_heart_rate_achieved', 'st_depression', 'target']
    data_ = data[cols]
    g = sns.pairplot(data_, hue="target", corner=True, diag_kind='hist', palette=MYPAL[1::4])
    plt.suptitle('Pairplot: Numerical Features', fontsize=24)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_regression_plots(data, save_path='plots/regression_plots.png'):
    """Plot regression plots of selected features."""
    fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    sns.regplot(data=data[data['target'] == 1], x='age', y='cholesterol', ax=ax[0], color=MYPAL[0], label='1')
    sns.regplot(data=data[data['target'] == 0], x='age', y='cholesterol', ax=ax[0], color=MYPAL[5], label='0')
    sns.regplot(data=data[data['target'] == 1], x='age', y='max_heart_rate_achieved', ax=ax[1], color=MYPAL[0], label='1')
    sns.regplot(data=data[data['target'] == 0], x='age', y='max_heart_rate_achieved', ax=ax[1], color=MYPAL[5], label='0')
    sns.regplot(data=data[data['target'] == 1], x='age', y='resting_blood_pressure', ax=ax[2], color=MYPAL[0], label='1')
    sns.regplot(data=data[data['target'] == 0], x='age', y='resting_blood_pressure', ax=ax[2], color=MYPAL[5], label='0')
    sns.regplot(data=data[data['target'] == 1], x='age', y='st_depression', ax=ax[3], color=MYPAL[0], label='1')
    sns.regplot(data=data[data['target'] == 0], x='age', y='st_depression', ax=ax[3], color=MYPAL[5], label='0')
    plt.suptitle('Reg plots of selected features')
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def count_plot(data, cat_feats, save_path='plots/categorical_distributions.png'):
    """Plot distributions of categorical features (Fixed Layout)."""
    L = len(cat_feats)
    ncol = 2
    nrow = int(np.ceil(L / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(18, 24), facecolor='#F6F5F4')
    fig.subplots_adjust(top=0.92)
    
    axes_flat = axes.flatten()

    for i in range(L, len(axes_flat)):
        axes_flat[i].set_visible(False)

    for i, col in enumerate(cat_feats):
        ax = axes_flat[i]
        
        sns.countplot(data=data, x=col, hue="target", palette=MYPAL[1::4], ax=ax)
        
        ax.set_facecolor('#F6F5F4')
        ax.set_xlabel(col, fontsize=16)
        ax.set_ylabel("count", fontsize=16)
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        sns.despine(right=True, ax=ax)
        sns.despine(offset=0, trim=False, ax=ax)
        ax.legend(facecolor='#F6F5F4')

        for p in ax.patches:
            height = p.get_height()
            if height > 0: 
                ax.text(p.get_x() + p.get_width() / 2., height + 3, '{:1.0f}'.format((height)), ha="center",
                        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.5))

    plt.suptitle('Distribution of Categorical Features', fontsize=24)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_pearson_correlation(data, num_feats, save_path='plots/pearson_correlation.png'):
    """Plot Pearson correlation heatmap for numerical features."""
    df_ = data[num_feats]
    corr = df_.corr(method='pearson')
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(8, 5), facecolor=None)
    cmap = sns.color_palette(MYPAL, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0, annot=True,
                square=False, linewidths=.5, cbar_kws={"shrink": 0.75})
    ax.set_title("Numerical features correlation (Pearson's)", fontsize=20, y=1.05)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def point_biserial(x, y):
    """Calculate point-biserial correlation."""
    pb = stats.pointbiserialr(x, y)
    return pb[0]


def plot_point_biserial_correlation(data, save_path='plots/point_biserial_correlation.png'):
    """Plot point-biserial correlation heatmap."""
    feats_ = ['age', 'cholesterol', 'resting_blood_pressure', 'max_heart_rate_achieved', 
              'st_depression', 'num_major_vessels', 'target']

    rows = []
    for x in feats_:
        col = []
        for y in feats_:
            pbs = point_biserial(data[x], data[y])
            col.append(round(pbs, 2))
        rows.append(col)

    pbs_results = np.array(rows)
    DF = pd.DataFrame(pbs_results, columns=data[feats_].columns, index=data[feats_].columns)

    mask = np.triu(np.ones_like(DF, dtype=bool))
    corr = DF.mask(mask)

    f, ax = plt.subplots(figsize=(8, 5), facecolor=None)
    cmap = sns.color_palette(MYPAL, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1, center=0, annot=True,
                square=False, linewidths=.5, cbar_kws={"shrink": 0.75})
    ax.set_title("Cont feats vs target correlation (point-biserial)", fontsize=20, y=1.05)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def cramers_v(x, y):
    """
    Calculate Cramer's V correlation coefficient.
    Source: https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def plot_cramers_v_correlation(data, cat_feats, save_path='plots/cramers_v_correlation.png'):
    """Plot Cramer's V correlation heatmap for categorical features."""
    data_ = data[cat_feats]
    rows = []
    for x in data_:
        col = []
        for y in data_:
            cramers = cramers_v(data_[x], data_[y])
            col.append(round(cramers, 2))
        rows.append(col)

    cramers_results = np.array(rows)
    df = pd.DataFrame(cramers_results, columns=data_.columns, index=data_.columns)

    mask = np.triu(np.ones_like(df, dtype=bool))
    corr = df.mask(mask)
    f, ax = plt.subplots(figsize=(10, 6), facecolor=None)
    cmap = sns.color_palette(MYPAL_1, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=0, center=0, annot=True,
                square=False, linewidths=.01, cbar_kws={"shrink": 0.75})
    ax.set_title("Categorical Features Correlation (Cramer's V)", fontsize=20, y=1.05)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# 5. MODEL EVALUATION FUNCTIONS
# ============================================================
def score_summary(names, classifiers, X_train, y_train, X_val, y_val):
    """
    Given a list of classifiers, this function calculates the accuracy,
    ROC_AUC and Recall and returns the values in a dataframe.
    """
    cols = ["Classifier", "Accuracy", "ROC_AUC", "Recall", "Precision", "F1"]
    data_table = pd.DataFrame(columns=cols)

    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)

        pred = clf.predict(X_val)
        accuracy = accuracy_score(y_val, pred)

        pred_proba = clf.predict_proba(X_val)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_val, pred_proba)
        roc_auc = auc(fpr, tpr)

        # confusion matrix, cm
        cm = confusion_matrix(y_val, pred)

        # recall: TP/(TP+FN)
        recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])

        # precision: TP/(TP+FP)
        precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])

        # F1 score
        f1 = 2 * recall * precision / (recall + precision)

        df = pd.DataFrame([[name, accuracy * 100, roc_auc, recall, precision, f1]], columns=cols)
        data_table = pd.concat([data_table, df], ignore_index=True)

    return np.round(data_table.reset_index(drop=True), 2)


def plot_conf_matrix(names, classifiers, X_train, y_train, X_val, y_val, 
                     nrows, ncols, fig_a, fig_b, save_path='plots/confusion_matrices.png'):
    """
    Plots confusion matrices in subplots.

    Args:
        names : list of names of the classifier
        classifiers : list of classification algorithms
        nrows, ncols : number of rows and columns in the subplots
        fig_a, fig_b : dimensions of the figure size
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_a, fig_b))

    i = 0
    for clf, ax in zip(classifiers, axes.flatten()):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        # Compute confusion matrix with labels=[0, 1], then flip rows
        cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
        cm_display = cm[::-1]  # Flip rows: Y-axis becomes 1 (top) -> 0 (bottom)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_display, display_labels=[1, 0])
        disp.plot(ax=ax, colorbar=False)
        # Fix X-axis labels to show 0 (left) -> 1 (right)
        ax.set_xticklabels([0, 1])
        ax.set_title(names[i])
        i = i + 1

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def roc_auc_curve(names, classifiers, X_train, y_train, X_val, y_val, 
                  save_path='plots/roc_curves.png'):
    """
    Given a list of classifiers, this function plots the ROC curves.
    """
    plt.figure(figsize=(12, 8))

    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)

        pred_proba = clf.predict_proba(X_val)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_val, pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=3, label=name + ' ROC curve (area = %0.2f)' % (roc_auc))

    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) curves', fontsize=20)
    plt.legend(loc="lower right")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# 6. MAIN FUNCTION
# ============================================================
def main():
    """Main function to run the heart disease prediction pipeline."""
    import os
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # ----------------------------------------------------------
    # 6.1 LOAD AND PREPROCESS DATA
    # ----------------------------------------------------------
    print("=" * 60)
    print("LOADING AND PREPROCESSING DATA")
    print("=" * 60)
    
    # Load data - modify this path to your data location
    data = load_data('heart.csv')
    
    print("\nFirst 5 rows:")
    print(data.head())
    
    print("\nData types:")
    print(data.dtypes)
    
    # Clean data
    data = clean_data(data)
    
    # Rename columns
    data = rename_columns(data)
    
    # Encode categorical values
    data = encode_categorical_values(data)
    
    print("\nData types after encoding:")
    print(data.dtypes)
    
    print("\nFirst 5 rows after preprocessing:")
    print(data.head())
    
    # Define feature types
    num_feats = ['age', 'cholesterol', 'resting_blood_pressure', 
                 'max_heart_rate_achieved', 'st_depression', 'num_major_vessels']
    bin_feats = ['sex', 'fasting_blood_sugar', 'exercise_induced_angina', 'target']
    nom_feats = ['chest_pain_type', 'resting_electrocardiogram', 'st_slope', 'thalassemia']
    cat_feats = nom_feats + bin_feats
    
    # ----------------------------------------------------------
    # 6.2 EXPLORATORY DATA ANALYSIS
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Target distribution
    plot_target_distribution(data)
    
    # Numerical features statistics
    print("\nNumerical Features Statistics:")
    print(data[num_feats].describe().T)
    
    # Plot numerical distributions
    plot_numerical_distributions(data, num_feats)
    
    # Pairplot
    plot_pairplot(data)
    
    # Regression plots
    plot_regression_plots(data)
    
    # Categorical distributions
    count_plot(data, cat_feats[0:-1])
    
    # ----------------------------------------------------------
    # 6.3 CORRELATION ANALYSIS
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Pearson correlation
    plot_pearson_correlation(data, num_feats)
    
    # Point-biserial correlation (need numeric data)
    data_encoded = label_encode_cat_features(data, cat_feats)
    plot_point_biserial_correlation(data_encoded)
    
    # Cramer's V correlation
    plot_cramers_v_correlation(data, cat_feats)
    
    # ----------------------------------------------------------
    # 6.4 PREPARE DATA FOR MODELING
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("PREPARING DATA FOR MODELING")
    print("=" * 60)
    
    # Label encode categorical features
    data = label_encode_cat_features(data, cat_feats)
    
    features = data.columns[:-1]
    X = data[features]
    y = data['target']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    
    # ----------------------------------------------------------
    # 6.5 TRAIN AND EVALUATE BASELINE MODELS
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING AND EVALUATING BASELINE MODELS")
    print("=" * 60)
    
    # Define classifiers
    names = [
        'Logistic Regression',
        'Nearest Neighbors',
        'Support Vectors',
        'Nu SVC',
        'Decision Tree',
        'Random Forest',
        'AdaBoost',
        'Gradient Boosting',
        'Naive Bayes',
        'Linear DA',
        'Quadratic DA',
        "Neural Net"
    ]
    
    classifiers = [
        LogisticRegression(solver="liblinear", random_state=SEED),
        KNeighborsClassifier(2),
        SVC(probability=True, random_state=SEED),
        NuSVC(probability=True, random_state=SEED),
        DecisionTreeClassifier(random_state=SEED),
        RandomForestClassifier(random_state=SEED),
        AdaBoostClassifier(random_state=SEED),
        GradientBoostingClassifier(random_state=SEED),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        MLPClassifier(random_state=SEED),
    ]
    
    # Get score summary
    results = score_summary(names, classifiers, X_train, y_train, X_val, y_val)
    results_sorted = results.sort_values(by='Accuracy', ascending=False)
    print("\nModel Performance Summary:")
    print(results_sorted.to_string(index=False))
    
    # Save results to CSV
    results_sorted.to_csv('plots/baseline_model_results.csv', index=False)
    print("Saved: plots/baseline_model_results.csv")
    
    # Plot ROC curves
    roc_auc_curve(names, classifiers, X_train, y_train, X_val, y_val)
    
    # Plot confusion matrices
    plot_conf_matrix(names, classifiers, X_train, y_train, X_val, y_val,
                     nrows=4, ncols=3, fig_a=12, fig_b=12)
    
    # ----------------------------------------------------------
    # 6.6 HYPERPARAMETER TUNING - LOGISTIC REGRESSION
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING - LOGISTIC REGRESSION")
    print("=" * 60)
    
    lr = LogisticRegression(tol=1e-4, max_iter=1000, random_state=SEED)
    
    space = dict(
        C=uniform(loc=0, scale=5),
        penalty=['l2', 'l1'],
        solver=['liblinear']
    )
    
    search = RandomizedSearchCV(
        lr,
        space,
        random_state=SEED,
        cv=5,
        scoring='f1'
    )
    
    rand_search = search.fit(X_train, y_train)
    
    print(f'Best Hyperparameters: {rand_search.best_params_}')
    
    params = rand_search.best_params_
    lr_tuned = LogisticRegression(**params)
    lr_tuned.fit(X_train, y_train)
    
    print("\nLogistic Regression (Tuned) Classification Report:")
    print(classification_report(y_val, lr_tuned.predict(X_val)))
    
    # Plot confusion matrix for tuned LR
    fig, ax = plt.subplots(figsize=(6, 5))
    y_pred_lr = lr_tuned.predict(X_val)
    cm_lr = confusion_matrix(y_val, y_pred_lr, labels=[0, 1])
    cm_lr_display = cm_lr[::-1]  # Flip rows
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_lr_display, display_labels=[1, 0])
    disp.plot(ax=ax)
    ax.set_xticklabels([0, 1])  # Fix X-axis labels
    ax.set_title('Logistic Regression (Tuned)')
    plt.savefig('plots/lr_tuned_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: plots/lr_tuned_confusion_matrix.png")
    
    # ----------------------------------------------------------
    # 6.7 BOOSTING MODELS
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING BOOSTING MODELS")
    print("=" * 60)
    
    names_boost = [
        'Catboost',
        'XGBoost',
        'LightGBM'
    ]
    
    classifiers_boost = [
        CatBoostClassifier(random_state=SEED, verbose=0),
        XGBClassifier(objective='binary:logistic', random_state=SEED, eval_metric='logloss'),
        LGBMClassifier(random_state=SEED, verbose=-1)
    ]
    
    results_boost = score_summary(names_boost, classifiers_boost, X_train, y_train, X_val, y_val)
    results_boost_sorted = results_boost.sort_values(by='Accuracy', ascending=False)
    print("\nBoosting Models Performance Summary:")
    print(results_boost_sorted.to_string(index=False))
    
    # Save results
    results_boost_sorted.to_csv('plots/boosting_model_results.csv', index=False)
    print("Saved: plots/boosting_model_results.csv")
    
    # Plot confusion matrices for boosting models
    plot_conf_matrix(names_boost, classifiers_boost, X_train, y_train, X_val, y_val,
                     nrows=1, ncols=3, fig_a=12, fig_b=3,
                     save_path='plots/boosting_confusion_matrices.png')
    
    # ----------------------------------------------------------
    # 6.8 HYPERPARAMETER TUNING - LIGHTGBM
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING - LIGHTGBM")
    print("=" * 60)
    
    rs_params = {
        'num_leaves': [20, 100],
        'max_depth': [5, 15],
        'min_data_in_leaf': [80, 120],
    }
    
    rs_cv = GridSearchCV(
        estimator=LGBMClassifier(random_state=SEED, verbose=-1),
        param_grid=rs_params,
        cv=5
    )
    
    rs_cv.fit(X_train, y_train)
    params = rs_cv.best_params_
    print(f"Best Parameters: {params}")
    
    lgbm = LGBMClassifier(**params, random_state=SEED, verbose=-1)
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
    )
    
    print("\nLightGBM (Tuned) Classification Report:")
    print(classification_report(y_val, lgbm.predict(X_val)))
    
    # Plot confusion matrix for tuned LGBM
    fig, ax = plt.subplots(figsize=(6, 5))
    y_pred_lgbm = lgbm.predict(X_val)
    cm_lgbm = confusion_matrix(y_val, y_pred_lgbm, labels=[0, 1])
    cm_lgbm_display = cm_lgbm[::-1]  # Flip rows
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_lgbm_display, display_labels=[1, 0])
    disp.plot(ax=ax)
    ax.set_xticklabels([0, 1])  # Fix X-axis labels
    ax.set_title('LightGBM (Tuned)')
    plt.savefig('plots/lgbm_tuned_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: plots/lgbm_tuned_confusion_matrix.png")
    
    # ----------------------------------------------------------
    # 6.9 FEATURE IMPORTANCE AND EXPLAINABILITY
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE AND EXPLAINABILITY")
    print("=" * 60)
    
    # Permutation Importance
    print("\nCalculating Permutation Importance...")
    perm_imp = PermutationImportance(lgbm, random_state=SEED).fit(X_train, y_train)
    
    # Get feature importances
    perm_importance = pd.DataFrame({
        'feature': X_val.columns.tolist(),
        'importance': perm_imp.feature_importances_,
        'std': perm_imp.feature_importances_std_
    }).sort_values('importance', ascending=False)
    
    print("\nPermutation Importance:")
    print(perm_importance.to_string(index=False))
    
    # Plot permutation importance
    plt.figure(figsize=(10, 6))
    plt.barh(perm_importance['feature'], perm_importance['importance'], 
             xerr=perm_importance['std'], color=MYPAL[0])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Permutation Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('plots/permutation_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: plots/permutation_importance.png")
    
    # SHAP Analysis
    print("\nCalculating SHAP values...")
    explainer = shap.TreeExplainer(lgbm)
    shap_values = explainer.shap_values(X_val)
    
    # SHAP Summary Bar Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_val, feature_names=features, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('plots/shap_summary_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: plots/shap_summary_bar.png")
    
    # SHAP Summary Dot Plot
    plt.figure(figsize=(10, 6))
    # For binary classification, use the positive class shap values
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X_val, show=False)
    else:
        shap.summary_plot(shap_values, X_val, show=False)
    plt.tight_layout()
    plt.savefig('plots/shap_summary_dot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: plots/shap_summary_dot.png")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nAll plots have been saved to the 'plots' directory.")
    
    return lgbm, X_train, X_val, y_train, y_val


# ============================================================
# 7. ENTRY POINT
# ============================================================
if __name__ == "__main__":
    model, X_train, X_val, y_train, y_val = main()