import pandas as pd
import numpy as np
# Graphs
import matplotlib.pyplot as plt

# Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer, OneHotEncoder
from sklearn.model_selection import train_test_split

# Metrics
from sklearn.metrics import roc_curve, roc_auc_score, auc, ConfusionMatrixDisplay, classification_report
from sklearn import tree

# SHAP 
import shap
from imblearn.over_sampling import SMOTE

def train_preprocessing(df, encoder_type='LabelEncoder', normalizer='StandardScaler',
                        pca=False, pca_components=None, random_state=0):

    df_aux = df.copy()

    list_categorical = df_aux.select_dtypes(include='object').columns

    enc = dict()
    if encoder_type == 'LabelEncoder':
        for col in list_categorical:
            enc[col] = LabelEncoder()
            df_aux[col] = enc[col].fit_transform(df_aux[col])

    elif encoder_type == 'OneHotEncoder':
        for col in list_categorical:
            enc[col] = OneHotEncoder(handle_unknown='ignore')
            ohe_results = enc[col].fit_transform(df_aux[[col]])
            df1 = pd.DataFrame(ohe_results.toarray(), columns=[f'{col}_{name}' for name in enc[col].categories_[0]], index=df_aux[col].index)
            df_aux = df_aux.merge(df1, how='left', left_index=True, right_index=True)

        df_aux.drop(columns=list_categorical, inplace=True)

    feat_cols = df_aux.columns

    if normalizer == 'StandardScaler':
        norm = StandardScaler()
    elif normalizer == 'MinMaxScaler':
        norm = MinMaxScaler((0, 1))
    elif normalizer == 'MaxAbsScaler':
        norm = MaxAbsScaler()
    elif normalizer == 'QuantileTransformer':
        norm = QuantileTransformer(output_distribution='normal')
    
    df_aux = norm.fit_transform(df_aux)

    if pca:
        pca = PCA(pca_components, random_state=random_state)
        df_aux = pca.fit_transform(df_aux)

        return df_aux, enc, norm, pca, feat_cols

    else:
        return df_aux, enc, norm, feat_cols

#-------------------------------------------------------------------------------
def get_train_test(df, drop_cols, label, test_size=0.25, random_state=0):

    df_aux = df.copy()

    cols = df_aux.columns.drop(drop_cols)
    lb = df_aux[label].copy()
    cols = cols.drop(label)
    feat = df_aux[cols]

    X_train, X_test, y_train, y_test = train_test_split(feat, lb, 
                                                        test_size=test_size, 
                                                        random_state=random_state,
                                                        stratify=lb)

    return X_train, X_test, y_train, y_test

#-------------------------------------------------------------------------------
#Função para preprocessemanto
def preprocessing(df, cols_drop, label, test_size=0.25, encoder_type='LabelEncoder',
                  norm_name='StandardScaler', return_enc_norm=False, pca=False, 
                  pca_components=None, balance_data=True, group_years=False,
                  first_year=None, last_year=None, morpho3=False, random_state=0):
# Inicializa df_aux com df
    df_aux = df.copy()
# Grouped years
    if group_years and first_year != None and last_year != None:
        df_aux = df_aux[(df_aux.ANODIAG >= first_year) & (df_aux.ANODIAG <= last_year)].copy()
        
    # Train Test split
    X_train, X_test, y_train, y_test = get_train_test(df_aux, cols_drop, label, 
                                                      test_size, 
                                                      random_state=random_state)

    # Preprocessing
    if pca and pca_components != None:
        X_train_enc, enc, norm, pca, feat_cols = train_preprocessing(X_train, encoder_type=encoder_type, 
                                                                     normalizer=norm_name, pca=pca,
                                                                     pca_components=pca_components,
                                                                     random_state=random_state)
        X_test_ = test_preprocessing(X_test, enc, norm, 
                                     encoder_type, pca)

    else:
        X_train_enc, enc, norm, feat_cols = train_preprocessing(X_train, encoder_type=encoder_type,
                                                                normalizer=norm_name)
        X_test_ = test_preprocessing(X_test, enc, norm, encoder_type)

    # Balancing
    if balance_data:
        X_train_, y_train_ = SMOTE(random_state=random_state).fit_resample(X_train_enc, y_train)
    
    else:
        X_train_, y_train_ = X_train_enc, y_train

    print(f'X_train = {X_train_.shape}, X_test = {X_test_.shape}')
    print(f'y_train = {y_train_.shape}, y_test = {y_test.shape}')

    if return_enc_norm:
        return X_train_, X_test_, y_train_, y_test, feat_cols, enc, norm
    else:
        return X_train_, X_test_, y_train_, y_test, feat_cols
#-------------------------------------------------------------------------------
def test_preprocessing(df, enc, norm, encoder_type='LabelEncoder', pca=None):

    df_aux = df.copy()

    df_aux.fillna(0, inplace=True)

    list_categorical = df_aux.select_dtypes(include='object').columns

    if encoder_type == 'LabelEncoder':
        for col in list_categorical:
            df_aux.loc[~df_aux[col].isin(enc[col].classes_), col] = -1 
            df_aux.loc[df_aux[col].isin(enc[col].classes_), col] = enc[col].transform(df_aux[col][df_aux[col].isin(enc[col].classes_)])
    
    elif encoder_type == 'OneHotEncoder':
        for col in list_categorical:
            ohe_results = enc[col].transform(df_aux[[col]])
            df1 = pd.DataFrame(ohe_results.toarray(), columns=[f'{col}_{name}' for name in enc[col].categories_[0]], index=df_aux[col].index)
            df_aux = df_aux.merge(df1, how='left', left_index=True, right_index=True)

        df_aux.drop(columns=list_categorical, inplace=True)

    df_aux = norm.transform(df_aux)

    if pca != None:
        df_aux = pca.transform(df_aux)

    return df_aux 
#-------------------------------------------------------------------------------
def show_tree(model, feat_cols, max_depth=3, estimator=0):

    
    plt.figure(figsize = (22, 10))
    tree.plot_tree(model.estimators_[estimator],
                   feature_names=feat_cols,
                   filled=True, 
                   max_depth=max_depth);
    
    plt.show()
#-------------------------------------------------------------------------------
def plot_shap_values(model, x, features, max_display=10):
    shap_values = shap.TreeExplainer(model).shap_values(x)
    
    # Ajustar o tamanho da figura antes de plotar
    plt.figure(figsize=(12, 8))
    
    try:
        shap.summary_plot(shap_values[1], x, 
                          feature_names=features,
                          max_display=max_display,
                          plot_size=(12, 8))  # Ajuste o plot_size 
    except AssertionError:
        shap.summary_plot(shap_values, x, 
                          feature_names=features,
                          max_display=max_display,
                          plot_size=(12, 8))  # Ajuste o plot_size 
    
    plt.show()

#-------------------------------------------------------------------------------
def plot_confusion_matrix(model, x, y, format='.3f'):

    with plt.rc_context({'font.size': 12, 'font.weight': 'bold'}):
        ConfusionMatrixDisplay.from_estimator(model, x, y, values_format=format,
                                              cmap='Blues', normalize='true')
        plt.show()

    print(f'\n{classification_report(y, model.predict(x), digits=3)}')
#-------------------------------------------------------------------------------
def plot_roc_curve(model, X_train, X_test, y_train, y_test):
    probas_train = model.predict_proba(X_train)[:, 1]
    probas_test = model.predict_proba(X_test)[:, 1]

    fp_train, tp_train, _ = roc_curve(y_train, probas_train)
    fp_test, tp_test, _ = roc_curve(y_test, probas_test)

    plt.figure(figsize=(10, 7))
    plt.plot(fp_train, tp_train, 'b', label=f'Train (AUC = {auc(fp_train, tp_train):.3f})')
    plt.plot(fp_test, tp_test, 'r', label=f'Test (AUC = {auc(fp_test, tp_test):.3f})')
    plt.plot(np.linspace(0, 1, 100),
             np.linspace(0, 1, 100),
             label='Baseline',
             linestyle='--', 
             color='k')
    plt.xlabel('False Positives')
    plt.ylabel('True Positives')
    plt.grid(True)
    plt.legend()
    plt.show()
#-------------------------------------------------------------------------------
def plot_feat_importances(model, feat_cols, n=10):
    feat_import = pd.Series(model.feature_importances_, index=feat_cols)
    feat_import.nlargest(n).plot(kind='barh', figsize=(10, 8))
    plt.show()
#-------------------------------------------------------------------------------