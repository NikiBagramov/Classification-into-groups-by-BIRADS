import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import statsmodels.api as sm

# Загрузка данных и предобработка
file_path = 'info.xlsx'
data = pd.read_excel(file_path, sheet_name=0)

# Определение целевой переменной (по названию "BI-RADS")
target = data['BI-RADS']

data = data.drop(columns=['BI-RADS', 'СЭГ заключение', 'Размер пальпация (мм)', 'Клиничекий диагноз (КД)', 'УЗ заключение', 'ЭГтипы','CEUS заключение']) #'BI-RADS', 'СЭГ заключение', 'Размер пальпация (мм)'

print(len(data.columns))

# Явное указание классов 2, 3, 4, 5
class_labels = [2, 3, 5]

# Разделение числовых и категориальных данных
numerical_data = data.select_dtypes(include=['int64', 'float64'])
categorical_data = data.select_dtypes(include=['object'])

# Масштабирование числовых данных
scaler = StandardScaler()
scaled_numerical_data = scaler.fit_transform(numerical_data)

# Кодирование категориальных данных
le = LabelEncoder()
for col in categorical_data.columns:
    categorical_data[col] = le.fit_transform(categorical_data[col].astype(str))

# Объединение числовых и категориальных данных без таргета
features = np.hstack([scaled_numerical_data, categorical_data.values])

# Стратифицированное разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42, stratify=target)

# Проверка распределения классов в тренировочной и тестовой выборках
print("Train target distribution:", np.bincount(y_train))
print("Test target distribution:", np.bincount(y_test))

# Параметры для случайного леса, которые мы будем настраивать
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [50, 100, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', 'balanced_subsample']
}

# Инициализация модели случайного леса
rf_model = RandomForestClassifier(random_state=42)

# Инициализация GridSearchCV для поиска наилучших параметров
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=2, scoring='f1_weighted')

# Обучение модели на тренировочных данных
grid_search.fit(X_train, y_train)

# Лучшие параметры
print("Best parameters found: ", grid_search.best_params_)

# Предсказания с использованием лучшей модели
best_rf_model = grid_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)

# Оценка модели с наилучшими параметрами
accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

# Вывод результатов для случайного леса после обучения по сетке
print(f'Random Forest with Grid Search - Accuracy: {accuracy_rf:.4f}, F1 Score (weighted): {f1_rf:.4f}')
print('\nRandom Forest Classification Report (Grid Search):')
print(classification_report(y_test, y_pred_rf, labels=class_labels, zero_division=0))

# Вывод графической матрицы ошибок для случайного леса
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf, labels=class_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Бинаризация целевой переменной для многоклассовой задачи
y_test_binarized = label_binarize(y_test, classes=class_labels)
n_classes = y_test_binarized.shape[1]

# Получение вероятностей предсказаний для случайного леса
y_score_rf = best_rf_model.predict_proba(X_test)

# Построение ROC-кривой для случайного леса
fpr_rf = dict()
tpr_rf = dict()
roc_auc_rf = dict()
for i in range(n_classes):
    fpr_rf[i], tpr_rf[i], _ = roc_curve(y_test_binarized[:, i], y_score_rf[:, i])
    roc_auc_rf[i] = auc(fpr_rf[i], tpr_rf[i])

# ROC-AUC для случайного леса
for i in range(n_classes):
    plt.plot(fpr_rf[i], tpr_rf[i], label=f'Class {class_labels[i]} (AUC = {roc_auc_rf[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc='lower right')
plt.show()

# ROC-AUC для случайного леса в целом (средний показатель)
roc_auc_rf_macro = roc_auc_score(y_test_binarized, y_score_rf, average='macro')
print(f'Random Forest ROC-AUC (macro): {roc_auc_rf_macro:.4f}')