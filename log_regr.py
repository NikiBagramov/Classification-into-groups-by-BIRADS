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

# Загрузка данных и предобработка
file_path = 'info.xlsx'
data = pd.read_excel(file_path, sheet_name=0)

# Определение целевой переменной (по названию "BI-RADS")
target = data['BI-RADS']

data = data.drop(columns=['BI-RADS', 'СЭГ заключение', 'Размер пальпация (мм)', 'Клиничекий диагноз (КД)', 'УЗ заключение', 'ЭГтипы','CEUS заключение'])

print(len(data.columns))

# Явное указание классов 2, 3, 5
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

# Логистическая регрессия
log_reg_model = LogisticRegression(random_state=42, max_iter=6999)
log_reg_model.fit(X_train, y_train)
y_pred_log_reg = log_reg_model.predict(X_test)

# Оценка логистической регрессии
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
f1_log_reg = f1_score(y_test, y_pred_log_reg, average='weighted')

print('\nLogistic Regression Classification Report:')
print(classification_report(y_test, y_pred_log_reg, labels=class_labels, zero_division=0))

# Вывод графической матрицы ошибок для логистической регрессии
conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg, labels=class_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_log_reg, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# Бинаризация целевой переменной для многоклассовой задачи
y_test_binarized = label_binarize(y_test, classes=class_labels)
n_classes = y_test_binarized.shape[1]

y_score_log_reg = log_reg_model.predict_proba(X_test)

# Построение ROC-кривой для логистической регрессии
fpr_log_reg = dict()
tpr_log_reg = dict()
roc_auc_log_reg = dict()
for i in range(n_classes):
    fpr_log_reg[i], tpr_log_reg[i], _ = roc_curve(y_test_binarized[:, i], y_score_log_reg[:, i])
    roc_auc_log_reg[i] = auc(fpr_log_reg[i], tpr_log_reg[i])

# ROC-AUC для логистической регрессии
for i in range(n_classes):
    plt.plot(fpr_log_reg[i], tpr_log_reg[i], label=f'Class {class_labels[i]} (AUC = {roc_auc_log_reg[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc='lower right')
plt.show()

# ROC-AUC для логистической регрессии в целом (средний показатель)
roc_auc_log_reg_macro = roc_auc_score(y_test_binarized, y_score_log_reg, average='macro')
print(f'Logistic Regression ROC-AUC (macro): {roc_auc_log_reg_macro:.4f}')
