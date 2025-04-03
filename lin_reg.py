import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, mean_squared_error, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
from scipy.stats import t

# Загрузка данных и предобработка
file_path = 'info.xlsx'
data = pd.read_excel(file_path, sheet_name=0)

# Определение целевой переменной (по названию "BI-RADS")
target = data['BI-RADS']

data = data.drop(columns=['BI-RADS', 'СЭГ заключение', 'Размер пальпация (мм)', 'Клиничекий диагноз (КД)', 'УЗ заключение', 'ЭГтипы','CEUS заключение'])

# Разделение числовых и категориальных данных
numerical_data = data.select_dtypes(include=['int64', 'float64'])
categorical_data = data.select_dtypes(include=['object'])

# Масштабирование числовых данных
scaler = StandardScaler()
scaled_numerical_data = scaler.fit_transform(numerical_data)
print(scaled_numerical_data)

# Кодирование категориальных данных
le = LabelEncoder()
for col in categorical_data.columns:
    categorical_data[col] = le.fit_transform(categorical_data[col].astype(str))

# Объединение числовых и категориальных данных без таргета
features = np.hstack([scaled_numerical_data, categorical_data.values])

# Разделение данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Создание и обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test)

# Округление предсказанных значений до ближайшего класса (2, 3, 5)
class_labels = [2, 3, 5]
y_pred_rounded = np.array([min(class_labels, key=lambda x: abs(x - pred)) for pred in y_pred])

# Оценка модели: F-score по каждому классу и среднеквадратичная ошибка
print("\nОтчет классификации:")
print(classification_report(y_test, y_pred_rounded, target_names=['Class 2', 'Class 3', 'Class 5']))

mse = mean_squared_error(y_test, y_pred_rounded)
print(f"\nСреднеквадратичная ошибка: {mse:.2f}")

# Построение графической матрицы ошибок
conf_matrix = confusion_matrix(y_test, y_pred_rounded, labels=class_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 2', 'Class 3', 'Class 5'], yticklabels=['Class 2', 'Class 3', 'Class 5'])
plt.title('Confusion Matrix for Linear Regression')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Оценка важности признаков
numerical_features = numerical_data.columns.tolist()
categorical_features = categorical_data.columns.tolist()
feature_names = numerical_features + categorical_features

coefficients = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model.coef_
})

coefficients['Absolute Coefficient'] = coefficients['Coefficient'].abs()
coefficients = coefficients.sort_values(by='Absolute Coefficient', ascending=False)

# Вывод значения биаса
bias = model.intercept_
print(f"\nЗначение биаса (свободного члена): {bias:.4f}")

# Вывод таблицы коэффициентов
print("\nТаблица коэффициентов линейной регрессии:")
print(coefficients[['Feature', 'Coefficient']])


# Бинаризация целевой переменной для уменьшенного количества классов
y_test_binarized = label_binarize(y_test, classes=class_labels)
n_classes = y_test_binarized.shape[1]

# Преобразуем предсказанные значения в вероятности для каждого класса
y_pred_proba = np.zeros((y_pred.shape[0], n_classes))
for i, pred in enumerate(y_pred):
    # Заполняем вероятность принадлежности к каждому классу в зависимости от округления
    for j, class_label in enumerate(class_labels):
        y_pred_proba[i, j] = 1 - abs(pred - class_label)

# Построение ROC-кривой для линейной регрессии
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Построение ROC-кривой для каждого класса
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {class_labels[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Linear Regression (Classes 2, 3, 5)')
plt.legend(loc='lower right')
plt.show()

# ROC-AUC в среднем по всем классам (macro)
roc_auc_macro = roc_auc_score(y_test_binarized, y_pred_proba, average='macro')
print(f'Linear Regression ROC-AUC (macro): {roc_auc_macro:.4f}')

# Функция для вычисления t-статистики
def t_statistic(X_train, y_train, y_pred, coef, numb_features):
    n = len(y_train)
    # Вычисление остатков (разница между фактическими и предсказанными значениями)
    residuals = y_train - y_pred
    # Вычисление суммы квадратов ошибок (SSE)
    sse = np.sum(residuals ** 2)
    # Стандартная ошибка модели
    sigma = np.sqrt(sse / (n - numb_features))

    # Стандартная ошибка для каждого коэффициента
    X = np.hstack([np.ones((X_train.shape[0], 1)), X_train])  # Добавляем столбец единиц для b_0
    XtX_inv = np.linalg.inv(X.T @ X)  # (X^T X)^-1
    se = np.sqrt(np.diag(XtX_inv)) * sigma  # Стандартная ошибка коэффициентов

    # t-статистика
    t_values = coef / se
    return t_values, se


# Использование функции для проверки значимости коэффициентов
y_pred_train = model.predict(X_train)  # Предсказанные значения на тренировочной выборке
coef = np.append(model.intercept_, model.coef_)  # Включаем b_0 и остальные коэффициенты

class_labels = [2, 3, 5]
y_pred_rounded = np.array([min(class_labels, key=lambda x: abs(x - pred)) for pred in y_pred_train])

t_values, standard_errors = t_statistic(X_train, y_train, y_pred_rounded, coef, len(data.columns))

# Названия признаков, включая свободный член
features = ['Intercept'] + numerical_features + categorical_features

# Критическое значение для уровня значимости 0.05
alpha = 0.05
n = len(y_train)
t_critical = t.ppf(1 - alpha / 2, df=n - 2)

# Выводим результаты для каждого коэффициента
print(f"Критическое значение t: {t_critical}\n")
print("Результаты значимости коэффициентов:\n")
for i, feature in enumerate(features):
    is_significant = np.abs(t_values[i]) > t_critical
    result = "Значим" if is_significant else "Не значим"
    print(
        f"Коэффициент: {feature[:30]}, t-статистика: {t_values[i]:.4f}, Стандартная ошибка: {standard_errors[i]:.4f}, Результат: {result}")