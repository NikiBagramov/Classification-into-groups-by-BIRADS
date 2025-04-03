from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Функция для расчета VIF
def calculate_vif(data):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return vif_data

# Итеративное удаление признаков с наибольшим VIF
def remove_high_vif_feature(data, removed_features):
    vif_data = calculate_vif(data)
    print(vif_data)  # Вывод текущих значений VIF в консоль
    max_vif_feature = vif_data.loc[vif_data['VIF'].idxmax(), 'Feature']
    print(f"\nУдаляем признак: {max_vif_feature} с VIF: {vif_data['VIF'].max()}\n")
    removed_features.append(max_vif_feature)  # Добавляем удаленный признак в список
    updated_data = data.drop(columns=[max_vif_feature])
    return updated_data

# Загрузка данных и предобработка
file_path = 'info.xlsx'
data = pd.read_excel(file_path, sheet_name=0)

# Явное указание классов 2, 3, 4, 5
class_labels = [2, 3, 4, 5]

# Разделение числовых и категориальных данных
numerical_data = data.select_dtypes(include=['int64', 'float64'])
categorical_data = data.select_dtypes(include=['object'])

# Определение целевой переменной (по названию "BI-RADS")
target = data['BI-RADS']
numerical_data = numerical_data.drop(columns=['BI-RADS'])

# Масштабирование числовых данных
scaler = StandardScaler()
scaled_numerical_data = scaler.fit_transform(numerical_data)

# Кодирование категориальных данных
le = LabelEncoder()
for col in categorical_data.columns:
    categorical_data[col] = le.fit_transform(categorical_data[col].astype(str))

# Объединение данных после обработки
processed_data = pd.DataFrame(scaled_numerical_data, columns=numerical_data.columns)
processed_data = pd.concat([processed_data, categorical_data], axis=1)

# Список для хранения удаленных признаков
removed_features = []

# Итеративный процесс удаления признаков с пересчетом VIF
while True:
    processed_data = remove_high_vif_feature(processed_data, removed_features)
    user_input = input("Нажмите Enter для удаления следующего признака с наибольшим VIF или введите 'q' для выхода: ")
    if user_input.lower() == 'q':
        print("\nЗавершение работы.")
        print("Удаленные признаки: ", removed_features)
        break
