import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Загрузка данных из файла Excel
file_path = 'info.xlsx'
data = pd.read_excel(file_path, sheet_name=0)

# Разделение численных и категориальных данных
numerical_data = data.select_dtypes(include=['int64', 'float64'])
categorical_data = data.select_dtypes(include=['object'])

# Кодирование категориальных данных
le = LabelEncoder()
for col in categorical_data:
    data[col] = le.fit_transform(data[col].astype(str))

# Ограничиваемся первыми 32 столбцами для визуализации
encoded_data = data.iloc[:, :30]

# Функция для построения гистограмм по 8 признаков за раз
def plot_histograms_in_batches(data, batch_size=6):
    num_columns = data.shape[1]
    column_names = data.columns

    # Проход по столбцам с шагом batch_size
    for start_col in range(0, num_columns, batch_size):
        plt.figure(figsize=(15, 10))

        for i in range(batch_size):
            col_idx = start_col + i
            if col_idx >= num_columns:
                break

            # Построение гистограммы для отдельного признака
            plt.subplot(2, 4, i + 1)  # 2 строки по 4 графика в каждой
            sns.histplot(data.iloc[:, col_idx], kde=False, bins=20)
            plt.title(column_names[col_idx], fontsize=12)
            plt.ylabel('')
            plt.xlabel('')

        plt.tight_layout()
        plt.show()

# Вывод гистограмм по 8 признаков за раз
plot_histograms_in_batches(encoded_data, batch_size=8)
