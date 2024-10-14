import pandas as pd
import os

# Данные
data = [
    [10, 1086, 810],
    [10, 1530, 864],
    [9, 1150, 420],
    [9, 1300, 650],
    [5, 1440, 1040],
]

# Создание DataFrame
df = pd.DataFrame(data, columns=["Column1", "Column2", "Column3"])

# Сохранение в CSV файл
file_path = os.path.join('D:\Гиперграфы', 'data.csv')
df.to_csv(file_path, index=False)

print("CSV файл создан.")