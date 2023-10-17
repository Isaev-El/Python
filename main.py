import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

# Загрузим исторические данные курса валюты
data = pd.read_csv('currency_data.csv')
prices = data['ClosingPrice'].values

# Масштабирование данных от 0 до 1
scaler = MinMaxScaler()
prices = scaler.fit_transform(prices.reshape(-1, 1))

# Создаем последовательные временные шаги для обучения
X = []
y = []
look_back = 10  # Количество предыдущих дней, используемых для прогноза

for i in range(len(prices) - look_back):
    X.append(prices[i:i+look_back])
    y.append(prices[i+look_back])

X = np.array(X)
y = np.array(y)

# Разделим данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=False)
# Создаем модель RNN
model = Sequential()
model.add(SimpleRNN(50, activation='relu', input_shape=(look_back, 1)))
model.add(Dense(1))

# Скомпилируем модель
model.compile(optimizer='adam', loss='mean_squared_error')

# Обучаем модель
model.fit(X_train, y_train, epochs=100, batch_size=64)

# Проведем прогноз
predicted_prices = model.predict(X_test)

# Инвертируем масштабирование данных, чтобы получить реальные цены
predicted_prices = scaler.inverse_transform(predicted_prices)
y_test = scaler.inverse_transform(y_test)

# Отобразим результаты
plt.plot(predicted_prices, label='Predicted Prices')
plt.plot(y_test, label='Actual Prices')
plt.legend()
plt.show()
