FROM python:3.11-slim

# Установка необходимых системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libfreetype6-dev \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app

# Копирование и установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Открытие порта для API
EXPOSE 8000

# Запуск API сервера
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
