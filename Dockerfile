# Используем базовый образ Python 3.8
FROM python:3.8-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем все файлы проекта в контейнер
COPY . /app

# Устанавливаем зависимости из файла requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Открываем порт, на котором будет работать бот
EXPOSE 5000

# Запускаем бота
CMD ["python", "bot.py"]
