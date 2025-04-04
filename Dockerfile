# Используем базовый образ Python 3.9
FROM python:3.9-slim

# Устанавливаем зависимости для FFmpeg
RUN apt-get update && apt-get install -y ffmpeg

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
