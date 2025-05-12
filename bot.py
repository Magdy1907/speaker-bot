import telebot
import numpy as np
import librosa
import subprocess
import os
from tensorflow.keras.models import load_model

# 🔐 Токен Telegram-бота
TOKEN = "7424010381:AAF1_4x5XJpUj7V_d0KgmbZynggT7bJqxvg"
bot = telebot.TeleBot(TOKEN)

# 📦 Загрузка обученной модели
model = load_model("speaker_classifier.keras")

# 🏷️ Словарь классов (имена говорящих)
labels = {
    0: "Анна",
    1: "Бабушка",
    2: "Влад",
    3: "Дедушка",
    4: "Никита"
}

# 💬 Ответ на текстовые сообщения
@bot.message_handler(content_types=['text'])
def handle_text(message):
    bot.reply_to(
        message,
        "👋 Привет! Отправь мне аудиофайл (WAV, OGG, MP3), и я скажу, кто говорит."
    )

# 🎧 Обработка аудиофайлов
@bot.message_handler(content_types=['audio', 'document'])
def handle_audio(message):
    try:
        # 📥 Получение файла
        file_info = bot.get_file(message.audio.file_id if message.audio else message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        # 📝 Сохранение оригинального файла
        ext = os.path.splitext(file_info.file_path)[1]
        input_filename = f"input{ext}"
        with open(input_filename, 'wb') as f:
            f.write(downloaded_file)

        # 🔄 Конвертация в WAV, если нужно
        if ext.lower() != '.wav':
            subprocess.call(['ffmpeg', '-y', '-i', input_filename, 'converted.wav'])
            input_path = 'converted.wav'
        else:
            input_path = input_filename

        # 🧠 Извлечение MFCC
        y, sr = librosa.load(input_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T

        # 📏 Приведение к нужной длине
        max_len = model.input_shape[1]
        if mfcc.shape[0] > max_len:
            mfcc = mfcc[:max_len]
        else:
            mfcc = np.pad(mfcc, ((0, max_len - mfcc.shape[0]), (0, 0)))

        mfcc = np.expand_dims(mfcc, axis=0)

        # 🔍 Предсказание
        pred = model.predict(mfcc)
        max_prob = np.max(pred)
        pred_class = np.argmax(pred)

        if max_prob < 0.7:
            bot.reply_to(message, "❌ Не удалось распознать голос. Попробуй другой файл.")
        else:
            bot.reply_to(message, f"🗣️ Говорящий: {labels[pred_class]} (достоверность: {max_prob:.2f})")

        # 🧹 Удаление временных файлов
        os.remove(input_filename)
        if os.path.exists("converted.wav"):
            os.remove("converted.wav")

    except Exception as e:
        bot.reply_to(message, f"⚠️ Ошибка: {e}")

# 🚫 Удаляем вебхук (на всякий случай)
bot.remove_webhook()

# ▶️ Запуск бота
bot.polling()
