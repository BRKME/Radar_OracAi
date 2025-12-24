# 🚀 Быстрый старт BTC Forecast Bot

## Шаг 1: Получите API ключи (10 минут)

### 1.1 Telegram Bot Token
```
1. Найдите @BotFather в Telegram
2. Отправьте /newbot
3. Придумайте название и username
4. Скопируйте токен (выглядит как: 1234567890:ABCdefGHIjklMNOpqrsTUVwxyz)
```

### 1.2 Telegram Channel
```
1. Создайте публичный канал в Telegram
2. Добавьте бота как администратора канала
3. ID канала = @ваш_канал (например: @btc_forecast)
```

### 1.3 OpenAI API Key
```
1. Зайдите на platform.openai.com
2. Sign Up / Log In
3. API Keys → Create new secret key
4. Скопируйте ключ (sk-...)
```

### 1.4 FRED API Key (опционально)
```
1. Зайдите на fred.stlouisfed.org
2. My Account → API Keys
3. Request API Key
4. Скопируйте ключ
```

## Шаг 2: Настройка GitHub (5 минут)

### 2.1 Создайте репозиторий
```bash
# На GitHub.com:
New repository → btc-forecast-bot → Create

# Локально:
git clone https://github.com/ваш-username/btc-forecast-bot.git
cd btc-forecast-bot

# Скопируйте все файлы из этого проекта
# Затем:
git add .
git commit -m "Initial commit"
git push
```

### 2.2 Добавьте Secrets
```
GitHub → Settings → Secrets and variables → Actions → New secret

Добавьте:
- TELEGRAM_BOT_TOKEN (обязательно)
- TELEGRAM_CHANNEL_ID (обязательно)  
- OPENAI_API_KEY (обязательно)
- FRED_API_KEY (опционально)
```

## Шаг 3: Запуск (1 минута)

### Вариант А: Вручную
```
GitHub → Actions → BTC Forecast Bot → Run workflow
```

### Вариант Б: По расписанию
```
Workflow запустится автоматически каждый понедельник в 12:00 UTC
```

## Шаг 4: Проверка

1. Зайдите в Actions → посмотрите логи
2. Откройте ваш Telegram канал
3. Должно появиться сообщение с прогнозом

## 🔧 Настройка расписания

Откройте `.github/workflows/forecast.yml`:

```yaml
schedule:
  - cron: '0 12 * * 1'  # Понедельник 12:00 UTC
```

Примеры:
- `'0 9 * * *'` - каждый день в 09:00
- `'0 12 * * 1,3,5'` - Пн, Ср, Пт в 12:00
- `'0 0,12 * * *'` - каждый день в 00:00 и 12:00

## ⚡ Локальный тест (опционально)

```bash
# Установите зависимости
pip install -r requirements.txt

# Создайте .env файл
cp .env.example .env
# Заполните .env вашими ключами

# Запустите тест
python test_local.py

# Или сразу публикация
python main.py
```

## 📋 Чеклист

- [ ] ✅ Получил Telegram Bot Token
- [ ] ✅ Создал Telegram канал и добавил бота
- [ ] ✅ Получил OpenAI API Key
- [ ] ✅ Создал репозиторий на GitHub
- [ ] ✅ Добавил Secrets в GitHub
- [ ] ✅ Запустил workflow вручную
- [ ] ✅ Получил первый прогноз в канале

## 🆘 Проблемы?

**Бот не публикует в канал:**
- Проверьте, что бот добавлен как администратор канала
- Проверьте правильность TELEGRAM_CHANNEL_ID (должно быть @channel или -100...)

**Ошибка OpenAI API:**
- Проверьте, что на аккаунте есть баланс ($5+)
- Проверьте правильность ключа (должен начинаться с sk-)

**Ошибка получения данных:**
- GitHub Actions может иметь ограничения по сети
- Попробуйте запустить локально для диагностики

**Другие вопросы:**
- Смотрите логи в GitHub Actions
- Запустите `python test_local.py` для детальной диагностики

---

**Готово! Ваш BTC Forecast Bot работает! 🎉**
