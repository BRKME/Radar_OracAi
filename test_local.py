"""
Скрипт для тестирования BTC Forecast Bot локально
Выводит результаты в консоль вместо Telegram
"""

import os
from datetime import datetime
from main import BTCForecastBot
from dotenv import load_dotenv


def test_data_fetching():
    """Тест получения данных"""
    print("\n" + "="*60)
    print("ТЕСТ 1: Получение данных BTC")
    print("="*60)
    
    bot = BTCForecastBot()
    
    try:
        df_1h = bot.fetch_btc_data('1h', limit=168)
        print(f"✅ Получено {len(df_1h)} свечей для 1h таймфрейма")
        print(f"   Последняя цена: ${df_1h['close'].iloc[-1]:,.2f}")
        
        df_1d = bot.fetch_btc_data('1d', limit=90)
        print(f"✅ Получено {len(df_1d)} свечей для 1d таймфрейма")
        
        df_1w = bot.fetch_btc_data('1w', limit=52)
        print(f"✅ Получено {len(df_1w)} свечей для 1w таймфрейма")
        
        return True, bot, df_1h, df_1d, df_1w
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False, None, None, None, None


def test_technical_analysis(bot, df_1h, df_1d, df_1w):
    """Тест технического анализа"""
    print("\n" + "="*60)
    print("ТЕСТ 2: Технический анализ")
    print("="*60)
    
    try:
        ta_weekly = bot.calculate_technical_indicators(df_1h)
        print(f"✅ Недельный TA:")
        print(f"   RSI: {ta_weekly['rsi']:.2f}")
        print(f"   MACD: {ta_weekly['macd']:.4f}")
        print(f"   Тренд: {ta_weekly['trend']}")
        print(f"   Поддержка: ${ta_weekly['support']:,.2f}")
        print(f"   Сопротивление: ${ta_weekly['resistance']:,.2f}")
        
        ta_monthly = bot.calculate_technical_indicators(df_1d)
        print(f"✅ Месячный TA:")
        print(f"   EMA50: ${ta_monthly['ema50']:,.2f}")
        print(f"   EMA200: ${ta_monthly['ema200']:,.2f}")
        
        ta_yearly = bot.calculate_technical_indicators(df_1w)
        print(f"✅ Годовой TA:")
        print(f"   Долгосрочный тренд: {ta_yearly['trend']}")
        
        return True, ta_weekly, ta_monthly, ta_yearly
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False, None, None, None


def test_macro_data(bot, df_1d):
    """Тест получения макроэкономических данных"""
    print("\n" + "="*60)
    print("ТЕСТ 3: Макроэкономические данные")
    print("="*60)
    
    try:
        sp500_data = bot.fetch_sp500_data()
        if sp500_data['current_price']:
            print(f"✅ S&P 500: ${sp500_data['current_price']:,.2f}")
            print(f"   Изменение за месяц: {sp500_data['change_1m']:+.2f}%")
        else:
            print("⚠️  S&P 500: данные недоступны")
        
        macro_data = bot.fetch_macro_data()
        if macro_data['fed_rate']:
            print(f"✅ Ставка ФРС: {macro_data['fed_rate']:.2f}%")
        else:
            print("⚠️  Ставка ФРС: данные недоступны (нужен FRED_API_KEY)")
        
        correlation = bot.calculate_btc_sp500_correlation(df_1d)
        if correlation is not None:
            print(f"✅ Корреляция BTC-SPX: {correlation:.2f}")
        else:
            print("⚠️  Корреляция: не удалось рассчитать")
        
        return True, sp500_data, macro_data, correlation
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False, None, None, None


def test_ai_forecast(bot, ta_weekly, ta_monthly, ta_yearly, sp500_data, macro_data, correlation):
    """Тест генерации AI прогноза"""
    print("\n" + "="*60)
    print("ТЕСТ 4: Генерация AI прогноза")
    print("="*60)
    
    try:
        forecast = bot.generate_ai_forecast(
            ta_weekly, ta_monthly, ta_yearly,
            sp500_data, macro_data, correlation
        )
        print("✅ AI прогноз успешно сгенерирован")
        print("\nПРОГНОЗ:")
        print("-" * 60)
        print(forecast)
        print("-" * 60)
        
        return True, forecast
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False, None


def test_message_formatting(bot, ta_weekly, forecast):
    """Тест форматирования сообщения"""
    print("\n" + "="*60)
    print("ТЕСТ 5: Форматирование сообщения для Telegram")
    print("="*60)
    
    try:
        message = bot.format_telegram_message(ta_weekly, forecast)
        print("✅ Сообщение успешно отформатировано")
        print("\nИТОГОВОЕ СООБЩЕНИЕ:")
        print("="*60)
        # Удаляем HTML теги для читабельности в консоли
        clean_message = message.replace('<b>', '').replace('</b>', '')
        clean_message = clean_message.replace('<i>', '').replace('</i>', '')
        print(clean_message)
        print("="*60)
        
        return True, message
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False, None


def main():
    """Запуск всех тестов"""
    print("\n" + "🧪 " + "="*58 + " 🧪")
    print("🧪  BTC FORECAST BOT - ЛОКАЛЬНОЕ ТЕСТИРОВАНИЕ  🧪")
    print("🧪 " + "="*58 + " 🧪")
    
    # Загружаем .env
    load_dotenv()
    
    # Проверяем наличие ключей
    required_keys = ['OPENAI_API_KEY', 'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHANNEL_ID']
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print(f"\n❌ Отсутствуют обязательные переменные: {', '.join(missing_keys)}")
        print("   Создайте файл .env на основе .env.example")
        return
    
    # Тест 1: Получение данных
    success, bot, df_1h, df_1d, df_1w = test_data_fetching()
    if not success:
        return
    
    # Тест 2: Технический анализ
    success, ta_weekly, ta_monthly, ta_yearly = test_technical_analysis(bot, df_1h, df_1d, df_1w)
    if not success:
        return
    
    # Тест 3: Макроданные
    success, sp500_data, macro_data, correlation = test_macro_data(bot, df_1d)
    if not success:
        return
    
    # Тест 4: AI прогноз
    success, forecast = test_ai_forecast(
        bot, ta_weekly, ta_monthly, ta_yearly,
        sp500_data, macro_data, correlation
    )
    if not success:
        return
    
    # Тест 5: Форматирование
    success, message = test_message_formatting(bot, ta_weekly, forecast)
    if not success:
        return
    
    # Финальный отчет
    print("\n" + "✅ " + "="*56 + " ✅")
    print("✅  ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!  ✅")
    print("✅ " + "="*56 + " ✅")
    print("\n💡 Чтобы опубликовать в Telegram, запустите: python main.py")
    print("💡 Для автоматизации настройте GitHub Actions по README.md\n")


if __name__ == "__main__":
    main()
