"""
BTC Forecast Bot - MVP Version
Автоматический технический анализ Bitcoin с AI прогнозом на неделю, месяц и год
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import logging

import ccxt
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from openai import OpenAI
from fredapi import Fred
from telegram import Bot
from telegram.constants import ParseMode
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BTCForecastBot:
    """Бот для технического анализа BTC и AI прогнозирования"""
    
    def __init__(self):
        """Инициализация бота с загрузкой конфигурации"""
        load_dotenv()
        
        # Telegram
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.channel_id = os.getenv('TELEGRAM_CHANNEL_ID')
        
        # OpenAI
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        # FRED API
        self.fred_api_key = os.getenv('FRED_API_KEY')
        
        # Проверка обязательных переменных
        self._validate_config()
        
        # Инициализация клиентов
        self.exchange = ccxt.binance()
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.telegram_bot = Bot(token=self.telegram_token)
        
        if self.fred_api_key:
            self.fred = Fred(api_key=self.fred_api_key)
        else:
            self.fred = None
            logger.warning("FRED API key не указан, макроэкономические данные будут ограничены")
    
    def _validate_config(self):
        """Проверка наличия обязательных переменных окружения"""
        required_vars = {
            'TELEGRAM_BOT_TOKEN': self.telegram_token,
            'TELEGRAM_CHANNEL_ID': self.channel_id,
            'OPENAI_API_KEY': self.openai_api_key
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        
        if missing_vars:
            logger.error(f"Отсутствуют обязательные переменные: {', '.join(missing_vars)}")
            sys.exit(1)
    
    def fetch_btc_data(self, timeframe: str, limit: int) -> pd.DataFrame:
        """
        Получение исторических данных BTC
        
        Args:
            timeframe: Таймфрейм (1h, 4h, 1d, 1w)
            limit: Количество свечей
        
        Returns:
            DataFrame с OHLCV данными
        """
        try:
            logger.info(f"Получение данных BTC/{timeframe} (limit={limit})")
            
            ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', timeframe, limit=limit)
            
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Получено {len(df)} свечей для {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Ошибка получения данных BTC: {e}")
            raise
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """
        Расчет технических индикаторов
        
        Args:
            df: DataFrame с ценовыми данными
        
        Returns:
            Словарь с рассчитанными индикаторами
        """
        try:
            # RSI
            df.ta.rsi(length=14, append=True)
            rsi = df['RSI_14'].iloc[-1]
            
            # MACD
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
            macd = df['MACD_12_26_9'].iloc[-1]
            macd_signal = df['MACDs_12_26_9'].iloc[-1]
            macd_hist = df['MACDh_12_26_9'].iloc[-1]
            
            # EMA
            df.ta.ema(length=20, append=True)
            df.ta.ema(length=50, append=True)
            df.ta.ema(length=200, append=True)
            ema20 = df['EMA_20'].iloc[-1]
            ema50 = df['EMA_50'].iloc[-1]
            ema200 = df['EMA_200'].iloc[-1]
            
            # Bollinger Bands
            df.ta.bbands(length=20, std=2, append=True)
            bb_upper = df['BBU_20_2.0'].iloc[-1]
            bb_middle = df['BBM_20_2.0'].iloc[-1]
            bb_lower = df['BBL_20_2.0'].iloc[-1]
            
            # Текущая цена
            current_price = df['close'].iloc[-1]
            
            # Позиция цены в BB
            bb_position = ((current_price - bb_lower) / (bb_upper - bb_lower)) * 100
            
            # Анализ тренда
            trend = self._analyze_trend(df)
            
            # Поддержка и сопротивление
            support, resistance = self._calculate_support_resistance(df)
            
            return {
                'current_price': current_price,
                'rsi': rsi,
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_hist': macd_hist,
                'ema20': ema20,
                'ema50': ema50,
                'ema200': ema200,
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
                'bb_position': bb_position,
                'trend': trend,
                'support': support,
                'resistance': resistance,
                'volume_avg': df['volume'].tail(20).mean(),
                'volume_current': df['volume'].iloc[-1]
            }
            
        except Exception as e:
            logger.error(f"Ошибка расчета индикаторов: {e}")
            raise
    
    def _analyze_trend(self, df: pd.DataFrame) -> str:
        """Определение тренда на основе EMA"""
        current_price = df['close'].iloc[-1]
        ema20 = df['EMA_20'].iloc[-1]
        ema50 = df['EMA_50'].iloc[-1]
        ema200 = df['EMA_200'].iloc[-1]
        
        if current_price > ema20 > ema50 > ema200:
            return "сильный восходящий"
        elif current_price > ema20 > ema50:
            return "восходящий"
        elif current_price < ema20 < ema50 < ema200:
            return "сильный нисходящий"
        elif current_price < ema20 < ema50:
            return "нисходящий"
        else:
            return "боковой"
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Расчет уровней поддержки и сопротивления"""
        # Используем последние 30 свечей для определения уровней
        recent_data = df.tail(30)
        
        # Поддержка - минимум последних свечей
        support = recent_data['low'].min()
        
        # Сопротивление - максимум последних свечей
        resistance = recent_data['high'].max()
        
        return support, resistance
    
    def fetch_sp500_data(self) -> Dict:
        """
        Получение данных S&P 500
        
        Returns:
            Словарь с данными S&P 500
        """
        try:
            logger.info("Получение данных S&P 500")
            
            sp500 = yf.download('^GSPC', period='3mo', progress=False)
            
            current_price = sp500['Close'].iloc[-1]
            price_1m_ago = sp500['Close'].iloc[-30] if len(sp500) >= 30 else sp500['Close'].iloc[0]
            
            change_1m = ((current_price - price_1m_ago) / price_1m_ago) * 100
            
            return {
                'current_price': current_price,
                'change_1m': change_1m
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения данных S&P 500: {e}")
            return {
                'current_price': None,
                'change_1m': None
            }
    
    def fetch_macro_data(self) -> Dict:
        """
        Получение макроэкономических данных
        
        Returns:
            Словарь с макроэкономическими данными
        """
        macro_data = {
            'fed_rate': None,
            'qe_status': 'неизвестно'
        }
        
        if not self.fred:
            logger.warning("FRED API недоступен, пропуск макроданных")
            return macro_data
        
        try:
            logger.info("Получение данных ФРС")
            
            # Federal Funds Rate (эффективная ставка ФРС)
            fed_rate_series = self.fred.get_series('DFF')
            macro_data['fed_rate'] = fed_rate_series.iloc[-1]
            
            logger.info(f"Ставка ФРС: {macro_data['fed_rate']}%")
            
        except Exception as e:
            logger.error(f"Ошибка получения макроданных: {e}")
        
        return macro_data
    
    def calculate_btc_sp500_correlation(self, btc_df: pd.DataFrame) -> Optional[float]:
        """
        Расчет корреляции BTC и S&P 500
        
        Args:
            btc_df: DataFrame с данными BTC
        
        Returns:
            Коэффициент корреляции или None
        """
        try:
            # Получаем S&P 500 за тот же период
            start_date = btc_df.index[0].strftime('%Y-%m-%d')
            end_date = btc_df.index[-1].strftime('%Y-%m-%d')
            
            sp500 = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
            
            if len(sp500) < 10:
                return None
            
            # Приводим к дневным данным для корреляции
            btc_daily = btc_df['close'].resample('D').last()
            sp500_daily = sp500['Close']
            
            # Объединяем и считаем корреляцию
            combined = pd.DataFrame({
                'btc': btc_daily,
                'sp500': sp500_daily
            }).dropna()
            
            if len(combined) < 10:
                return None
            
            correlation = combined['btc'].corr(combined['sp500'])
            
            return correlation
            
        except Exception as e:
            logger.error(f"Ошибка расчета корреляции: {e}")
            return None
    
    def generate_ai_forecast(
        self,
        ta_weekly: Dict,
        ta_monthly: Dict,
        ta_yearly: Dict,
        sp500_data: Dict,
        macro_data: Dict,
        correlation: Optional[float]
    ) -> str:
        """
        Генерация AI прогноза на основе технического анализа и макроданных
        
        Args:
            ta_weekly: Технический анализ для недельного прогноза
            ta_monthly: Технический анализ для месячного прогноза
            ta_yearly: Технический анализ для годового прогноза
            sp500_data: Данные S&P 500
            macro_data: Макроэкономические данные
            correlation: Корреляция BTC-SPX
        
        Returns:
            Текст AI прогноза
        """
        try:
            logger.info("Генерация AI прогноза")
            
            # Формируем контекст для GPT
            context = self._build_context(
                ta_weekly, ta_monthly, ta_yearly,
                sp500_data, macro_data, correlation
            )
            
            # Промпт для GPT
            system_prompt = """Ты опытный криптоаналитик с глубоким пониманием технического анализа и макроэкономики.
Твоя задача - дать профессиональный прогноз по Bitcoin на разные временные горизонты.

Требования к прогнозу:
1. Используй вероятностный подход (не "будет 100k", а "вероятность 60% роста к диапазону X-Y")
2. Учитывай технические уровни поддержки/сопротивления
3. Учитывай макроэкономические факторы (ставка ФРС, корреляция с S&P 500)
4. Указывай ключевые триггеры, которые могут изменить сценарий
5. Будь объективным и реалистичным

Формат ответа:
📅 НЕДЕЛЬНЫЙ ПРОГНОЗ (1-7 дней):
[2-3 предложения с конкретными уровнями и вероятностями]

📊 МЕСЯЧНЫЙ ПРОГНОЗ (30 дней):
[2-3 предложения с диапазоном и факторами влияния]

📈 ГОДОВОЙ ПРОГНОЗ (365 дней):
[2-3 предложения с бычьим и медвежьим сценариями и их вероятностями]

⚠️ РИСКИ:
[1-2 ключевых риска, которые могут изменить прогноз]"""

            user_prompt = f"""Проанализируй следующие данные и дай прогноз:

{context}

Дай прогноз в указанном формате."""

            # Запрос к GPT
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            forecast = response.choices[0].message.content
            logger.info("AI прогноз успешно сгенерирован")
            
            return forecast
            
        except Exception as e:
            logger.error(f"Ошибка генерации AI прогноза: {e}")
            raise
    
    def _build_context(
        self,
        ta_weekly: Dict,
        ta_monthly: Dict,
        ta_yearly: Dict,
        sp500_data: Dict,
        macro_data: Dict,
        correlation: Optional[float]
    ) -> str:
        """Формирование контекста для GPT"""
        
        # RSI интерпретация
        rsi_1h = ta_weekly['rsi']
        rsi_status = "перекуплен" if rsi_1h > 70 else "перепродан" if rsi_1h < 30 else "нейтрален"
        
        # MACD сигнал
        macd_signal = "бычий кросс" if ta_weekly['macd_hist'] > 0 else "медвежий кросс"
        
        # Позиция в EMA
        price = ta_weekly['current_price']
        ema_position = "выше" if price > ta_weekly['ema20'] else "ниже"
        
        # BB позиция
        bb_pos = ta_weekly['bb_position']
        bb_zone = "верхней зоне" if bb_pos > 80 else "нижней зоне" if bb_pos < 20 else "средней зоне"
        
        context = f"""
📊 ТЕХНИЧЕСКИЙ АНАЛИЗ BTC/USDT

💰 Текущая цена: ${price:,.2f}

📈 НЕДЕЛЬНЫЙ ПРОГНОЗ (1h/4h таймфрейм):
- RSI(14): {rsi_1h:.1f} ({rsi_status})
- MACD: {macd_signal}
- EMA20: ${ta_weekly['ema20']:,.2f} (цена {ema_position})
- Bollinger Bands: цена в {bb_zone} (позиция {bb_pos:.1f}%)
- Поддержка: ${ta_weekly['support']:,.2f}
- Сопротивление: ${ta_weekly['resistance']:,.2f}
- Тренд: {ta_weekly['trend']}

📊 МЕСЯЧНЫЙ ПРОГНОЗ (1d таймфрейм):
- EMA50: ${ta_monthly['ema50']:,.2f}
- EMA200: ${ta_monthly['ema200']:,.2f}
- Тренд: {ta_monthly['trend']}
- Поддержка: ${ta_monthly['support']:,.2f}
- Сопротивление: ${ta_monthly['resistance']:,.2f}

📈 ГОДОВОЙ ПРОГНОЗ (1w таймфрейм):
- MA200: ${ta_yearly['ema200']:,.2f}
- Долгосрочный тренд: {ta_yearly['trend']}

🌐 МАКРОЭКОНОМИКА:"""
        
        if sp500_data['current_price']:
            context += f"""
- S&P 500: ${sp500_data['current_price']:,.2f} ({sp500_data['change_1m']:+.1f}% за месяц)"""
        
        if correlation is not None:
            corr_strength = "сильная" if abs(correlation) > 0.7 else "умеренная" if abs(correlation) > 0.4 else "слабая"
            context += f"""
- Корреляция BTC-SPX (30d): {correlation:.2f} ({corr_strength})"""
        
        if macro_data['fed_rate']:
            context += f"""
- Ставка ФРС: {macro_data['fed_rate']:.2f}%"""
        
        return context
    
    def format_telegram_message(self, ta_weekly: Dict, forecast: str) -> str:
        """
        Форматирование сообщения для Telegram
        
        Args:
            ta_weekly: Технический анализ для текущих данных
            forecast: AI прогноз
        
        Returns:
            Отформатированное сообщение
        """
        current_price = ta_weekly['current_price']
        
        # Эмодзи для тренда
        trend_emoji = {
            'сильный восходящий': '🚀',
            'восходящий': '📈',
            'боковой': '➡️',
            'нисходящий': '📉',
            'сильный нисходящий': '🔻'
        }
        
        emoji = trend_emoji.get(ta_weekly['trend'], '📊')
        
        message = f"""
{emoji} <b>BTC ПРОГНОЗ</b> {emoji}

💰 <b>Текущая цена:</b> ${current_price:,.2f}
📊 <b>Тренд:</b> {ta_weekly['trend']}
🎯 <b>Поддержка:</b> ${ta_weekly['support']:,.0f}
🎯 <b>Сопротивление:</b> ${ta_weekly['resistance']:,.0f}

{forecast}

<i>🤖 Автоматический анализ на основе TA + AI
⏰ Обновлено: {datetime.now().strftime('%d.%m.%Y %H:%M UTC')}</i>
"""
        
        return message
    
    def publish_to_telegram(self, message: str):
        """
        Публикация сообщения в Telegram канал
        
        Args:
            message: Текст сообщения
        """
        try:
            logger.info(f"Публикация в Telegram канал: {self.channel_id}")
            
            self.telegram_bot.send_message(
                chat_id=self.channel_id,
                text=message,
                parse_mode=ParseMode.HTML
            )
            
            logger.info("Сообщение успешно опубликовано")
            
        except Exception as e:
            logger.error(f"Ошибка публикации в Telegram: {e}")
            raise
    
    def run(self):
        """Основной метод запуска бота"""
        try:
            logger.info("=" * 50)
            logger.info("Запуск BTC Forecast Bot")
            logger.info("=" * 50)
            
            # 1. Получаем данные BTC для разных таймфреймов
            logger.info("Шаг 1: Получение данных BTC")
            df_1h = self.fetch_btc_data('1h', limit=168)  # Неделя
            df_1d = self.fetch_btc_data('1d', limit=90)   # ~3 месяца
            df_1w = self.fetch_btc_data('1w', limit=52)   # Год
            
            # 2. Рассчитываем технические индикаторы
            logger.info("Шаг 2: Расчет технических индикаторов")
            ta_weekly = self.calculate_technical_indicators(df_1h)
            ta_monthly = self.calculate_technical_indicators(df_1d)
            ta_yearly = self.calculate_technical_indicators(df_1w)
            
            # 3. Получаем данные S&P 500
            logger.info("Шаг 3: Получение данных S&P 500")
            sp500_data = self.fetch_sp500_data()
            
            # 4. Получаем макроэкономические данные
            logger.info("Шаг 4: Получение макроэкономических данных")
            macro_data = self.fetch_macro_data()
            
            # 5. Рассчитываем корреляцию BTC-SPX
            logger.info("Шаг 5: Расчет корреляции BTC-SPX")
            correlation = self.calculate_btc_sp500_correlation(df_1d)
            
            # 6. Генерируем AI прогноз
            logger.info("Шаг 6: Генерация AI прогноза")
            forecast = self.generate_ai_forecast(
                ta_weekly, ta_monthly, ta_yearly,
                sp500_data, macro_data, correlation
            )
            
            # 7. Форматируем сообщение
            logger.info("Шаг 7: Форматирование сообщения")
            message = self.format_telegram_message(ta_weekly, forecast)
            
            # 8. Публикуем в Telegram
            logger.info("Шаг 8: Публикация в Telegram")
            self.publish_to_telegram(message)
            
            logger.info("=" * 50)
            logger.info("BTC Forecast Bot завершил работу успешно!")
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"Критическая ошибка в работе бота: {e}", exc_info=True)
            sys.exit(1)


def main():
    """Точка входа"""
    bot = BTCForecastBot()
    bot.run()


if __name__ == "__main__":
    main()
