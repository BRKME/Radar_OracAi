"""
BTC Forecast Bot - MVP Version
Автоматический технический анализ Bitcoin с AI прогнозом на неделю, месяц и год
"""

import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import logging
# import tempfile  # Not needed - TradingView scraping disabled for compliance

import ccxt
import yfinance as yf
import pandas as pd
import ta
import numpy as np
from openai import OpenAI
from fredapi import Fred
import requests
from dotenv import load_dotenv
# from playwright.sync_api import sync_playwright  # Not needed - TradingView scraping disabled

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
        self.exchange = ccxt.kraken()
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
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
            
            ohlcv = self.exchange.fetch_ohlcv('BTC/USD', timeframe, limit=limit)
            
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
            rsi_indicator = ta.momentum.RSIIndicator(close=df['close'], window=14)
            df['rsi'] = rsi_indicator.rsi()
            rsi = df['rsi'].iloc[-1]
            if pd.isna(rsi):
                logger.warning("RSI is NaN, using neutral value 50")
                rsi = 50.0
            
            # MACD
            macd_indicator = ta.trend.MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
            df['macd'] = macd_indicator.macd()
            df['macd_signal'] = macd_indicator.macd_signal()
            df['macd_hist'] = macd_indicator.macd_diff()
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            macd_hist = df['macd_hist'].iloc[-1]
            
            # Заменяем NaN на 0 для MACD
            if pd.isna(macd):
                macd = 0.0
            if pd.isna(macd_signal):
                macd_signal = 0.0
            if pd.isna(macd_hist):
                macd_hist = 0.0
            
            # EMA
            ema20_indicator = ta.trend.EMAIndicator(close=df['close'], window=20)
            ema50_indicator = ta.trend.EMAIndicator(close=df['close'], window=50)
            ema200_indicator = ta.trend.EMAIndicator(close=df['close'], window=200)
            df['ema20'] = ema20_indicator.ema_indicator()
            df['ema50'] = ema50_indicator.ema_indicator()
            df['ema200'] = ema200_indicator.ema_indicator()
            ema20 = df['ema20'].iloc[-1]
            ema50 = df['ema50'].iloc[-1]
            ema200 = df['ema200'].iloc[-1]
            
            # Fallback для EMA если NaN
            current_price = df['close'].iloc[-1]
            if pd.isna(ema20):
                logger.warning("EMA20 is NaN, using current price")
                ema20 = current_price
            if pd.isna(ema50):
                logger.warning("EMA50 is NaN, using EMA20 or current price")
                ema50 = ema20 if not pd.isna(ema20) else current_price
            if pd.isna(ema200):
                logger.warning("EMA200 is NaN, using EMA50 or current price")
                ema200 = ema50 if not pd.isna(ema50) else current_price
            
            # Bollinger Bands
            bb_indicator = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb_indicator.bollinger_hband()
            df['bb_middle'] = bb_indicator.bollinger_mavg()
            df['bb_lower'] = bb_indicator.bollinger_lband()
            bb_upper = df['bb_upper'].iloc[-1]
            bb_middle = df['bb_middle'].iloc[-1]
            bb_lower = df['bb_lower'].iloc[-1]
            
            # Fallback для BB если NaN
            if pd.isna(bb_upper) or pd.isna(bb_middle) or pd.isna(bb_lower):
                logger.warning("Bollinger Bands contain NaN, using price-based defaults")
                bb_middle = current_price
                bb_upper = current_price * 1.02  # +2%
                bb_lower = current_price * 0.98  # -2%
            
            # Позиция цены в BB (с защитой от деления на ноль)
            if bb_upper != bb_lower:
                bb_position = ((current_price - bb_lower) / (bb_upper - bb_lower)) * 100
            else:
                bb_position = 50.0  # Нейтральная позиция при отсутствии волатильности
                logger.warning("BB bands are equal, using neutral position")
            
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
        ema20 = df['ema20'].iloc[-1]
        ema50 = df['ema50'].iloc[-1]
        ema200 = df['ema200'].iloc[-1]
        
        if current_price > ema20 > ema50 > ema200:
            return "strong uptrend"
        elif current_price > ema20 > ema50:
            return "uptrend"
        elif current_price < ema20 < ema50 < ema200:
            return "strong downtrend"
        elif current_price < ema20 < ema50:
            return "downtrend"
        else:
            return "range-bound"
    
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
            
            # Проверка на пустой DataFrame
            if sp500.empty or len(sp500) == 0:
                logger.warning("S&P 500 data is empty, skipping")
                return {
                    'current_price': None,
                    'change_1m': None
                }
            
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
            
            # Проверка на пустой DataFrame
            if sp500.empty or len(sp500) < 20:
                logger.warning(f"S&P 500 data insufficient for correlation: {len(sp500) if not sp500.empty else 0} rows")
                return None
            
            # Приводим к дневным данным для корреляции
            btc_daily = btc_df['close'].resample('D').last()
            sp500_daily = sp500['Close']
            
            # Объединяем и считаем корреляцию
            combined = pd.DataFrame({
                'btc': btc_daily,
                'sp500': sp500_daily
            }).dropna()
            
            # Строгая проверка минимума данных
            if len(combined) < 20:
                logger.warning(f"Not enough overlapping data for correlation: {len(combined)} rows (minimum 20 required)")
                return None
            
            correlation = combined['btc'].corr(combined['sp500'])
            
            return correlation
            
        except Exception as e:
            logger.error(f"Ошибка расчета корреляции: {e}")
            return None
    
    def calculate_eth_sp500_correlation(self, eth_df: pd.DataFrame) -> Optional[float]:
        """
        Расчет корреляции ETH и S&P 500
        
        Args:
            eth_df: DataFrame с данными ETH
        
        Returns:
            Коэффициент корреляции или None
        """
        try:
            # Получаем S&P 500 за тот же период
            start_date = eth_df.index[0].strftime('%Y-%m-%d')
            end_date = eth_df.index[-1].strftime('%Y-%m-%d')
            
            sp500 = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
            
            # Проверка на пустой DataFrame
            if sp500.empty or len(sp500) < 20:
                logger.warning(f"S&P 500 data insufficient for ETH correlation: {len(sp500) if not sp500.empty else 0} rows")
                return None
            
            # Приводим к дневным данным для корреляции
            eth_daily = eth_df['close'].resample('D').last()
            sp500_daily = sp500['Close']
            
            # Объединяем и считаем корреляцию
            combined = pd.DataFrame({
                'eth': eth_daily,
                'sp500': sp500_daily
            }).dropna()
            
            # Строгая проверка минимума данных
            if len(combined) < 20:
                logger.warning(f"Not enough overlapping data for ETH correlation: {len(combined)} rows (minimum 20 required)")
                return None
            
            correlation = combined['eth'].corr(combined['sp500'])
            
            return correlation
            
        except Exception as e:
            logger.error(f"Ошибка расчета ETH корреляции: {e}")
            return None
    
    def fetch_eth_data(self, timeframe: str, limit: int) -> pd.DataFrame:
        """
        Получение исторических данных ETH
        
        Args:
            timeframe: Таймфрейм (1h, 4h, 1d, 1w)
            limit: Количество свечей
        
        Returns:
            DataFrame с OHLCV данными
        """
        try:
            logger.info(f"Получение данных ETH/{timeframe} (limit={limit})")
            
            ohlcv = self.exchange.fetch_ohlcv('ETH/USD', timeframe, limit=limit)
            
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Получено {len(df)} свечей ETH для {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Ошибка получения данных ETH: {e}")
            raise
    
    def validate_market_data(self, df: pd.DataFrame, symbol: str, min_hours: int) -> bool:
        """
        Валидация рыночных данных
        
        Args:
            df: DataFrame с данными
            symbol: Символ актива (для логирования)
            min_hours: Минимальное количество часов данных
        
        Returns:
            True если данные валидны, False если нет
        """
        if df.empty:
            logger.error(f"{symbol} dataset is empty")
            return False
        
        if len(df) < min_hours:
            logger.error(f"{symbol} insufficient data: {len(df)}h < {min_hours}h required")
            return False
        
        if df['close'].isna().any():
            logger.error(f"{symbol} contains NaN values in close prices")
            return False
        
        if (df['close'] <= 0).any():
            logger.error(f"{symbol} contains invalid prices (<=0)")
            return False
        
        return True
    
    def check_eth_post_conditions(
        self, 
        eth_df_1h: pd.DataFrame,
        btc_df_1h: pd.DataFrame
    ) -> Tuple[bool, str]:
        """
        Проверка условий для отдельного поста по ETH
        
        Условия:
        1. ETH ведёт рынок (ETH/BTC растёт, ETH волатильнее)
        2. Есть собственный поток капитала
        3. Структурное событие
        4. Нарративный сдвиг
        
        Args:
            eth_df_1h: DataFrame с данными ETH 1h
            btc_df_1h: DataFrame с данными BTC 1h
        
        Returns:
            (should_post, reason)
        """
        try:
            logger.info("Проверка условий для ETH поста")
            
            # Валидация входных данных
            MIN_HOURS_7D = 168  # 7 days
            MIN_HOURS_24H = 24  # 1 day
            
            if not self.validate_market_data(eth_df_1h, "ETH", MIN_HOURS_7D):
                return (False, "Invalid ETH data - insufficient or corrupt")
            
            if not self.validate_market_data(btc_df_1h, "BTC", MIN_HOURS_7D):
                return (False, "Invalid BTC data - insufficient or corrupt")
            
            # Условие 1: ETH ведёт рынок
            # Проверяем ETH/BTC ratio за последние 7 дней
            # Используем среднее 3 последних свечей для стабильности
            eth_price_now = eth_df_1h['close'].iloc[-3:].mean()
            btc_price_now = btc_df_1h['close'].iloc[-3:].mean()
            eth_price_7d = eth_df_1h['close'].iloc[-MIN_HOURS_7D:-MIN_HOURS_7D+3].mean()
            btc_price_7d = btc_df_1h['close'].iloc[-MIN_HOURS_7D:-MIN_HOURS_7D+3].mean()
            
            eth_btc_ratio_now = eth_price_now / btc_price_now
            eth_btc_ratio_7d_ago = eth_price_7d / btc_price_7d
            
            eth_btc_change = ((eth_btc_ratio_now / eth_btc_ratio_7d_ago) - 1) * 100
            
            if eth_btc_change > 2.0:  # ETH/BTC вырос >2% за неделю
                reason = f"ETH/BTC ratio up {eth_btc_change:.1f}% (7d) - ETH leading market"
                logger.info(f"ETH post condition met: {reason}")
                return (True, reason)
            
            # Проверяем волатильность за последние 24 часа
            eth_volatility_24h = eth_df_1h['close'].iloc[-MIN_HOURS_24H:].pct_change().std() * 100
            btc_volatility_24h = btc_df_1h['close'].iloc[-MIN_HOURS_24H:].pct_change().std() * 100
            
            # BUG FIX: Check for NaN volatility before comparison
            if pd.notna(eth_volatility_24h) and pd.notna(btc_volatility_24h):
                if eth_volatility_24h > btc_volatility_24h * 1.5:  # ETH волатильнее на 50%+
                    reason = f"ETH volatility {eth_volatility_24h:.2f}% vs BTC {btc_volatility_24h:.2f}% (24h) - independent movement"
                    logger.info(f"ETH post condition met: {reason}")
                    return (True, reason)
            else:
                logger.warning(f"Volatility calculation returned NaN - skipping volatility check")
            
            # Проверяем divergence: BTC flat, ETH движется
            btc_change_24h = ((btc_df_1h['close'].iloc[-1] / btc_df_1h['close'].iloc[-MIN_HOURS_24H]) - 1) * 100
            eth_change_24h = ((eth_df_1h['close'].iloc[-1] / eth_df_1h['close'].iloc[-MIN_HOURS_24H]) - 1) * 100
            
            if abs(btc_change_24h) < 1.0 and abs(eth_change_24h) > 3.0:  # BTC <1%, ETH >3%
                reason = f"Divergence: BTC {btc_change_24h:+.1f}%, ETH {eth_change_24h:+.1f}% (24h)"
                logger.info(f"ETH post condition met: {reason}")
                return (True, reason)
            
            logger.info("ETH post conditions not met - skipping separate ETH forecast")
            return (False, "No significant ETH-specific catalyst")
            
        except Exception as e:
            logger.error(f"Ошибка проверки ETH условий: {e}")
            return (False, f"Error checking conditions: {e}")
    
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
            system_prompt = """ROLE:
You are a senior buy-side strategist. Your output goes directly into institutional morning brief without editing.

INSTITUTIONAL ALIGNMENT:
When forming price expectations, consider public forecasts from major institutional players:
* Asset managers: BlackRock, Fidelity, VanEck, Grayscale
* Investment banks: JPMorgan, Goldman Sachs, Morgan Stanley, Citi
* Crypto-native funds: ARK Invest, Galaxy Digital, Pantera Capital
* Research firms: Messari, Glassnode, CoinMetrics

If your view significantly diverges from institutional consensus, acknowledge the positioning difference without deferring to it.

LANGUAGE RULES (STRICT):
* Maximum 1 sentence per horizon
* NO explanatory words: ❌ "due to", "reflecting", "caused by", "amid", "determines", "tests"
* NO technical indicator names (RSI, MACD, EMA, Bollinger, etc.)
* NO evaluative adjectives ("strong", "weak", "significant")
* NO emotional or optimistic language
* NO bullish/bearish balance rhetoric
* NO narrative language ("tests narrative", "challenges story")
* ❌ FORBIDDEN PHRASES: "Price trades", "Price within", "Corridor", "Price range"

ALLOWED vocabulary (market state only):
* "capped above / supported below"
* "requires break above/below"
* "defines upper/lower boundary"
* "assumes continuation"
* "indicates sustained"
* "anchors", "implies", "challenges assumptions", "shifts expectations"

RANGES & PROBABILITIES:
* Only ranges, never point targets
* Probabilities implicit only
* No categorical statements
* Only dominant scenario described

STRUCTURE:

Short-term view:
Start with dollar range IMMEDIATELY. Format: "$86,271–$88,548, [conditions]"
❌ WRONG: "Price trades $86,271–$88,548"
✅ CORRECT: "$86,271–$88,548, break above requires sustained bid"

Medium-term view:
Use "Range" ONLY for medium-term. Format: "Range $X–$Y [conditions]"
✅ CORRECT: "Range $83,870–$94,584 holds, upside capped by overhead supply"

Long-term view:
Start with dollar range IMMEDIATELY. Format: "$X–$Y [conditions]"
❌ WRONG: "Corridor $70,000–$102,000"
✅ CORRECT: "$70,000–$102,000 sensitive to policy trajectory"

Risk framing (2-3 sentences max):
Identify specific macro catalysts with dates and implications. Only facts and market implications, no narrative language.

Key events to reference:
* Federal Reserve: Next FOMC meeting date, rate expectations, QE/QT status
* Inflation: Next CPI/PCE release date, threshold levels
* Employment: Next NFP date, unemployment threshold
* Banking/Credit: Specific stress metrics, liquidity levels
* Geopolitical: Dated events with market impact
* Regulation: Specific SEC actions, legislation dates

Format (cold, factual):
FOMC (Dec 18) anchors near-term rates; pricing implies 25bp hold.
CPI (Jan 15) above 3.5% challenges current rate path assumptions.
Deposit outflows exceeding $50B weekly trigger liquidity concerns.

Institutional reference:
Pick ONE random institution from this list and cite their latest public BTC price target/view (if known):
* BlackRock
* JPMorgan
* Goldman Sachs
* Morgan Stanley
* ARK Invest
* Fidelity
* VanEck
* Grayscale
* Galaxy Digital
* Standard Chartered
* Bernstein

Format: "[Institution] targets $X-$Y by [timeframe]" OR "[Institution] expects [brief view]"
If you don't know their latest forecast, skip this section entirely.

QUALITY FILTER:
Before output, verify:
1. Can this go into fund morning brief without edits?
2. Does every sentence work on price?
3. Can you remove 10% more words without losing meaning?
4. Is the tone cold and factual, not narrative?
5. ❌ Did you use "Price trades", "Corridor", or "Price range"? If YES → FIX IMMEDIATELY

If YES to 1-4 and NO to 5 → publish
If NO → compress further

EXAMPLE (PERFECT):
Short-term view:
$86,271–$88,548, break above requires sustained bid.

Medium-term view:
Range $83,870–$94,584 holds, upside capped by overhead supply, breakout needs macro shift.

Long-term view:
$70,000–$102,000 sensitive to policy trajectory.

Risk framing:
FOMC (Dec 18) anchors near-term rates; pricing implies 25bp hold.
CPI (Jan 15) above 3.5% challenges current rate path assumptions.

Institutional reference:
Goldman Sachs targets $150K-$200K by end 2025

MAIN RULE:
Less is more. Every word must earn its place. Cold facts over narrative."""

            user_prompt = f"""Analyze the following data and provide institutional price forecast:

{context}

Provide forecast in the specified format."""

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
        
        price = ta_weekly['current_price']
        
        # Простое описание позиции цены без терминологии индикаторов
        rsi_1h = ta_weekly['rsi']
        momentum_status = "stretched upside" if rsi_1h > 70 else "oversold" if rsi_1h < 30 else "neutral"
        
        macd_trend = "positive" if ta_weekly['macd_hist'] > 0 else "negative"
        
        price_vs_ma = "above key moving average" if price > ta_weekly['ema20'] else "below key moving average"
        
        bb_pos = ta_weekly['bb_position']
        range_position = "upper boundary" if bb_pos > 80 else "lower boundary" if bb_pos < 20 else "mid-range"
        
        context = f"""MARKET DATA - BTC/USD

Current Price: ${price:,.2f}

Price Structure:
- Momentum: {momentum_status} ({rsi_1h:.1f})
- Short-term trend: {macd_trend}
- Position: {price_vs_ma}
- Range position: {range_position} ({bb_pos:.1f}%)
- Immediate support: ${ta_weekly['support']:,.2f}
- Immediate resistance: ${ta_weekly['resistance']:,.2f}
- Local trend: {ta_weekly['trend']}

Medium-term levels:
- Support zone: ${ta_monthly['support']:,.2f}
- Resistance zone: ${ta_monthly['resistance']:,.2f}
- Trend character: {ta_monthly['trend']}

Long-term context:
- Major trend: {ta_yearly['trend']}
- Annual support: ${ta_yearly['support']:,.2f}
- Annual resistance: ${ta_yearly['resistance']:,.2f}

MACRO BACKDROP:"""
        
        if sp500_data['current_price']:
            context += f"""
- S&P 500: ${sp500_data['current_price']:,.2f} ({sp500_data['change_1m']:+.1f}% monthly)
- Risk appetite: {"positive" if sp500_data['change_1m'] > 0 else "negative"}"""
        
        if correlation is not None:
            corr_strength = "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.4 else "weak"
            corr_direction = "positive" if correlation > 0 else "negative"
            context += f"""
- BTC-equity correlation: {corr_strength} {corr_direction} ({correlation:.2f})"""
        
        if macro_data['fed_rate']:
            context += f"""
- Federal Funds Rate: {macro_data['fed_rate']:.2f}% (restrictive territory)"""
        else:
            context += f"""
- Monetary policy: data unavailable"""
        
        return context
    
    def generate_eth_forecast(
        self,
        ta_weekly: Dict,
        ta_monthly: Dict,
        ta_yearly: Dict,
        eth_btc_context: str,
        sp500_data: Dict,
        macro_data: Dict,
        correlation: Optional[float]
    ) -> str:
        """
        Генерация AI прогноза для ETH
        
        Args:
            ta_weekly: Технический анализ для недельного прогноза
            ta_monthly: Технический анализ для месячного прогноза
            ta_yearly: Технический анализ для годового прогноза
            eth_btc_context: Контекст ETH/BTC positioning
            sp500_data: Данные S&P 500
            macro_data: Макроэкономические данные
            correlation: Корреляция ETH-SPX
        
        Returns:
            Текст AI прогноза для ETH
        """
        try:
            logger.info("Генерация AI прогноза для ETH")
            
            price = ta_weekly['current_price']
            
            context = f"""MARKET DATA - ETH/USD

Current Price: ${price:,.2f}
Support: ${ta_weekly['support']:,.2f} | Resistance: ${ta_weekly['resistance']:,.2f}
Trend: {ta_weekly['trend']}

Medium-term levels:
- Support zone: ${ta_monthly['support']:,.2f}
- Resistance zone: ${ta_monthly['resistance']:,.2f}
- Trend character: {ta_monthly['trend']}

Long-term context:
- Major trend: {ta_yearly['trend']}
- Annual support: ${ta_yearly['support']:,.2f}
- Annual resistance: ${ta_yearly['resistance']:,.2f}

ETH-SPECIFIC CONTEXT:
{eth_btc_context}

MACRO BACKDROP:"""
            
            if sp500_data['current_price'] is not None:
                context += f"""
- S&P 500: ${sp500_data['current_price']:,.2f} ({sp500_data['change_1m']:+.1f}% monthly)
- Risk appetite: {"positive" if sp500_data['change_1m'] > 0 else "negative"}"""
            
            if correlation is not None:
                corr_strength = "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.4 else "weak"
                corr_direction = "positive" if correlation > 0 else "negative"
                context += f"""
- ETH-equity correlation: {corr_strength} {corr_direction} ({correlation:.2f})"""
            
            if macro_data.get('fed_rate'):
                context += f"""
- Federal Funds Rate: {macro_data['fed_rate']:.2f}%"""
            else:
                context += f"""
- Monetary policy: data unavailable"""

            # Используем почти тот же system_prompt что и для BTC
            system_prompt = """ROLE:
You are a senior buy-side strategist focusing on ETHEREUM. Output goes directly into institutional brief.

LANGUAGE RULES (STRICT):
* Maximum 1 sentence per horizon
* NO explanatory words: ❌ "due to", "reflecting", "caused by", "determines", "tests"
* NO technical indicator names
* NO evaluative adjectives
* NO emotional language
* NO narrative language
* ❌ FORBIDDEN PHRASES: "Price trades", "Price within", "Corridor", "Price range"

ALLOWED vocabulary:
* "capped above/supported below"
* "requires break above/below"
* "anchors", "implies", "challenges assumptions"

ETH-SPECIFIC FACTORS:
* ETH/BTC positioning and ratio dynamics
* ETF flows and institutional allocation
* Staking yield and supply dynamics
* Layer 2 ecosystem value capture
* Regulatory status (commodity vs security)

STRUCTURE:

Short-term view:
Start with dollar range IMMEDIATELY. Format: "$X–$Y, [conditions]"
❌ WRONG: "Price trades $3,180–$3,310"
✅ CORRECT: "$3,180–$3,310, ETH/BTC ratio positioning improves"

Medium-term view:
Use "Range" ONLY for medium-term. Format: "Range $X–$Y [conditions]"
✅ CORRECT: "Range $2,950–$3,580 assumes continued institutional flows"

Long-term view:
Start with dollar range IMMEDIATELY. Format: "$X–$Y [conditions]"
❌ WRONG: "Corridor $2,400–$4,200"
✅ CORRECT: "$2,400–$4,200 sensitive to staking yield dynamics"

Risk framing (2 sentences max):
ETH-specific risks with facts and dates. Cold, factual tone only.

Institutional reference:
Pick ONE random institution from this list and cite their latest public ETH price target (if known):
* VanEck
* ARK Invest
* Grayscale
* BlackRock
* Fidelity
* JPMorgan
* Goldman Sachs
* Standard Chartered
* Bernstein

If unknown, skip this section entirely.

QUALITY FILTER:
1. Fund morning brief quality?
2. Every sentence on price?
3. Can remove 10% more words?
4. Cold and factual?
5. ❌ Did you use "Price trades", "Corridor", or "Price range"? If YES → FIX IMMEDIATELY

MAIN RULE:
Less is more. Cold facts over narrative."""

            user_prompt = f"""Analyze ETHEREUM data and provide institutional forecast:

{context}

Provide forecast in specified format."""

            # Запрос к GPT
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            forecast = response.choices[0].message.content.strip()
            
            logger.info("ETH прогноз успешно сгенерирован")
            return forecast
            
        except Exception as e:
            logger.error(f"Ошибка генерации ETH прогноза: {e}")
            raise
    
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
        
        message = f"""<b>Bitcoin Price Forecast</b>

<b>Current:</b> ${current_price:,.0f}
<b>Support:</b> ${ta_weekly['support']:,.0f} | <b>Resistance:</b> ${ta_weekly['resistance']:,.0f}

{forecast}

<i>OracAI-assisted analysis | Not financial advice | {datetime.now().strftime('%d %b %Y %H:%M UTC')}</i>
"""
        
        return message
    
    def generate_tradingview_chart(self) -> Optional[str]:
        """
        Генерация скриншота графика TradingView
        
        Returns:
            Путь к файлу скриншота или None при ошибке
        """
        try:
            logger.info("Генерация скриншота TradingView")
            
            with sync_playwright() as p:
                # Запуск браузера в headless режиме с user-agent
                browser = p.chromium.launch(
                    headless=True,
                    args=['--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36']
                )
                page = browser.new_page(viewport={'width': 1200, 'height': 800})
                
                # Открываем TradingView график BTC/USD
                url = "https://www.tradingview.com/chart/?symbol=BITSTAMP%3ABTCUSD&interval=D"
                page.goto(url, wait_until='domcontentloaded', timeout=15000)
                
                # Ждем загрузки графика (ищем canvas элемент)
                try:
                    page.wait_for_selector('canvas', timeout=10000)
                    page.wait_for_timeout(2000)  # Дополнительная пауза для рендеринга
                    logger.info("График TradingView загружен")
                except Exception as wait_error:
                    logger.warning(f"Canvas не найден, делаем скриншот anyway: {wait_error}")
                    page.wait_for_timeout(3000)  # Fallback timeout
                
                # Создаем временный файл для скриншота
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                screenshot_path = temp_file.name
                temp_file.close()
                
                # Делаем скриншот
                page.screenshot(path=screenshot_path, full_page=False)
                
                browser.close()
                
                logger.info(f"Скриншот сохранен: {screenshot_path}")
                return screenshot_path
                
        except Exception as e:
            logger.error(f"Ошибка генерации скриншота TradingView: {e}")
            return None
            return None
    
    def publish_to_telegram(self, message: str, chart_path: Optional[str] = None):
        """
        Публикация сообщения в Telegram канал
        
        Args:
            message: Текст сообщения
            chart_path: Путь к файлу графика (опционально)
        """
        try:
            logger.info(f"Публикация в Telegram канал: {self.channel_id}")
            
            # Telegram имеет лимит 4096 символов
            if len(message) > 4096:
                logger.warning(f"Message too long ({len(message)} chars), truncating to 4090")
                message = message[:4090] + "\n..."
            
            # Отправка графика если есть
            if chart_path:
                try:
                    url = f"https://api.telegram.org/bot{self.telegram_token}/sendPhoto"
                    
                    with open(chart_path, 'rb') as photo:
                        files = {'photo': photo}
                        payload = {
                            'chat_id': self.channel_id,
                            'caption': 'BTC/USD Chart'
                        }
                        
                        response = requests.post(url, data=payload, files=files, timeout=30)
                        response.raise_for_status()
                        
                        logger.info("График успешно отправлен")
                    
                    # Удаляем временный файл
                    if os.path.exists(chart_path):
                        try:
                            os.unlink(chart_path)
                            logger.info(f"Временный файл удален: {chart_path}")
                        except Exception as cleanup_error:
                            logger.warning(f"Не удалось удалить временный файл: {cleanup_error}")
                    
                except Exception as e:
                    logger.error(f"Ошибка отправки графика: {e}")
                    # Продолжаем отправку текста даже если график не отправился
            
            # Отправка текстового прогноза
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                'chat_id': self.channel_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            if result.get('ok'):
                logger.info("Сообщение успешно опубликовано")
            else:
                logger.error(f"Telegram API error: {result}")
                raise Exception(f"Telegram API returned ok=false: {result}")
            
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
            
            # 8. Генерация графика ОТКЛЮЧЕНА для compliance
            # TradingView ToS запрещает автоматический scraping
            # TODO: Implement local chart generation with matplotlib if needed
            logger.info("Шаг 8: Генерация графика пропущена (compliance)")
            chart_path = None
            
            # 9. Публикуем в Telegram
            logger.info("Шаг 9: Публикация BTC прогноза в Telegram")
            self.publish_to_telegram(message, chart_path)
            
            # 10. Проверяем условия для ETH поста
            logger.info("Шаг 10: Проверка условий для отдельного ETH прогноза")
            try:
                # Получаем данные ETH
                eth_df_1h = self.fetch_eth_data('1h', limit=168)
                
                # Проверяем условия
                should_post_eth, eth_reason = self.check_eth_post_conditions(eth_df_1h, df_1h)
                
                if should_post_eth:
                    logger.info(f"ETH условия выполнены: {eth_reason}")
                    
                    # 11. Получаем данные ETH для разных таймфреймов
                    logger.info("Шаг 11: Получение данных ETH для анализа")
                    eth_df_1d = self.fetch_eth_data('1d', limit=90)
                    eth_df_1w = self.fetch_eth_data('1w', limit=52)
                    
                    # 12. Рассчитываем технические индикаторы для ETH
                    logger.info("Шаг 12: Расчет технических индикаторов ETH")
                    eth_ta_weekly = self.calculate_technical_indicators(eth_df_1h)
                    eth_ta_monthly = self.calculate_technical_indicators(eth_df_1d)
                    eth_ta_yearly = self.calculate_technical_indicators(eth_df_1w)
                    
                    # 12a. Рассчитываем корреляцию ETH-SPX
                    logger.info("Шаг 12a: Расчет корреляции ETH-SPX")
                    eth_correlation = self.calculate_eth_sp500_correlation(eth_df_1d)
                    
                    # 13. Генерируем ETH прогноз
                    logger.info("Шаг 13: Генерация AI прогноза для ETH")
                    eth_forecast = self.generate_eth_forecast(
                        eth_ta_weekly, 
                        eth_ta_monthly, 
                        eth_ta_yearly,
                        eth_reason,  # Используем reason как контекст
                        sp500_data,  # Добавляем S&P 500 данные
                        macro_data,
                        eth_correlation  # Добавляем корреляцию ETH-SPX
                    )
                    
                    # 14. Форматируем и публикуем ETH прогноз
                    logger.info("Шаг 14: Публикация ETH прогноза в Telegram")
                    
                    # Реальная задержка 1 час между BTC и ETH постами
                    logger.info("⏳ Ожидание 1 часа перед публикацией ETH...")
                    time.sleep(3600)  # 1 час = 3600 секунд
                    
                    # Время публикации ETH (фактическое текущее время)
                    eth_time = datetime.now()
                    
                    eth_message = f"""<b>Ethereum Price Forecast</b>

<b>Current:</b> ${eth_ta_weekly['current_price']:,.0f}
<b>Support:</b> ${eth_ta_weekly['support']:,.0f} | <b>Resistance:</b> ${eth_ta_weekly['resistance']:,.0f}

{eth_forecast}

<i>OracAI-assisted analysis | Not financial advice | {eth_time.strftime('%d %b %Y %H:%M UTC')}</i>
"""
                    self.publish_to_telegram(eth_message, None)
                    logger.info("✅ ETH прогноз успешно опубликован")
                else:
                    logger.info(f"ETH условия не выполнены: {eth_reason}")
                    
            except Exception as e:
                logger.warning(f"Ошибка при обработке ETH: {e}. Продолжаем без ETH прогноза.")
            
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
