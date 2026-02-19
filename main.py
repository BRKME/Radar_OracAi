"""
OracAI Radar Bot - Unified Version
Адаптивная публикация по триггерам + AI анализ режима
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Optional, Tuple
import logging

import ccxt
import yfinance as yf
import pandas as pd
import ta
import numpy as np
from openai import OpenAI
import requests
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

STATE_FILE = 'state.json'


class OracAIRadar:
    """Unified Radar Bot с триггерной логикой и AI анализом"""
    
    def __init__(self):
        load_dotenv()
        
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.channel_id = os.getenv('TELEGRAM_CHANNEL_ID')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        self._validate_config()
        
        self.exchange = ccxt.kraken()
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.state = self._load_state()
    
    def _validate_config(self):
        required = {
            'TELEGRAM_BOT_TOKEN': self.telegram_token,
            'TELEGRAM_CHANNEL_ID': self.channel_id,
            'OPENAI_API_KEY': self.openai_api_key
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            logger.error(f"Missing: {', '.join(missing)}")
            sys.exit(1)
    
    def _load_state(self) -> Dict:
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"State load error: {e}")
        return {
            'last_regime': None,
            'last_publish': None,
            'last_round_level': None,
            'last_round_publish': None
        }
    
    def _save_state(self):
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    # ═══════════════════════════════════════════════════════════
    # DATA FETCHING
    # ═══════════════════════════════════════════════════════════
    
    def fetch_market_data(self) -> Dict:
        """Получение всех рыночных данных"""
        try:
            btc_ticker = self.exchange.fetch_ticker('BTC/USD')
            eth_ticker = self.exchange.fetch_ticker('ETH/USD')
            
            btc_ohlcv_1h = self.exchange.fetch_ohlcv('BTC/USD', '1h', limit=168)
            btc_ohlcv_1d = self.exchange.fetch_ohlcv('BTC/USD', '1d', limit=90)
            
            eth_ohlcv_1h = self.exchange.fetch_ohlcv('ETH/USD', '1h', limit=48)
            
            btc_df_1h = pd.DataFrame(btc_ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            btc_df_1d = pd.DataFrame(btc_ohlcv_1d, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            eth_df_1h = pd.DataFrame(eth_ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            btc_price = btc_ticker.get('last', btc_ticker.get('close', 0))
            eth_price = eth_ticker.get('last', eth_ticker.get('close', 0))
            
            btc_7d_ago = float(btc_df_1d['close'].iloc[-7]) if len(btc_df_1d) >= 7 else btc_price
            btc_change_7d = ((btc_price / btc_7d_ago) - 1) * 100
            
            eth_7d_ago = float(eth_df_1h['close'].iloc[-168]) if len(eth_df_1h) >= 168 else eth_price
            eth_change_7d = ((eth_price / eth_7d_ago) - 1) * 100 if eth_7d_ago else 0
            
            return {
                'btc': {
                    'price': btc_price,
                    'change_24h': btc_ticker.get('percentage') or 0.0,
                    'change_7d': btc_change_7d,
                    'df_1h': btc_df_1h,
                    'df_1d': btc_df_1d
                },
                'eth': {
                    'price': eth_price,
                    'change_24h': eth_ticker.get('percentage') or 0.0,
                    'change_7d': eth_change_7d,
                    'df_1h': eth_df_1h
                }
            }
        except Exception as e:
            logger.error(f"Data fetch error: {e}")
            raise
    
    def enrich_with_indicators(self, data: Dict) -> Dict:
        """Добавление технических индикаторов"""
        btc_df = data['btc']['df_1d'].copy()
        btc_df['close'] = pd.to_numeric(btc_df['close'])
        
        # RSI
        rsi_indicator = ta.momentum.RSIIndicator(btc_df['close'], window=14)
        data['btc']['rsi'] = float(rsi_indicator.rsi().iloc[-1])
        
        # EMAs
        data['btc']['ema20'] = float(ta.trend.EMAIndicator(btc_df['close'], window=20).ema_indicator().iloc[-1])
        data['btc']['ema50'] = float(ta.trend.EMAIndicator(btc_df['close'], window=50).ema_indicator().iloc[-1])
        
        ema200_series = ta.trend.EMAIndicator(btc_df['close'], window=200).ema_indicator()
        data['btc']['ema200'] = float(ema200_series.iloc[-1]) if not pd.isna(ema200_series.iloc[-1]) else data['btc']['ema50']
        
        price = data['btc']['price']
        data['btc']['above_ema20'] = price > data['btc']['ema20']
        data['btc']['above_ema50'] = price > data['btc']['ema50']
        data['btc']['above_ema200'] = price > data['btc']['ema200']
        
        # MACD
        macd = ta.trend.MACD(btc_df['close'])
        data['btc']['macd_hist'] = float(macd.macd_diff().iloc[-1])
        
        # Bollinger Bands position
        bb = ta.volatility.BollingerBands(btc_df['close'], window=20, window_dev=2)
        bb_upper = float(bb.bollinger_hband().iloc[-1])
        bb_lower = float(bb.bollinger_lband().iloc[-1])
        data['btc']['bb_position'] = ((price - bb_lower) / (bb_upper - bb_lower)) * 100 if bb_upper != bb_lower else 50
        
        # Volume ratio
        vol_sma = btc_df['volume'].rolling(20).mean().iloc[-1]
        data['btc']['vol_ratio'] = float(btc_df['volume'].iloc[-1] / vol_sma) if vol_sma > 0 else 1.0
        
        return data
    
    # ═══════════════════════════════════════════════════════════
    # REGIME CLASSIFICATION
    # ═══════════════════════════════════════════════════════════
    
    def classify_regime(self, data: Dict) -> Dict:
        """Классификация рыночного режима"""
        price = data['btc']['price']
        rsi = data['btc'].get('rsi', 50)
        above_ema20 = data['btc'].get('above_ema20', True)
        above_ema50 = data['btc'].get('above_ema50', True)
        change_7d = data['btc'].get('change_7d', 0)
        macd_hist = data['btc'].get('macd_hist', 0)
        
        # Scoring
        score = 0
        
        # EMA structure
        if above_ema20 and above_ema50:
            score += 2
        elif not above_ema20 and not above_ema50:
            score -= 2
        elif above_ema20:
            score += 1
        else:
            score -= 1
        
        # 7d momentum
        if change_7d > 8:
            score += 2
        elif change_7d > 3:
            score += 1
        elif change_7d < -8:
            score -= 2
        elif change_7d < -3:
            score -= 1
        
        # RSI
        if rsi > 60:
            score += 1
        elif rsi < 40:
            score -= 1
        
        # MACD
        if macd_hist > 0:
            score += 1
        else:
            score -= 1
        
        # Determine regime
        if score >= 4:
            regime = "BULL"
            qualifier = None
        elif score >= 2:
            regime = "BULL"
            qualifier = "early"
        elif score <= -4:
            regime = "BEAR"
            qualifier = None
        elif score <= -2:
            regime = "BEAR"
            qualifier = "early"
        else:
            regime = "TRANSITION"
            qualifier = None
        
        # Confidence
        base_conf = min(abs(score) * 12, 50)
        if (above_ema20 and above_ema50) or (not above_ema20 and not above_ema50):
            base_conf += 15
        if (regime == "BULL" and rsi > 55) or (regime == "BEAR" and rsi < 45):
            base_conf += 10
        confidence = min(base_conf, 85)
        
        # Tail risk
        tail_risk = "INACTIVE"
        tail_dir = None
        bb_pos = data['btc'].get('bb_position', 50)
        
        if regime == "BULL" or (qualifier == "early" and score > 0):
            if rsi > 75 or bb_pos > 90:
                tail_risk = "ACTIVE"
                tail_dir = "↓"
            elif rsi > 68 or bb_pos > 80:
                tail_risk = "ELEVATED"
                tail_dir = "↓"
        elif regime == "BEAR":
            if rsi < 25 or bb_pos < 10:
                tail_risk = "ACTIVE"
                tail_dir = "↓"
            elif rsi < 32 or bb_pos < 20:
                tail_risk = "ELEVATED"
                tail_dir = "↓"
        
        # Format regime string
        regime_str = f"{regime} ({qualifier})" if qualifier else regime
        
        return {
            'regime': regime_str,
            'regime_base': regime,
            'qualifier': qualifier,
            'confidence': confidence,
            'tail_risk': tail_risk,
            'tail_direction': tail_dir,
            'score': score
        }
    
    # ═══════════════════════════════════════════════════════════
    # TRIGGER LOGIC
    # ═══════════════════════════════════════════════════════════
    
    def get_round_level(self, price: float) -> int:
        """Определение ближайшего круглого уровня (5k шаг)"""
        return int(price // 5000) * 5000
    
    def check_triggers(self, data: Dict, regime_data: Dict) -> Tuple[bool, str]:
        """Проверка триггеров публикации"""
        triggers = []
        
        # 1. Значительное движение 24h (>5%)
        btc_change = abs(data['btc']['change_24h'])
        if btc_change > 5.0:
            triggers.append(f"BTC {data['btc']['change_24h']:+.1f}% за 24h")
        
        # 2. Значительное движение 7d (>10%)
        btc_7d = abs(data['btc'].get('change_7d', 0))
        if btc_7d > 10.0:
            triggers.append(f"BTC {data['btc']['change_7d']:+.1f}% за 7d")
        
        # 3. Смена режима
        current_regime = regime_data['regime']
        last_regime = self.state.get('last_regime')
        if last_regime and current_regime != last_regime:
            triggers.append(f"Режим: {last_regime} → {current_regime}")
        
        # 4. Пробой круглого уровня
        current_level = self.get_round_level(data['btc']['price'])
        last_level = self.state.get('last_round_level')
        
        if last_level and current_level != last_level:
            # Cooldown 4 часа
            last_round_pub = self.state.get('last_round_publish')
            cooldown_ok = True
            
            if last_round_pub:
                try:
                    elapsed = (datetime.utcnow() - datetime.fromisoformat(last_round_pub)).total_seconds()
                    cooldown_ok = elapsed > 4 * 3600
                except:
                    pass
            
            if cooldown_ok:
                direction = "выше" if current_level > last_level else "ниже"
                triggers.append(f"BTC пробил {direction} ${current_level:,}")
                self.state['last_round_publish'] = datetime.utcnow().isoformat()
        
        # 5. Tail risk активен
        if regime_data['tail_risk'] == "ACTIVE":
            triggers.append("Tail risk ACTIVE")
        
        return (len(triggers) > 0, ' | '.join(triggers) if triggers else '')
    
    # ═══════════════════════════════════════════════════════════
    # INTERPRETATION
    # ═══════════════════════════════════════════════════════════
    
    def build_interpretation(self, data: Dict, regime_data: Dict) -> list:
        """Построение интерпретации"""
        lines = []
        regime = regime_data['regime_base']
        rsi = data['btc'].get('rsi', 50)
        change_24h = data['btc'].get('change_24h', 0)
        change_7d = data['btc'].get('change_7d', 0)
        above_ema20 = data['btc'].get('above_ema20', True)
        above_ema50 = data['btc'].get('above_ema50', True)
        vol_ratio = data['btc'].get('vol_ratio', 1.0)
        
        # Structure
        if regime == "BEAR":
            if change_24h > 2:
                lines.append("Short-term bounce в рамках нисходящего тренда")
            else:
                lines.append("Ценовая структура остаётся медвежьей")
        elif regime == "BULL":
            if change_24h < -2:
                lines.append("Краткосрочная коррекция в рамках восходящего тренда")
            else:
                lines.append("Ценовая структура остаётся бычьей")
        else:
            lines.append("Рынок в переходном состоянии")
        
        # Momentum
        if rsi > 70:
            lines.append("Моментум перегрет — риск отката")
        elif rsi < 30:
            lines.append("Моментум перепродан — возможен отскок")
        elif (regime == "BEAR" and rsi < 45) or (regime == "BULL" and rsi > 55):
            lines.append("Моментум подтверждает текущий режим")
        else:
            lines.append("Моментум нейтральный")
        
        # Volume
        if vol_ratio < 0.6:
            lines.append("Объёмы аномально низкие")
        elif vol_ratio > 1.5:
            lines.append("Повышенные объёмы — возможно начало движения")
        
        # EMA structure
        if not above_ema20 and not above_ema50 and regime == "BEAR":
            lines.append("Цена ниже ключевых MA — слабость подтверждена")
        elif above_ema20 and above_ema50 and regime == "BULL":
            lines.append("Цена выше ключевых MA — сила подтверждена")
        
        return lines[:4]  # Max 4 lines
    
    # ═══════════════════════════════════════════════════════════
    # AI FORECAST
    # ═══════════════════════════════════════════════════════════
    
    def generate_ai_analysis(self, data: Dict, regime_data: Dict) -> str:
        """Генерация AI анализа"""
        try:
            price = data['btc']['price']
            rsi = data['btc'].get('rsi', 50)
            ema200 = data['btc'].get('ema200', price)
            change_7d = data['btc'].get('change_7d', 0)
            
            system_prompt = """You are a crypto market regime analyst. Output ONLY the following sections, nothing else:

Bias:
• [1 sentence about directional pressure]
• [1 sentence about risk asymmetry]

Key implication:
[1 sentence: what this regime means for positioning]

Directional policy:
• Longs: [encouraged/neutral/discouraged]
• Shorts: [encouraged/tactical only/discouraged]

Reference zones (not signals):
• $[X] — [what breach means]
• $[Y] — [what breach means]

What would change the view:
• [Condition 1]
• [Condition 2]

RULES:
- NO bold, NO emojis, NO hashtags
- Under 150 words
- Zones are REFERENCES not trade signals
- Use "breach would imply" language"""

            user_prompt = f"""Regime: {regime_data['regime']}
Confidence: {regime_data['confidence']}%
Tail risk: {regime_data['tail_risk']} {regime_data['tail_direction'] or ''}
Score: {regime_data['score']}

BTC Price: ${price:,.0f}
RSI: {rsi:.1f}
EMA200: ${ema200:,.0f}
7d change: {change_7d:+.1f}%

Generate regime analysis."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"AI generation error: {e}")
            return self._fallback_analysis(data, regime_data)
    
    def _fallback_analysis(self, data: Dict, regime_data: Dict) -> str:
        """Fallback анализ без AI"""
        regime = regime_data['regime_base']
        price = data['btc']['price']
        ema200 = data['btc'].get('ema200', price)
        
        if regime == "BEAR":
            return f"""Bias:
• Directional downside pressure persists
• Risk asymmetry favors downside moves

Key implication:
Current conditions favor capital preservation over aggressive positioning.

Directional policy:
• Longs: discouraged
• Shorts: tactical only
• Position size: reduced

Reference zones (not signals):
• ${ema200:,.0f} — breach above would imply bias invalidation
• ${price * 0.9:,.0f} — breach below would imply acceleration

What would change the view:
• Sustained move above EMA200
• Shift in macro risk sentiment"""
        else:
            return f"""Bias:
• Directional structure improving
• Risk asymmetry shifting neutral

Key implication:
Market in transition, await confirmation before aggressive positioning.

Directional policy:
• Longs: neutral
• Shorts: discouraged
• Position size: normal

Reference zones (not signals):
• ${price * 1.05:,.0f} — breach would confirm strength
• ${price * 0.95:,.0f} — breach would resume weakness

What would change the view:
• Clear break of current range
• Volume confirmation"""
    
    # ═══════════════════════════════════════════════════════════
    # MESSAGE FORMATTING
    # ═══════════════════════════════════════════════════════════
    
    def format_message(self, data: Dict, regime_data: Dict, trigger_reason: str, ai_analysis: str) -> str:
        """Форматирование сообщения для Telegram"""
        
        regime = regime_data['regime']
        confidence = regime_data['confidence']
        tail_risk = regime_data['tail_risk']
        tail_dir = regime_data['tail_direction'] or ''
        
        btc_price = data['btc']['price']
        btc_24h = data['btc']['change_24h']
        btc_7d = data['btc'].get('change_7d', 0)
        eth_price = data['eth']['price']
        eth_24h = data['eth']['change_24h']
        
        # Regime emoji
        if 'BULL' in regime:
            regime_emoji = '🟢'
        elif 'BEAR' in regime:
            regime_emoji = '🔴'
        else:
            regime_emoji = '🟡'
        
        # Interpretation
        interpretation = self.build_interpretation(data, regime_data)
        interp_text = '\n'.join([f"• {line}" for line in interpretation])
        
        message = f"""<b>BITCOIN · MARKET STATE</b>

{regime_emoji} <b>{regime}</b>
Confidence: {confidence}%
Tail risk: {tail_risk} {tail_dir}

<b>Prices</b>
• BTC: ${btc_price:,.0f} ({btc_24h:+.1f}% 24h | {btc_7d:+.1f}% 7d)
• ETH: ${eth_price:,.0f} ({eth_24h:+.1f}% 24h)

<b>Interpretation</b>
{interp_text}

{ai_analysis}

<b>What changed</b>
{trigger_reason}

<i>OracAI Radar | {datetime.utcnow().strftime('%d %b %Y %H:%M UTC')}</i>"""
        
        return message
    
    # ═══════════════════════════════════════════════════════════
    # TELEGRAM
    # ═══════════════════════════════════════════════════════════
    
    def publish_telegram(self, message: str):
        """Публикация в Telegram"""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                'chat_id': self.channel_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("Published to Telegram")
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            raise
    
    # ═══════════════════════════════════════════════════════════
    # MAIN RUN
    # ═══════════════════════════════════════════════════════════
    
    def run(self):
        """Основной метод"""
        try:
            logger.info("=" * 50)
            logger.info("OracAI Radar starting")
            logger.info("=" * 50)
            
            # 1. Fetch data
            logger.info("Step 1: Fetching market data")
            data = self.fetch_market_data()
            
            # 2. Add indicators
            logger.info("Step 2: Calculating indicators")
            data = self.enrich_with_indicators(data)
            
            # 3. Classify regime
            logger.info("Step 3: Classifying regime")
            regime_data = self.classify_regime(data)
            logger.info(f"Regime: {regime_data['regime']} (conf: {regime_data['confidence']}%)")
            
            # 4. Check triggers
            logger.info("Step 4: Checking triggers")
            should_publish, trigger_reason = self.check_triggers(data, regime_data)
            
            if not should_publish:
                logger.info("No triggers met - skipping publication")
                # Update state anyway
                self.state['last_regime'] = regime_data['regime']
                self.state['last_round_level'] = self.get_round_level(data['btc']['price'])
                self._save_state()
                logger.info("State updated, exiting")
                return
            
            logger.info(f"Trigger: {trigger_reason}")
            
            # 5. Generate AI analysis
            logger.info("Step 5: Generating AI analysis")
            ai_analysis = self.generate_ai_analysis(data, regime_data)
            
            # 6. Format message
            logger.info("Step 6: Formatting message")
            message = self.format_message(data, regime_data, trigger_reason, ai_analysis)
            
            # 7. Publish
            logger.info("Step 7: Publishing to Telegram")
            self.publish_telegram(message)
            
            # 8. Update state
            self.state['last_regime'] = regime_data['regime']
            self.state['last_publish'] = datetime.utcnow().isoformat()
            self.state['last_round_level'] = self.get_round_level(data['btc']['price'])
            self._save_state()
            
            logger.info("=" * 50)
            logger.info("OracAI Radar completed successfully!")
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"Critical error: {e}", exc_info=True)
            sys.exit(1)


def main():
    bot = OracAIRadar()
    bot.run()


if __name__ == "__main__":
    main()
