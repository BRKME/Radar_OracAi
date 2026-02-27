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
            'last_round_publish': None,
            'last_tail_risk_publish': None
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
            
            btc_change_24h = btc_ticker.get('percentage') or 0.0
            eth_change_24h = eth_ticker.get('percentage') or 0.0
            
            # Log for debugging
            logger.info(f"BTC: ${btc_price:,.0f} | 24h: {btc_change_24h}% | 7d: {btc_change_7d:.1f}%")
            logger.info(f"ETH: ${eth_price:,.0f} | 24h: {eth_change_24h}%")
            
            return {
                'btc': {
                    'price': btc_price,
                    'change_24h': btc_change_24h,
                    'change_7d': btc_change_7d,
                    'df_1h': btc_df_1h,
                    'df_1d': btc_df_1d
                },
                'eth': {
                    'price': eth_price,
                    'change_24h': eth_change_24h,
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
        """Check publication triggers"""
        triggers = []
        btc_price = data['btc']['price']
        
        # 1. Significant 24h move (>5%)
        # Kraken returns percentage as-is (e.g., 5.2 means 5.2%)
        btc_change = data['btc']['change_24h']
        
        logger.info(f"Checking 24h trigger: {btc_change:.2f}% (threshold: >5%)")
        if abs(btc_change) > 5.0:
            triggers.append(f"BTC {btc_change:+.1f}% in 24h")
            logger.info(f"Trigger FIRED: 24h change {btc_change:+.1f}%")
        
        # 2. Significant 7d move (>10%)
        btc_7d = data['btc'].get('change_7d', 0)
        
        logger.info(f"Checking 7d trigger: {btc_7d:.2f}% (threshold: >10%)")
        if abs(btc_7d) > 10.0:
            triggers.append(f"BTC {btc_7d:+.1f}% in 7d")
            logger.info(f"Trigger FIRED: 7d change {btc_7d:+.1f}%")
        
        # 3. Regime change
        current_regime = regime_data['regime']
        last_regime = self.state.get('last_regime')
        if last_regime and current_regime != last_regime:
            triggers.append(f"Regime: {last_regime} → {current_regime}")
            logger.info(f"Trigger FIRED: Regime change {last_regime} → {current_regime}")
        
        # 4. Round level breakout ($5000 step)
        current_level = self.get_round_level(btc_price)
        last_level = self.state.get('last_round_level')
        
        logger.info(f"Price: ${btc_price:,.0f}, Current level: ${current_level:,}, Last level: {last_level}")
        
        # Initialize last_level if None
        if last_level is None:
            logger.info(f"Initializing last_round_level to ${current_level:,}")
            self.state['last_round_level'] = current_level
            last_level = current_level
        
        if current_level != last_level:
            # Check cooldown (4 hours)
            last_round_pub = self.state.get('last_round_publish')
            cooldown_ok = True
            
            if last_round_pub:
                try:
                    elapsed = (datetime.utcnow() - datetime.fromisoformat(last_round_pub)).total_seconds()
                    cooldown_ok = elapsed > 4 * 3600
                    logger.info(f"Round level cooldown: {elapsed/3600:.1f}h elapsed, ok={cooldown_ok}")
                except Exception as e:
                    logger.warning(f"Cooldown parse error: {e}")
            
            if cooldown_ok:
                # Calculate how many levels were crossed
                levels_crossed = abs(current_level - last_level) // 5000
                direction = "above" if current_level > last_level else "below"
                
                if levels_crossed > 1:
                    triggers.append(f"BTC broke {direction} ${current_level:,} (crossed {int(levels_crossed)} levels)")
                else:
                    triggers.append(f"BTC broke {direction} ${current_level:,}")
                    
                logger.info(f"Trigger FIRED: Round level ${last_level:,} → ${current_level:,}")
                self.state['last_round_publish'] = datetime.utcnow().isoformat()
                self.state['last_round_level'] = current_level
        
        # 5. Tail risk active (6 hours cooldown)
        if regime_data['tail_risk'] == "ACTIVE":
            last_tail_pub = self.state.get('last_tail_risk_publish')
            tail_cooldown_ok = True
            
            if last_tail_pub:
                try:
                    elapsed = (datetime.utcnow() - datetime.fromisoformat(last_tail_pub)).total_seconds()
                    tail_cooldown_ok = elapsed > 6 * 3600  # 6 hours cooldown
                    logger.info(f"Tail risk cooldown: {elapsed/3600:.1f}h elapsed, ok={tail_cooldown_ok}")
                except Exception as e:
                    logger.warning(f"Tail cooldown parse error: {e}")
            
            if tail_cooldown_ok:
                triggers.append("Tail risk ACTIVE")
                self.state['last_tail_risk_publish'] = datetime.utcnow().isoformat()
                logger.info("Trigger: Tail risk ACTIVE")
        
        logger.info(f"Total triggers: {len(triggers)} - {triggers}")
        return (len(triggers) > 0, ' | '.join(triggers) if triggers else '')
    
    # ═══════════════════════════════════════════════════════════
    # INTERPRETATION
    # ═══════════════════════════════════════════════════════════
    
    def build_interpretation(self, data: Dict, regime_data: Dict) -> list:
        """Build interpretation in English"""
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
                lines.append("Short-term bounce within downtrend")
            else:
                lines.append("Price structure remains bearish")
        elif regime == "BULL":
            if change_24h < -2:
                lines.append("Short-term pullback within uptrend")
            else:
                lines.append("Price structure remains bullish")
        else:
            lines.append("Market in transition")
        
        # Momentum
        if rsi > 70:
            lines.append("Momentum overheated — pullback risk elevated")
        elif rsi < 30:
            lines.append("Momentum oversold — bounce possible")
        elif (regime == "BEAR" and rsi < 45) or (regime == "BULL" and rsi > 55):
            lines.append("Momentum confirms current regime")
        else:
            lines.append("Momentum neutral")
        
        # Volume
        if vol_ratio < 0.6:
            lines.append("Volume abnormally low")
        elif vol_ratio > 1.5:
            lines.append("Elevated volume — potential move starting")
        
        # EMA structure
        if not above_ema20 and not above_ema50 and regime == "BEAR":
            lines.append("Price below key MAs — weakness confirmed")
        elif above_ema20 and above_ema50 and regime == "BULL":
            lines.append("Price above key MAs — strength confirmed")
        
        return lines[:4]  # Max 4 lines
    
    # ═══════════════════════════════════════════════════════════
    # AI FORECAST
    # ═══════════════════════════════════════════════════════════
    
    def generate_ai_analysis(self, data: Dict, regime_data: Dict) -> str:
        """Generate AI analysis - compact, action-focused"""
        try:
            price = data['btc']['price']
            rsi = data['btc'].get('rsi', 50)
            ema200 = data['btc'].get('ema200', price)
            change_7d = data['btc'].get('change_7d', 0)
            
            system_prompt = """You are a crypto market analyst. Write for a general audience.

Output EXACTLY this format:

◼️ [1-2 sentences: What's happening and what to do. Simple language.]

<b>Positioning</b>
🟢 New longs: [low risk / moderate risk / high risk]
⚠️ Aggressive buying: [encouraged / neutral / discouraged]  
🛡️ Defensive stance: [preferred / neutral / not needed]

<b>What Would Change This</b>
• [Simple condition 1]
• [Simple condition 2]

RULES:
- NO technical jargon
- Under 80 words total
- Focus on ACTION, not analysis
- Use the exact emoji and formatting shown"""

            user_prompt = f"""Regime: {regime_data['regime']}
Confidence: {regime_data['confidence']}%
Tail risk: {regime_data['tail_risk']}
BTC: ${price:,.0f}, RSI: {rsi:.0f}, 7d: {change_7d:+.1f}%

Generate compact analysis."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"AI generation error: {e}")
            return self._fallback_analysis(data, regime_data)
    
    def _fallback_analysis(self, data: Dict, regime_data: Dict) -> str:
        """Fallback analysis - compact format with icons"""
        regime = regime_data['regime_base']
        
        if regime == "BEAR":
            return """◼️ Market is weak. Selling pressure dominates. Wait for clearer signals before buying.

<b>Positioning</b>
🟢 New longs: high risk
⚠️ Aggressive buying: discouraged
🛡️ Defensive stance: preferred

<b>What Would Change This</b>
• Sustained price recovery
• Shift in market sentiment"""
        elif regime == "BULL":
            return """◼️ Market is strong. Buyers are in control. Dips may offer opportunities.

<b>Positioning</b>
🟢 New longs: low risk
⚠️ Aggressive buying: neutral
🛡️ Defensive stance: not needed

<b>What Would Change This</b>
• Significant price breakdown
• Loss of buying momentum"""
        else:
            return """◼️ Market is undecided. Neither buyers nor sellers dominate. Best to wait for direction.

<b>Positioning</b>
🟢 New longs: moderate risk
⚠️ Aggressive buying: discouraged
🛡️ Defensive stance: preferred

<b>What Would Change This</b>
• Clear trend formation
• Decisive breakout in either direction"""
    
    # ═══════════════════════════════════════════════════════════
    # MESSAGE FORMATTING
    # ═══════════════════════════════════════════════════════════
    
    def format_message(self, data: Dict, regime_data: Dict, trigger_reason: str, ai_analysis: str) -> str:
        """Format message for Telegram - Clean UX-focused structure"""
        
        regime = regime_data['regime']
        confidence = regime_data['confidence']
        tail_risk = regime_data['tail_risk']
        
        btc_price = data['btc']['price']
        btc_24h = data['btc']['change_24h']
        btc_7d = data['btc'].get('change_7d', 0)
        eth_price = data['eth']['price']
        eth_24h = data['eth']['change_24h']
        
        # Timestamp
        timestamp = datetime.utcnow().strftime('%d %b %Y · %H:%M UTC')
        
        # Regime emoji and name
        if 'BULL' in regime:
            regime_emoji = '🟢'
            regime_name = "Bullish"
            if 'early' in regime.lower():
                regime_name = "Bullish (early)"
        elif 'BEAR' in regime:
            regime_emoji = '🔴'
            regime_name = "Bearish"
            if 'early' in regime.lower():
                regime_name = "Bearish (early)"
        else:
            regime_emoji = '🟡'
            regime_name = "Transition"
        
        # Visual confidence bar [████░░░░░░] 27%
        filled = int(confidence / 10)
        empty = 10 - filled
        conf_bar = '█' * filled + '░' * empty
        
        # Price change formatting with color
        def format_change(val):
            if val > 0:
                return f"<b>+{val:.1f}%</b>"
            elif val < 0:
                return f"{val:.1f}%"
            else:
                return f"{val:.1f}%"
        
        # Key levels (round to nearest $1000)
        support = int((btc_price * 0.92) // 1000) * 1000
        resistance = int((btc_price * 1.08) // 1000) * 1000
        current_k = int(btc_price // 1000)
        
        # Visual price scale
        price_scale = f"📉 ${support//1000}k ····· ${current_k}k ····· ${resistance//1000}k 📈"
        
        # Risk flag
        risk_line = ""
        if tail_risk == "ACTIVE":
            risk_line = "\n⚠️ <b>Elevated risk of sharp moves</b>"
        
        message = f"""<b>BTC/USD</b> · {timestamp}

{regime_emoji} <b>{regime_name}</b>
[{conf_bar}] {confidence}%{risk_line}

<b>Prices</b>
BTC ${btc_price:,.0f}  {format_change(btc_24h)} 24h | {btc_7d:+.1f}% 7d
ETH ${eth_price:,.0f}  {format_change(eth_24h)} 24h

{ai_analysis}

<b>Key Levels</b>
{price_scale}

<i>OracAI Radar</i>"""
        
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
