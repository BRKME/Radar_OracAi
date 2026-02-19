# OracAI Radar

Адаптивный бот для мониторинга крипторынка с AI анализом.

## Особенности

- **Триггерная логика** — публикует только при значимых событиях:
  - Смена режима (BULL/BEAR/TRANSITION)
  - Движение >5% за 24h
  - Движение >10% за 7d
  - Пробой круглых уровней ($65k, $70k, etc)
  - Активация Tail Risk

- **AI анализ** — GPT-4o генерирует:
  - Bias и risk asymmetry
  - Directional policy
  - Reference zones
  - What would change the view

- **Формат сообщения:**
```
BITCOIN · MARKET STATE

🔴 BEAR (early)
Confidence: 61%
Tail risk: ELEVATED ↓

Prices
• BTC: $67,100 (+1.0% 24h | -2.5% 7d)
• ETH: $1,976 (+1.1% 24h)

Interpretation
• Ценовая структура остаётся медвежьей
• Моментум подтверждает текущий режим
• Объёмы аномально низкие

[AI Analysis]

What changed
Режим: TRANSITION → BEAR (early)
```

## Настройка

### Secrets (GitHub Actions):
- `TELEGRAM_BOT_TOKEN` — токен Telegram бота
- `TELEGRAM_CHANNEL_ID` — ID канала
- `OPENAI_API_KEY` — ключ OpenAI API

### Запуск:
```bash
pip install -r requirements.txt
python main.py
```

## Триггеры

| Триггер | Порог |
|---------|-------|
| 24h движение | >5% |
| 7d движение | >10% |
| Смена режима | Любая |
| Круглый уровень | 5k шаг, cooldown 4h |
| Tail Risk | ACTIVE |

## Workflow

Запускается каждый час (`0 * * * *`), публикует только при наличии триггера.
