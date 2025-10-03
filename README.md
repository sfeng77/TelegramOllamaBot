# Telegram Ollama Bot

A Telegram assistant that proxies chat messages to a local Ollama instance. The bot keeps per-chat memory, supports persona overlays, and records transcripts for review.

## Features

- Async bot built with `python-telegram-bot` 20.x that forwards messages to Ollama at `OLLAMA_API_URL` (defaults to `http://localhost:11434/api/generate`).
- `/model` inline keyboard for switching between the models declared in `AVAILABLE_MODELS` (default alias: `gpt-oss:20b`).
- Persona prompts loaded from `personas/*.txt` and selected per chat with `/persona`.
- Per-chat conversation memory with optional SQLite persistence, `/forget` for quick resets, and `/recap` summaries for the last hour or day.
- Optional display of model reasoning blocks between `<<BEGIN THOUGHT>>` and `<<END THOUGHT>>`, toggled via `/thoughts`.
- Rotating operational and conversation logs written to `logs/bot.log` and `logs/conversations.log`.

## Prerequisites

- Python 3.8 or newer with `pip`.
- A running Ollama installation with the models you want to expose.
- A Telegram bot token issued by @BotFather.
- (Optional) A writable location for the history database if you enable persistence.

## Setup

1. Create a virtual environment:

   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy the environment template and set secrets:

   ```bash
   copy .env.example .env  # Windows
   cp .env.example .env    # macOS / Linux
   ```

4. Edit `.env` and provide:

   - `TELEGRAM_BOT_TOKEN=<token from BotFather>`
   - `ALLOWED_TELEGRAM_USER_IDS=<comma separated numeric IDs>` (the bot rejects everyone until this is populated)
   - Optional `DB_PATH` to control where the SQLite file lives (defaults to `logs/conversations.db`)
   - Optional `OLLAMA_API_URL` if Ollama is not on `http://localhost:11434/api/generate`

### Prepare Ollama

Start the Ollama service locally and pull the models that appear in `AVAILABLE_MODELS` inside `bot.py`. The default build expects `gpt-oss:20b`, but you can add others:

```bash
ollama pull gpt-oss:20b
# add more:
# ollama pull deepseek-r1:1.5b
# ollama pull deepseek-r1:32b
```

Update `AVAILABLE_MODELS` with the alias-to-model pairs you want to expose; the alias is what users see on the `/model` keyboard.

## Running the bot

With the virtual environment active and Ollama running:

```bash
python bot.py
```

Windows helpers are included: `start_ollama.bat` launches Ollama, `start_bot.bat` activates the virtual environment and starts the bot, and `setup_autostart.bat` plus `run_bot_hidden.vbs` provide background startup options.

Keep `logs/` out of commits; it stores `bot.log`, `conversations.log`, and the optional SQLite database.

## Telegram commands

Keep `commands.txt` in sync and re-upload it via @BotFather so the Telegram client shows the latest shortcuts. The bot handles:

- `/start` — greet the user, apply the default model/persona, and reset in-memory state.
- `/help` — list available commands, current models, and usage tips.
- `/model` — open an inline keyboard to switch between configured Ollama models.
- `/persona` — pick one of the persona prompts sourced from `personas/*.txt`.
- `/thoughts [on|off]` — toggle whether reasoning blocks between `<<BEGIN THOUGHT>>` and `<<END THOUGHT>>` appear in replies.
- `/forget` — clear the active chat history from memory (persisted transcripts remain available for recaps).
- `/recap <hour|day>` — summarize recent messages saved in storage for the chosen window; omit the argument to get inline buttons.

Send regular text messages after `/start` to chat with the currently selected model and persona.

## Personas

Persona prompts live in `personas/` as UTF-8 text files. The filename (without extension) becomes the keyboard label and key. Add new personas by placing a `.txt` file in that directory and restart the bot to load the changes. Use `/persona` to switch the persona for the current chat; the selection is stored per chat.

## Storage and logs

Operational logs go to `logs/bot.log`, and conversational transcripts are written to `logs/conversations.log`. When `DB_PATH` is set (or left as the default), SQLite persistence is enabled through `backends.storage`. The storage backend powers `/recap` and keeps a rolling history while the bot also maintains an in-memory window of the last 20 exchanges per chat.

## Development tips

- `python-telegram-bot` 20.x requires async handlers; keep new coroutines verb-first and log through the shared `logger` and `conversation_logger`.
- Add dependencies to `requirements.txt` and rerun `pip install -r requirements.txt` to update the virtual environment.
- When you change commands, update `commands.txt` and refresh the list with @BotFather.
- Add tests under a new `tests/` package using `pytest` and `pytest.mark.asyncio`; run them with `pytest -q`.
- Place experimental automation inside a `scripts/` folder to keep the root tidy.

## Troubleshooting

- If you see "Access denied" in Telegram, confirm that your numeric ID is listed in `ALLOWED_TELEGRAM_USER_IDS`.
- Check `logs/bot.log` for runtime errors and for Ollama request details; failures are mirrored to the chat with a generic error message.
- Ensure Ollama is reachable at the URL in `OLLAMA_API_URL`; adjust the env value if Ollama runs elsewhere.
- Large responses are split into 4000-character chunks automatically, but Telegram rate limits still apply; retries and failures are noted in the logs.
