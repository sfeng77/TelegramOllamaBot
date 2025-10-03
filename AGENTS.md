# Repository Guidelines

## Project Structure & Module Organization
`bot.py` hosts the asynchronous Telegram <> Ollama workflow: command handlers (`/start`, `/model`, `/forget`), conversation memory, and API calls. Environment templates live in `.env.example`; copy to `.env` with `TELEGRAM_BOT_TOKEN`. Runtime artifacts go to `logs/` (rotating `bot.log` and `conversations.log`) and should stay out of commits. Windows helpers (`start_bot.bat`, `start_ollama.bat`) wrap activation and startup; adjust them if paths change. `commands.txt` mirrors the command list you register with @BotFather. Keep experimental scripts inside a new `scripts/` subfolder if you add more automation.

## Build, Test, and Development Commands
Create or reuse the virtual env in `venv/`: `python -m venv venv` then `venv\Scripts\activate`. Install dependencies with `pip install -r requirements.txt`. Start Ollama locally and ensure the models listed in `AVAILABLE_MODELS` are pulled, e.g. `ollama pull gpt-oss:20b`. Run the bot via `python bot.py`; `start_bot.bat` chains Ollama startup, env activation, and the bot for Windows sessions. When tweaking commands, update `commands.txt` and re-upload the list to @BotFather.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and snake_case for functions (`handle_message`, `query_ollama`). Constants and configuration flags stay uppercase (`DEFAULT_MODEL`, `OLLAMA_API_URL`). Coroutines should read verb-first and log context-rich messages through the shared `logger` or `conversation_logger`. Keep user-facing strings ASCII; surrounding comments may use UTF-8 but prefer plain English for new notes. Serialize payloads with `json.dumps` and type-hint async signatures to match existing patterns.

## Testing Guidelines
There is no dedicated suite yet; new features should ship with `pytest` cases under a new `tests/` directory. Name files `test_<feature>.py` and arrange async tests with `pytest.mark.asyncio`. Run `pytest -q` locally before opening a PR, and document any skipped coverage. For manual verification, run `python bot.py`, issue `/model` and `/forget`, and review `logs/` for conversation traces.

## Commit & Pull Request Guidelines
Existing history mixes concise imperatives (`handle long replies`) with scoped prefixes (`feat:`). Aim for present-tense summaries under 72 characters and add an optional tag (`feat`, `fix`, `chore`) where it clarifies intent. Reference related issues in the body and note key commands you ran (e.g. `pytest -q`, manual bot checks). Pull requests should include: what changed, why, screenshots or log excerpts for user-visible updates, and configuration reminders when `.env` keys evolve.

## Operations & Configuration Tips
`load_dotenv()` reads settings at startup; keep secrets in `.env` and never in source. The bot expects Ollama at `http://localhost:11434/api/generate`; adjust `OLLAMA_API_URL` if you proxy the service. Logs rotate at 10 MB; archive the `logs/` directory before debugging multi-session interactions. Update `AVAILABLE_MODELS` carefully: use the display alias as the key and the Ollama model name as the value to keep inline keyboards tidy.
