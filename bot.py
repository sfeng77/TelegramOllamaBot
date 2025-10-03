import asyncio
import logging
import logging.handlers
import os
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Awaitable, Callable

import aiohttp
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, Message
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from backends import storage

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            LOG_DIR / "bot.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        ),
    ],
)

conversation_logger = logging.getLogger("conversation")
conversation_handler = logging.handlers.RotatingFileHandler(
    LOG_DIR / "conversations.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
conversation_formatter = logging.Formatter("%(asctime)s - %(message)s")
conversation_handler.setFormatter(conversation_formatter)
conversation_logger.addHandler(conversation_handler)
conversation_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

STORAGE_AVAILABLE = False
STORAGE_DB_PATH: str | None = None


def split_long_message(text: str, max_length: int = 4000) -> list[str]:
    """将长消息分割成多个较短的消息"""
    if len(text) <= max_length:
        return [text]
    
    messages = []
    current_message = ""
    
    # 按段落分割
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        # 如果当前段落加上当前消息超过限制
        if len(current_message) + len(paragraph) + 2 > max_length:
            if current_message:
                messages.append(current_message.strip())
                current_message = paragraph
            else:
                # 如果单个段落就超过限制，按句子分割
                sentences = paragraph.split('. ')
                for sentence in sentences:
                    if len(current_message) + len(sentence) + 2 > max_length:
                        if current_message:
                            messages.append(current_message.strip())
                        current_message = sentence
                    else:
                        current_message += ". " + sentence if current_message else sentence
        else:
            current_message += "\n\n" + paragraph if current_message else paragraph
    
    if current_message:
        messages.append(current_message.strip())
    
    return messages

REASONING_KEYS: set[str] = {"reasoning", "thinking", "thoughts"}


def _stringify_reasoning_segments(obj: Any) -> list[str]:
    """Recursively flatten reasoning data into plain text segments."""
    segments: list[str] = []
    if isinstance(obj, str):
        stripped = obj.strip()
        if stripped:
            segments.append(stripped)
        return segments
    if isinstance(obj, dict):
        type_hint = obj.get("type")
        text_value = obj.get("text")
        if isinstance(text_value, str) and type_hint in {"reasoning", "thinking"}:
            stripped = text_value.strip()
            if stripped:
                segments.append(stripped)
        elif isinstance(text_value, str) and type_hint is None:
            stripped = text_value.strip()
            if stripped:
                segments.append(stripped)
        for key in ("content", "value", "messages", "parts"):
            if key in obj:
                segments.extend(_stringify_reasoning_segments(obj[key]))
        for key, value in obj.items():
            if key in REASONING_KEYS:
                segments.extend(_stringify_reasoning_segments(value))
        return segments
    if isinstance(obj, list):
        if obj and all(isinstance(item, (int, float)) for item in obj):
            return segments
        for item in obj:
            segments.extend(_stringify_reasoning_segments(item))
        return segments
    return segments


def extract_reasoning_text(payload: dict[str, Any]) -> str:
    """Pull any reasoning or thinking content from an Ollama response payload."""
    collected: list[str] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in {"context", "embedding"}:
                    continue
                if key in REASONING_KEYS:
                    collected.extend(_stringify_reasoning_segments(value))
                elif isinstance(value, (dict, list)):
                    walk(value)
        elif isinstance(obj, list):
            if obj and all(isinstance(item, (int, float)) for item in obj):
                return
            for item in obj:
                walk(item)

    walk(payload)

    unique_text: list[str] = []
    for item in collected:
        stripped = item.strip()
        if stripped and stripped not in unique_text:
            unique_text.append(stripped)

    return "\n\n".join(unique_text)

async def send_long_message(update: Update, text: str, max_length: int = 4000):
    """发送可能很长的消息，自动分割"""
    messages = split_long_message(text, max_length)
    
    for i, message in enumerate(messages):
        if i == 0:
            # 第一条消息直接回复
            await update.message.reply_text(message)
        else:
            # 后续消息作为新消息发送
            await update.message.reply_text(f"（续 {i+1}/{len(messages)}）\n\n{message}")
        
        # 添加短暂延迟，避免发送过快
        if i < len(messages) - 1:
            await asyncio.sleep(0.5)
    
    # 记录长消息分割情况
    if len(messages) > 1:
        logger.info(f"长消息已分割为 {len(messages)} 条消息发送")

async def send_ai_response(update: Update, ai_response: str, time_str: str, thinking: str | None = None) -> None:
    """Send the AI response and optional reasoning text to the user."""
    sections: list[str] = []
    if thinking:
        sanitized_thinking = thinking.strip()
        if sanitized_thinking:
            sections.append("Thought process (model reasoning):\n<<BEGIN THOUGHT>>\n" + sanitized_thinking + "\n<<END THOUGHT>>")

    response_body = ai_response.strip()
    if response_body:
        sections.append(response_body)

    sections.append(f"Generated in {time_str}")

    combined_message = "\n\n".join(section for section in sections if section)
    await send_long_message(update, combined_message)


async def on_startup(application: Application) -> None:
    """Initialise persistent storage when the bot starts."""
    global STORAGE_AVAILABLE, STORAGE_DB_PATH

    db_path = os.getenv('DB_PATH') or str(LOG_DIR / "conversations.db")

    try:
        await storage.init_db(db_path)
    except Exception as exc:
        STORAGE_AVAILABLE = False
        STORAGE_DB_PATH = None
        logger.error("Failed to initialise persistent storage at %s: %s", db_path, exc)
    else:
        STORAGE_AVAILABLE = True
        STORAGE_DB_PATH = db_path
        logger.info("Persistent history database initialised at %s", db_path)

# Load environment variables
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_WEB_SEARCH_URL = os.getenv('OLLAMA_WEB_SEARCH_URL', 'http://localhost:11434/api/web_search')
OLLAMA_API_KEY = os.getenv('OLLAMA_API_KEY')

raw_websearch_toggle = os.getenv('ENABLE_WEBSEARCH', '1').strip().lower()
ENABLE_WEBSEARCH = raw_websearch_toggle not in {'0', 'false', 'no'}

try:
    WEBSEARCH_MAX_RESULTS = int(os.getenv('WEBSEARCH_MAX_RESULTS', '5') or '5')
except ValueError:
    logger.warning("Invalid WEBSEARCH_MAX_RESULTS value; defaulting to 5.")
    WEBSEARCH_MAX_RESULTS = 5

if WEBSEARCH_MAX_RESULTS < 1:
    logger.warning("WEBSEARCH_MAX_RESULTS must be positive; defaulting to 1.")
    WEBSEARCH_MAX_RESULTS = 1

ALLOWED_USER_IDS_ENV = os.getenv('ALLOWED_TELEGRAM_USER_IDS', '')
ALLOWED_USER_IDS: set[int] = set()
if ALLOWED_USER_IDS_ENV:
    for raw_value in ALLOWED_USER_IDS_ENV.split(','):
        trimmed_value = raw_value.strip()
        if not trimmed_value:
            continue
        try:
            ALLOWED_USER_IDS.add(int(trimmed_value))
        except ValueError:
            logger.warning("Invalid user ID '%s' in ALLOWED_TELEGRAM_USER_IDS; ignoring.", trimmed_value)
if not ALLOWED_USER_IDS:
    logger.warning("ALLOWED_TELEGRAM_USER_IDS is empty; bot will deny all users until configured.")

PERSONAS_DIR = Path('personas')


def load_personas(directory: Path) -> dict[str, str]:
    personas: dict[str, str] = {}
    if not directory.exists():
        return personas
    for persona_file in sorted(directory.glob('*.txt')):
        try:
            content = persona_file.read_text(encoding='utf-8').strip()
        except UnicodeDecodeError:
            logging.getLogger(__name__).warning(
                "Unable to read persona file %s due to encoding error",
                persona_file
            )
            continue
        if content:
            personas[persona_file.stem] = content
    return personas


PERSONAS = load_personas(PERSONAS_DIR)
DEFAULT_PERSONA_KEY = 'ChatBuddy' if 'ChatBuddy' in PERSONAS else next(iter(PERSONAS), None)
if not PERSONAS:
    logger.warning("No persona definitions found in %s", PERSONAS_DIR)

# 可用的模型列表
AVAILABLE_MODELS = {
    # 'deepseek_small': 'deepseek-r1:1.5b',
    # 'deepseek_large': 'deepseek-r1:32b',
    # 'llama2': 'llama2',
    # 'llama2-uncensored': 'llama2-uncensored',
    # 'mistral': 'mistral',
    # 'neural-chat': 'neural-chat',
    'gpt-oss:20b': 'gpt-oss:20b'
}

# 默认模型
DEFAULT_MODEL = 'gpt-oss:20b'

RECAP_RANGES: dict[str, tuple[int, str]] = {
    "hour": (3600, "Last Hour"),
    "day": (86400, "Last Day"),
}

RECAP_ALIAS_MAP: dict[str, str] = {
    "1h": "hour",
    "h": "hour",
    "24h": "day",
    "d": "day",
    "last hour": "hour",
    "last day": "day",
}


def resolve_recap_timespan(selection: str) -> tuple[int, str] | None:
    """Normalize user input into a supported recap timespan definition."""
    normalized = selection.lower().strip()
    normalized = RECAP_ALIAS_MAP.get(normalized, normalized)
    return RECAP_RANGES.get(normalized)


async def refine_search_query(user_query: str, model: str) -> str:
    """Use the model to produce a concise, search-optimized query string."""
    instructions = (
        "Rewrite the user-provided text into a concise web search query. "
        "Keep key nouns and context, remove filler, limit to 120 characters. "
        "Return only the refined query with no extra commentary."
    )
    prompt = f"{instructions}\n\nUser query:\n{user_query}\n\nRefined query:"
    refined_text, _, _ = await query_ollama(
        prompt=prompt,
        model=model,
        context_history=None,
        persona_prompt="",
        speaker_label=None,
    )

    candidate = (refined_text or "").strip()
    if not candidate:
        return user_query.strip()

    first_line = candidate.splitlines()[0].strip()
    cleaned = first_line.strip(" \"'")
    if not cleaned:
        return user_query.strip()

    if len(cleaned) > 120:
        cleaned = cleaned[:120].rstrip()
    return cleaned


def _normalize_search_results(raw_results: Any) -> list[dict[str, str]]:
    """Convert Ollama web search output into a predictable list of dicts."""
    if isinstance(raw_results, dict):
        items = raw_results.get("results", [])
    else:
        items = raw_results

    normalized: list[dict[str, str]] = []
    if not isinstance(items, list):
        return normalized

    for entry in items:
        if not isinstance(entry, dict):
            continue
        normalized.append(
            {
                "title": str(entry.get("title") or "").strip(),
                "url": str(entry.get("url") or "").strip(),
                "content": str(entry.get("content") or "").strip(),
            }
        )

    return normalized


async def perform_web_search(query: str, max_results: int) -> list[dict[str, str]]:
    """Call Ollama web search REST API with optional Bearer auth."""
    headers: dict[str, str] = {}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"

    payload = {"query": query, "count": max_results}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(OLLAMA_WEB_SEARCH_URL, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Web search HTTP {resp.status}: {text}")
                data: Any = await resp.json()
    except Exception as exc:
        raise RuntimeError(f"Web search request failed: {exc}") from exc

    normalized = _normalize_search_results(data)
    return normalized[:max_results]


def build_websearch_report_prompt(refined_query: str, results: list[dict[str, str]]) -> str:
    """Construct the summarisation prompt from search results."""

    def clip_excerpt(text: str, limit: int = 600) -> str:
        stripped = (text or "").strip()
        if len(stripped) <= limit:
            return stripped
        return stripped[:limit].rstrip() + "..."

    sources_lines: list[str] = []
    for index, result in enumerate(results, start=1):
        title = result.get("title", "").strip() or "(untitled)"
        url = result.get("url", "").strip() or "(no url)"
        excerpt = clip_excerpt(result.get("content", ""))
        sources_lines.append(
            f"[{index}] {title}\nURL: {url}\nExcerpt: {excerpt}\n"
        )

    sources_block = "\n".join(sources_lines).strip()

    guidelines = (
        "Task: Using ONLY the sources provided, write a brief factual Markdown report. "
        "Include three sections: Summary (3-5 bullets), Key Findings (bullets with numbered citations like [1]), "
        "and Sources (numbered list). Keep the overall length between 250 and 400 words. "
        "Do not invent facts or URLs. If evidence is thin, note the limitation briefly."
        "Use the language of the query."
    )

    format_hint = (
        f"# Research Report: {refined_query}\n\n"
        "## Summary\n"
        "- ...\n"
        "- ...\n\n"
        "## Key Findings\n"
        "- ... [1]\n"
        "- ... [2]\n\n"
        "## Sources\n"
        "[1] Title — URL\n"
        "[2] Title — URL\n"
    )

    prompt = (
        f"{guidelines}\n\n"
        f"Query: {refined_query}\n\n"
        f"Sources:\n{sources_block}\n\n"
        f"Follow this format:\n{format_hint}\n"
        "Begin your answer now:"
    )

    return prompt


async def summarize_search_results(
    refined_query: str,
    results: list[dict[str, str]],
    model: str,
    persona_prompt: str,
) -> tuple[str, str | None, float]:
    """Ask the model to produce a Markdown report from search results."""
    prompt = build_websearch_report_prompt(refined_query, results)
    return await query_ollama(
        prompt=prompt,
        model=model,
        context_history=None,
        persona_prompt=persona_prompt,
        speaker_label=None,
    )

async def _deny_access(update: Update) -> None:
    """Inform the user that access is restricted."""
    if update.callback_query:
        await update.callback_query.answer('Access denied. This bot is limited to approved users.', show_alert=True)
        return
    if update.effective_message:
        await update.effective_message.reply_text('Access denied. This bot is limited to approved users.')


async def ensure_user_allowed(update: Update) -> bool:
    """Return True when the update comes from a whitelisted user."""
    user = update.effective_user
    if not user:
        logger.warning('Received update without an effective user; denying access.')
        await _deny_access(update)
        return False

    user_identifier = f"{user.username or 'Unknown'}({user.id})"
    if user.id in ALLOWED_USER_IDS:
        return True

    logger.warning('Unauthorized access attempt from %s', user_identifier)
    conversation_logger.info(f"[Unauthorized {user_identifier}] Access denied by whitelist policy")
    await _deny_access(update)
    return False


def whitelist_required(handler: Callable[..., Awaitable[Any]]):
    """Wrap a handler so it only runs for whitelisted users."""
    @wraps(handler)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args: Any, **kwargs: Any):
        if not await ensure_user_allowed(update):
            return
        return await handler(update, context, *args, **kwargs)

    return wrapper

@whitelist_required
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理 /start 命令"""
    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"

    state = context.chat_data

    if not state.get('model'):
        state['model'] = DEFAULT_MODEL

    if not state.get('conversation_history'):
        state['conversation_history'] = []

    if PERSONAS and not state.get('persona'):
        state['persona'] = DEFAULT_PERSONA_KEY

    logger.info(f"用户 {username}({user_id}) 执行/start命令")
    conversation_logger.info(f"[用户 {username}({user_id})] 执行/start命令 - 初始化对话")

    current_model_key = state['model']
    current_model = AVAILABLE_MODELS[current_model_key]
    persona_value = state.get('persona') if PERSONAS else None
    if not persona_value:
        persona_value = "None"

    welcome_message = (
        "欢迎使用 AI 聊天机器人！\n\n"
        "这个机器人使用 Ollama 来回答您的问题。\n"
        f"当前使用的模型是：{current_model}\n"
        f"当前的助手角色是：{persona_value}\n"
        "/help 命令可查看使用说明。"
    )
    await send_long_message(update, welcome_message)

@whitelist_required
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理 /help 命令"""
    help_text = """
使用说明：

1. 直接发送文字消息即可与 AI 对话
2. 机器人会记住对话上下文，提供更连贯的对话体验
3. 使用 /model 命令可以切换不同的 AI 模型
4. 使用 /forget 命令可以清除对话历史，重新开始
5. 使用 /persona 命令可以选择助手角色
6. 使用 /thoughts 命令可以开关思考过程展示
7. 当前支持的模型：
{}
8. 如果遇到问题，请尝试重新发送消息
""".format('\n'.join(f'   - {name}' for name in AVAILABLE_MODELS.keys()))
    help_text += "\n9. Use /recap hour or /recap day to summarize recent messages.\n"
    await send_long_message(update, help_text)

@whitelist_required
async def thoughts_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /thoughts command to toggle reasoning visibility."""
    if not update.message:
        return

    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"
    state = context.chat_data
    state.setdefault('show_thoughts', False)
    current_state = state['show_thoughts']

    new_state = current_state
    if context.args:
        choice = context.args[0].lower()
        if choice in {"on", "enable", "true", "1"}:
            new_state = True
        elif choice in {"off", "disable", "false", "0"}:
            new_state = False
        else:
            await update.message.reply_text("Usage: /thoughts [on|off]")
            return
    else:
        new_state = not current_state

    state['show_thoughts'] = new_state
    state_label = "enabled" if new_state else "disabled"
    logger.info(f"Thought display {state_label} for user {username}({user_id})")

    if new_state:
        await update.message.reply_text("Thought process display enabled. I'll include sections between <<BEGIN THOUGHT>> and <<END THOUGHT>>.")
    else:
        await update.message.reply_text("Thought process display disabled. I'll keep the reasoning hidden.")


@whitelist_required
async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理 /model 命令"""
    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"
    state = context.chat_data
    current_model_key = state.get('model', DEFAULT_MODEL)

    logger.info(f"用户 {username}({user_id}) 执行/model命令 - 当前模型: {current_model_key}")

    keyboard = []
    for name, model in AVAILABLE_MODELS.items():
        current = '✓ ' if state.get('model', DEFAULT_MODEL) == name else ''
        keyboard.append([InlineKeyboardButton(f"{current}{model}", callback_data=f"model_{name}")])

    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text('请选择要使用的 AI 模型：', reply_markup=reply_markup)

@whitelist_required
async def persona_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /persona command."""
    if not PERSONAS:
        await update.message.reply_text("No personas are configured for this bot.")
        return

    state = context.chat_data
    if not state.get('persona'):
        state['persona'] = DEFAULT_PERSONA_KEY

    keyboard: list[list[InlineKeyboardButton]] = []
    current_persona = state.get('persona')
    for name in PERSONAS:
        prefix = "[*] " if current_persona == name else ""
        keyboard.append([InlineKeyboardButton(f"{prefix}{name}", callback_data=f"persona_{name}")])

    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Select the assistant persona:", reply_markup=reply_markup)

@whitelist_required
async def forget_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理 /forget 命令"""
    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"

    state = context.chat_data
    history_length = len(state.get('conversation_history', []))

    state['conversation_history'] = []

    logger.info(f"用户 {username}({user_id}) 执行/forget命令 - 清除了{history_length}轮对话历史")
    conversation_logger.info(f"[用户 {username}({user_id})] 执行/forget命令 - 清除了{history_length}轮对话历史")

    await update.message.reply_text("🧹 已清除所有对话历史，我们可以重新开始对话了！")

async def _send_recap_output(message: Message, text: str) -> None:
    """Send a recap message, splitting it when needed."""
    chunks = split_long_message(text)
    for index, chunk in enumerate(chunks):
        if index == 0:
            await message.reply_text(chunk)
        else:
            await message.reply_text(f"Part {index + 1}/{len(chunks)}\n\n{chunk}")
        if index < len(chunks) - 1:
            await asyncio.sleep(0.5)


@whitelist_required
async def recap_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /recap command with selectable time ranges."""
    if not update.message:
        return

    user = update.effective_user
    user_id = user.id if user else None
    username = (user.username or user.full_name or "Unknown") if user else "Unknown"

    selection: tuple[int, str] | None = None
    if context.args:
        arg_text = ' '.join(context.args).lower().strip()
        selection = resolve_recap_timespan(arg_text)
        if not selection:
            await update.message.reply_text('Invalid recap range. Use /recap hour or /recap day.')
            return

    if not selection:
        keyboard = [
            [
                InlineKeyboardButton('Last hour', callback_data='recap_hour'),
                InlineKeyboardButton('Last day', callback_data='recap_day'),
            ]
        ]
        await update.message.reply_text('Select the recap range:', reply_markup=InlineKeyboardMarkup(keyboard))
        return

    timespan_seconds, timespan_label = selection
    await _execute_recap(context, update.message, user_id, username, timespan_seconds, timespan_label)


async def _execute_recap(
    context: ContextTypes.DEFAULT_TYPE,
    message: Message | None,
    user_id: int | None,
    username: str,
    timespan_seconds: int,
    timespan_label: str,
) -> None:
    """Shared recap generator used by both command and button flows."""
    if not message:
        logger.error('No message available to reply to for recap.')
        return

    state = context.chat_data
    state.setdefault('model', DEFAULT_MODEL)
    state.setdefault('conversation_history', [])
    state.setdefault('show_thoughts', False)
    if PERSONAS and not state.get('persona'):
        state['persona'] = DEFAULT_PERSONA_KEY

    chat = message.chat
    chat_id = chat.id if chat else None
    chat_title = chat.title if chat and chat.title else None
    if chat_title:
        chat_label = f"{chat_title}({chat_id})"
    elif chat_id is not None:
        chat_label = f"Chat {chat_id}"
    else:
        chat_label = "Chat unknown"

    span_phrase = timespan_label.lower()
    now_ts = time.time()
    since_ts = now_ts - timespan_seconds

    timeline: list[tuple[float, str]] = []
    user_message_count = 0
    assistant_message_count = 0
    history_source = 'database' if STORAGE_AVAILABLE else 'memory'

    if STORAGE_AVAILABLE and chat_id is not None:
        try:
            persisted = await storage.fetch_messages(chat_id, since_ts, now_ts, limit=600)
            for entry in persisted:
                text_value = (entry.text or '').strip()
                if not text_value:
                    continue
                ts_value = entry.ts
                speaker = 'Assistant' if entry.role == 'assistant' else entry.sender_username or (f"User {entry.sender_id}" if entry.sender_id else 'User')
                timestamp_label = datetime.fromtimestamp(ts_value).strftime('%H:%M')
                timeline.append((ts_value, f"[{timestamp_label}] {speaker}: {text_value}"))
                if entry.role == 'assistant':
                    assistant_message_count += 1
                else:
                    user_message_count += 1
        except Exception as exc:
            history_source = 'memory'
            logger.warning('Failed to read persisted history for chat %s: %s', chat_id, exc)
            timeline.clear()
            user_message_count = assistant_message_count = 0
    else:
        history_source = 'memory'

    if not timeline:
        conversation_history: list[dict[str, Any]] = state['conversation_history']
        for item in conversation_history:
            raw_ts = item.get('timestamp')
            try:
                item_ts = float(raw_ts) if raw_ts is not None else None
            except (TypeError, ValueError):
                item_ts = None
            if item_ts is None:
                item_ts = now_ts
            if item_ts < since_ts or item_ts > now_ts:
                continue
            ts_label = datetime.fromtimestamp(item_ts).strftime('%H:%M')
            user_line = item.get('user', '').strip()
            assistant_line = item.get('assistant', '').strip()
            if user_line:
                timeline.append((item_ts, f"[{ts_label}] User: {user_line}"))
                user_message_count += 1
            if assistant_line:
                timeline.append((item_ts + 0.001, f"[{ts_label}] Assistant: {assistant_line}"))
                assistant_message_count += 1

    if not timeline:
        conversation_logger.info(
            f"[{chat_label}] [User {username}({user_id})] /recap requested for {timespan_label} but no activity."
        )
        await message.reply_text(f"No conversation activity in the {span_phrase} to summarize.")
        return

    timeline.sort(key=lambda item: item[0])
    transcript = "\n".join(line for _, line in timeline)
    exchange_count = max(user_message_count, assistant_message_count, 1)

    conversation_logger.info(
        f"[{chat_label}] [User {username}({user_id})] requested /recap ({timespan_label}) - considering {len(timeline)} messages from {history_source}."
    )

    status_message = await message.reply_text(f"Generating {span_phrase} recap...")

    current_model_key = state.get('model', DEFAULT_MODEL)
    current_model = AVAILABLE_MODELS.get(current_model_key, AVAILABLE_MODELS[DEFAULT_MODEL])

    persona_prompt = ''
    if PERSONAS:
        persona_key = state.get('persona', DEFAULT_PERSONA_KEY)
        persona_prompt = PERSONAS.get(persona_key, '')

    recap_prompt = (
        "You are a helpful assistant that summarizes chat conversations.\n"
        f"Using the following transcript from roughly the past {span_phrase}, create a concise recap that highlights:\n"
        "- Main topics or themes\n"
        "- Decisions or conclusions\n"
        "- Action items or follow-ups\n"
        "- Open questions or unresolved points\n"
        "Keep the recap brief and formatted as bullet points when appropriate.\n"
        "Use the language of the original discussions.\n\n"
        "Transcript:\n"
        f"{transcript}\n"
        "Recap:"
    )

    try:
        ai_response, thinking_text, generation_time = await query_ollama(
            recap_prompt,
            current_model,
            [],
            persona_prompt,
        )

        try:
            await status_message.delete()
        except Exception:
            pass

        if generation_time < 1:
            time_str = f"{generation_time*1000:.0f}ms"
        else:
            time_str = f"{generation_time:.2f}s"

        show_thoughts = state['show_thoughts']
        display_thinking = thinking_text if show_thoughts else None

        header_lines: list[str] = []
        if history_source != 'database':
            header_lines.append('Note: Using in-session history only (persistent log unavailable).')
        header_lines.append(f'Recap for the {span_phrase} ({exchange_count} exchanges):')
        recap_body = "\n".join(header_lines) + f"\n\n{ai_response.strip()}"

        sections: list[str] = []
        if display_thinking:
            sanitized = display_thinking.strip()
            if sanitized:
                sections.append("Thought process (model reasoning):\n<<BEGIN THOUGHT>>\n" + sanitized + "\n<<END THOUGHT>>")
        sections.append(recap_body.strip())
        sections.append(f"Generated in {time_str}")
        combined_message = "\n\n".join(section for section in sections if section)

        await _send_recap_output(message, combined_message)

        conversation_logger.info(
            f"[{chat_label}] [User {username}({user_id})] recap ({timespan_label}) generated successfully."
        )
    except Exception as exc:
        logger.error('Failed to generate recap: %s', str(exc))
        conversation_logger.error(
            f"[{chat_label}] [User {username}({user_id})] recap ({timespan_label}) generation failed: {str(exc)}"
        )
        try:
            await status_message.edit_text('Failed to generate recap. Please try again later.')
        except Exception:
            pass


@whitelist_required
async def _process_websearch_request(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    raw_query: str,
) -> None:
    """Internal helper to perform the full web search workflow."""
    message = update.message
    if not message:
        return

    state = context.chat_data
    state.pop('awaiting_websearch_query', None)

    trimmed_query = raw_query.strip()
    if not trimmed_query:
        await message.reply_text("Please provide a non-empty query for web search.")
        return

    user = update.effective_user
    username = (user.username or user.full_name or "Unknown") if user else "Unknown"
    user_id = user.id if user else 0
    chat = update.effective_chat
    chat_id = chat.id if chat else None
    chat_label = chat.title if chat and chat.title else (f"Chat {chat_id}" if chat_id else "Chat unknown")

    if not state.get('model'):
        state['model'] = DEFAULT_MODEL
    state.setdefault('show_thoughts', False)

    model_key = state.get('model', DEFAULT_MODEL)
    model = AVAILABLE_MODELS.get(model_key, DEFAULT_MODEL)

    persona_prompt = ''
    if PERSONAS:
        persona_name = state.get('persona', DEFAULT_PERSONA_KEY)
        persona_prompt = PERSONAS.get(persona_name, '')

    conversation_logger.info(
        f"[{chat_label}] [User {username}({user_id})] /websearch requested: {trimmed_query}"
    )

    status_message = await message.reply_text("Refining query...")

    try:
        refined_query = await refine_search_query(trimmed_query, model)
    except Exception as exc:
        logger.error("Failed to refine websearch query: %s", exc)
        await status_message.edit_text("Failed to refine the query. Please try again later.")
        return

    try:
        await status_message.edit_text(f"Searching the web with query: {refined_query}")
    except Exception:
        pass

    try:
        search_results = await perform_web_search(refined_query, WEBSEARCH_MAX_RESULTS)
    except Exception as exc:
        logger.error("Web search execution failed: %s", exc)
        await status_message.edit_text(str(exc))
        return

    if not search_results:
        await status_message.edit_text("No web results were found for the refined query.")
        return

    try:
        await status_message.edit_text("Generating report...")
    except Exception:
        pass

    try:
        report_text, thinking_text, elapsed = await summarize_search_results(
            refined_query,
            search_results,
            model,
            persona_prompt,
        )
    except Exception as exc:
        logger.error("Failed to summarise web search results: %s", exc)
        await status_message.edit_text("Failed to generate a summary. Please try again later.")
        return

    try:
        await status_message.delete()
    except Exception:
        pass

    generation_time = elapsed
    if generation_time < 1:
        time_str = f"{generation_time*1000:.0f}ms"
    else:
        time_str = f"{generation_time:.2f}s"

    show_thoughts = state.get('show_thoughts', False)
    display_thinking = thinking_text if show_thoughts else None

    conversation_logger.info(
        f"[{chat_label}] [User {username}({user_id})] /websearch refined='{refined_query}' results={len(search_results)}"
    )
    conversation_logger.info(
        f"[{chat_label}] [User {username}({user_id})] /websearch report: {report_text}"
    )
    if display_thinking:
        conversation_logger.info(
            f"[{chat_label}] [User {username}({user_id})] /websearch thoughts: {display_thinking}"
        )

    if STORAGE_AVAILABLE and chat_id is not None:
        timestamp_now = time.time()
        try:
            await storage.save_user_message(
                chat_id,
                chat.type if chat and chat.type else "unknown",
                timestamp_now,
                user_id,
                username,
                f"/websearch {trimmed_query}",
            )
            await storage.save_assistant_message(
                chat_id,
                chat.type if chat and chat.type else "unknown",
                timestamp_now,
                report_text,
            )
        except Exception as exc:
            logger.warning("Failed to persist websearch conversation for chat %s: %s", chat_id, exc)

    await send_ai_response(update, report_text, time_str, display_thinking)


@whitelist_required
async def websearch_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /websearch command to perform an external lookup."""
    if not update.message:
        return

    if not ENABLE_WEBSEARCH:
        await update.message.reply_text(
            "Web search is disabled. Set ENABLE_WEBSEARCH=1 in your environment to enable it."
        )
        return

    parts = update.message.text.split(maxsplit=1)
    state = context.chat_data
    # If no query is provided, switch into prompt mode and wait for next user message.
    if len(parts) < 2 or not parts[1].strip():
        state['awaiting_websearch_query'] = True
        keyboard = [[InlineKeyboardButton('Cancel', callback_data='websearch_cancel')]]
        await update.message.reply_text(
            "Please enter your web search query:",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )
        return

    await _process_websearch_request(update, context, parts[1])


@whitelist_required
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理按钮回调"""
    query = update.callback_query
    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"
    state = context.chat_data

    await query.answer()

    if query.data == 'websearch_cancel':
        state.pop('awaiting_websearch_query', None)
        try:
            await query.edit_message_text('Web search cancelled.')
        except Exception:
            pass
        return

    if query.data.startswith('model_'):
        model_name = query.data[6:]  # 移除 'model_' 前缀
        if model_name in AVAILABLE_MODELS:
            old_model = state.get('model', DEFAULT_MODEL)
            state['model'] = model_name
            new_model = AVAILABLE_MODELS[model_name]

            logger.info(f"用户 {username}({user_id}) 切换模型: {old_model} -> {model_name}")
            conversation_logger.info(f"[用户 {username}({user_id})] 切换模型: {old_model} -> {new_model}")

            await query.edit_message_text(f'已切换到模型：{new_model}')
        else:
            logger.warning(f"用户 {username}({user_id}) 尝试选择无效模型: {model_name}")
            await query.edit_message_text('无效的模型选择')

    elif query.data.startswith('recap_'):
        span_key = query.data[6:]
        selection = resolve_recap_timespan(span_key)
        if not selection:
            logger.warning(f"User {username}({user_id}) selected invalid recap span: {span_key}")
            await query.edit_message_text('Invalid recap range. Please try again.')
            return

        timespan_seconds, timespan_label = selection
        await query.edit_message_text(f'Recap range selected: {timespan_label}')
        await _execute_recap(context, query.message, user_id, username, timespan_seconds, timespan_label)
        return

    elif query.data.startswith('persona_'):
        persona_name = query.data[8:]
        if persona_name in PERSONAS:
            old_persona = state.get('persona', DEFAULT_PERSONA_KEY)
            state['persona'] = persona_name
            state['conversation_history'] = []

            logger.info(f"User {username}({user_id}) switched persona: {old_persona} -> {persona_name}")
            conversation_logger.info(
                f"[User {username}({user_id})] switched persona: {old_persona} -> {persona_name}"
            )

            await query.edit_message_text(
                f'Persona updated to {persona_name}. Conversation history cleared.'
            )
        else:
            logger.warning(f"User {username}({user_id}) tried invalid persona: {persona_name}")
            await query.edit_message_text('Invalid persona selection')


async def query_ollama(
    prompt: str,
    model: str,
    context_history: list | None = None,
    persona_prompt: str = "",
    speaker_label: str | None = None,
) -> tuple[str, str | None, float]:
    """Call the Ollama API and return the generated text, reasoning text, and elapsed time."""
    start_time = time.time()

    history_count = len(context_history) if context_history else 0
    persona_label = "default"
    if persona_prompt:
        first_line = persona_prompt.strip().splitlines()[0]
        persona_label = first_line[:60]

    sections: list[str] = []

    if persona_prompt:
        sections.append("System instructions:\n" + persona_prompt.strip())

    if history_count:
        context_lines: list[str] = []
        for item in context_history[-10:]:
            user_line = item.get("user", "")
            assistant_line = item.get("assistant", "")
            context_lines.append(f"User: {user_line}")
            context_lines.append(f"Assistant: {assistant_line}")
        sections.append("Conversation history:\n" + "\n".join(context_lines))

    current_prompt = speaker_label or prompt
    sections.append(f"User: {current_prompt}\nAssistant:")
    full_prompt = "\n\n".join(sections)

    logger.info(
        "Calling Ollama - model: %s, history: %s, persona: %s",
        model,
        history_count,
        persona_label,
    )

    logger.info(
        "Full prompt:\n %s",
        full_prompt,
    )


    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                OLLAMA_API_URL,
                json={
                    "model": model,
                    "prompt": full_prompt,
                    "stream": False,
                },
            ) as response:
                if response.status == 200:
                    result: dict[str, Any] = await response.json()
                    logger.debug("Ollama response: %s", result)
                    end_time = time.time()
                    generation_time = end_time - start_time
                    response_text = result.get("response") or "The model returned no content."
                    thinking_text = extract_reasoning_text(result) or None
                    if thinking_text:
                        logger.debug("Ollama reasoning captured (%s chars)", len(thinking_text))
                    logger.info(
                        "Ollama response ready - model: %s, time: %.2fs, length: %s",
                        model,
                        generation_time,
                        len(response_text),
                    )
                    return response_text, thinking_text, generation_time

                end_time = time.time()
                generation_time = end_time - start_time
                error_msg = f"Ollama returned status {response.status}"
                logger.error(
                    "Ollama error - model: %s, status: %s, time: %.2fs",
                    model,
                    response.status,
                    generation_time,
                )
                return error_msg, None, generation_time
        except Exception as exc:
            logger.error("Error calling Ollama: %s", exc)
            end_time = time.time()
            generation_time = end_time - start_time
            error_msg = "Unable to reach the Ollama service right now."
            logger.error(
                "Ollama exception - model: %s, error: %s, time: %.2fs",
                model,
                exc,
                generation_time,
            )
            return error_msg, None, generation_time

@whitelist_required
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理用户消息"""
    if not update.message or not update.message.text:
        if update.message:
            await update.message.reply_text("抱歉，我只处理文本消息。请发送文字内容。")
        return

    user_message = update.message.text
    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"
    state = context.chat_data

    if state.get('awaiting_websearch_query'):
        await _process_websearch_request(update, context, user_message)
        return

    if not state.get('model'):
        state['model'] = DEFAULT_MODEL

    if not state.get('conversation_history'):
        state['conversation_history'] = []

    if PERSONAS:
        if not state.get('persona'):
            state['persona'] = DEFAULT_PERSONA_KEY
        current_persona = state.get('persona')
        persona_prompt = PERSONAS.get(current_persona, '')
    else:
        current_persona = None
        persona_prompt = ''

    state.setdefault('show_thoughts', False)
    show_thoughts = state['show_thoughts']

    current_model_key = state['model']
    current_model = AVAILABLE_MODELS[current_model_key]
    conversation_history = state['conversation_history']

    chat = update.effective_chat
    chat_id = chat.id if chat else None
    chat_title = chat.title if chat and chat.title else None
    if chat_title:
        chat_label = f"{chat_title}({chat_id})"
    elif chat_id is not None:
        chat_label = f"Chat {chat_id}"
    else:
        chat_label = "Chat unknown"

    user_timestamp = time.time()
    if update.message and update.message.date:
        user_timestamp = update.message.date.timestamp()

    sender_reference = None
    if update.effective_user:
        sender_reference = update.effective_user.username or update.effective_user.full_name

    if STORAGE_AVAILABLE and chat_id is not None:
        try:
            await storage.save_user_message(
                chat_id,
                chat.type if chat and chat.type else "unknown",
                user_timestamp,
                user_id,
                sender_reference,
                user_message,
            )
        except Exception as exc:
            logger.warning("Failed to persist user message for chat %s: %s", chat_id, exc)

    is_group_chat = bool(chat and chat.type in {'group', 'supergroup'})
    speaker_name = update.effective_user.username or update.effective_user.full_name or "Unknown"
    speaker_label = f"@{speaker_name} ({user_id})"
    user_entry_text = f"{speaker_label}: {user_message}" if is_group_chat else user_message

    conversation_round = len(conversation_history) + 1

    logger.info(f"用户 {username}({user_id}) 在 {chat_label} 发送消息 - 第{conversation_round}轮对话")
    conversation_logger.info(f"[{chat_label}] [用户 {username}({user_id})] 第{conversation_round}轮 - 用户: {user_entry_text}")

    persona_line = f"\nPersona: {current_persona}" if current_persona else ""
    thinking_message = await update.message.reply_text(
        f"Thinking...\nUsing model: {current_model}{persona_line}"
    )

    try:
        ai_response, thinking_text, generation_time = await query_ollama(
            user_message,
            current_model,
            conversation_history,
            persona_prompt,
            user_entry_text,
        )

        await thinking_message.delete()

        logger.info(f"AI回复生成完成 - 用户: {username}({user_id}), 模型: {current_model}, 时间: {generation_time:.2f}s")
        conversation_logger.info(f"[{chat_label}] [用户 {username}({user_id})] 第{conversation_round}轮 - AI回复: {ai_response}")
        if thinking_text:
            conversation_logger.info(f"[{chat_label}] [用户 {username}({user_id})] 第{conversation_round}轮 - AI思考: {thinking_text}")

        response_timestamp = time.time()

        conversation_history.append({
            'user': user_entry_text,
            'assistant': ai_response,
            'timestamp': response_timestamp,
        })

        if STORAGE_AVAILABLE and chat_id is not None:
            try:
                await storage.save_assistant_message(
                    chat_id,
                    chat.type if chat and chat.type else "unknown",
                    response_timestamp,
                    ai_response,
                )
            except Exception as exc:
                logger.warning("Failed to persist assistant message for chat %s: %s", chat_id, exc)

        if len(conversation_history) > 20:
            del conversation_history[:-20]
            logger.info(f"{chat_label} 的对话历史已限制为20轮")

        if generation_time < 1:
            time_str = f"{generation_time*1000:.0f}ms"
        else:
            time_str = f"{generation_time:.2f}s"

        display_thinking = thinking_text if show_thoughts else None
        await send_ai_response(update, ai_response, time_str, display_thinking)

        conversation_logger.info(f"[{chat_label}] [用户 {username}({user_id})] 第{conversation_round}轮对话完成 - 历史记录长度: {len(conversation_history)}")

    except Exception as e:
        logger.error(f"处理消息时出错: {str(e)}")
        conversation_logger.error(f"[{chat_label}] [用户 {username}({user_id})] 第{conversation_round}轮对话出错: {str(e)}")
        await thinking_message.edit_text("抱歉，处理您的消息时出现错误。请稍后重试。")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """错误处理器"""
    logger.error(f"处理更新时出错: {context.error}")
    if update and update.effective_message:
        try:
            await update.effective_message.reply_text("抱歉，处理您的消息时出现错误。请稍后重试。")
        except Exception as e:
            logger.error(f"发送错误消息失败: {str(e)}")

def main() -> None:
    """主函数"""
    if not TELEGRAM_BOT_TOKEN:
        logger.error("未设置 TELEGRAM_BOT_TOKEN 环境变量")
        return

    # 创建应用
    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(on_startup)
        .build()
    )

    # 添加错误处理器
    application.add_error_handler(error_handler)

    # 添加处理器
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("thoughts", thoughts_command))
    application.add_handler(CommandHandler("model", model_command))
    application.add_handler(CommandHandler("persona", persona_command))
    application.add_handler(CommandHandler("forget", forget_command))
    application.add_handler(CommandHandler("recap", recap_command))
    application.add_handler(CommandHandler("websearch", websearch_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # 启动机器人
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == '__main__':
    main() 
