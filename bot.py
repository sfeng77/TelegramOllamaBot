import asyncio
import logging
import logging.handlers
import os
import time
from functools import wraps
from pathlib import Path
from typing import Any, Awaitable, Callable

import aiohttp
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

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

# Load environment variables
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
OLLAMA_API_URL = "http://localhost:11434/api/generate"

ALLOWED_USERNAMES_ENV = os.getenv('ALLOWED_TELEGRAM_USERNAMES', '')
ALLOWED_USERNAMES: set[str] = set()
if ALLOWED_USERNAMES_ENV:
    for raw_value in ALLOWED_USERNAMES_ENV.split(','):
        trimmed_value = raw_value.strip().lstrip('@')
        if not trimmed_value:
            continue
        ALLOWED_USERNAMES.add(trimmed_value.lower())
if not ALLOWED_USERNAMES:
    logger.warning("ALLOWED_TELEGRAM_USERNAMES is empty; bot will deny all users until configured.")

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

    username = (user.username or '').strip()
    user_identifier = f"{username or 'Unknown'}({user.id})"
    if not username:
        logger.warning('User %s has no username; denying access in username whitelist mode.', user_identifier)
        await _deny_access(update)
        return False

    if username.lower() in ALLOWED_USERNAMES:
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
    
    # 初始化用户设置
    if not context.user_data.get('model'):
        context.user_data['model'] = DEFAULT_MODEL
    
    # 初始化对话历史
    if not context.user_data.get('conversation_history'):
        context.user_data['conversation_history'] = []

    if PERSONAS and not context.user_data.get('persona'):
        context.user_data['persona'] = DEFAULT_PERSONA_KEY

    logger.info(f"用户 {username}({user_id}) 执行/start命令")
    conversation_logger.info(f"[用户 {username}({user_id})] 执行/start命令 - 初始化对话")

    welcome_message = """
欢迎使用 AI 聊天机器人！

这个机器人使用 Ollama 来回答您的问题。
当前使用的模型是：{}
当前的助手角色是：{}
/help 命令可查看使用说明。
""".format(AVAILABLE_MODELS[context.user_data['model']], context.user_data['persona'] if PERSONAS else "null")
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
9. 如需重新开始对话，请使用 /start 命令
""".format('\n'.join(f'   - {name}: {model}' for name, model in AVAILABLE_MODELS.items()))
    await send_long_message(update, help_text)
    await update.message.reply_text("Use /thoughts [on|off] or run without arguments to toggle the thought process display.")

    if PERSONAS:
        await update.message.reply_text("Use /persona to switch the assistant persona.")

@whitelist_required
async def thoughts_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /thoughts command to toggle reasoning visibility."""
    if not update.message:
        return

    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"
    context.user_data.setdefault('show_thoughts', False)
    current_state = context.user_data['show_thoughts']

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

    context.user_data['show_thoughts'] = new_state
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
    current_model = context.user_data.get('model', DEFAULT_MODEL)
    
    logger.info(f"用户 {username}({user_id}) 执行/model命令 - 当前模型: {current_model}")
    
    keyboard = []
    # 创建模型选择按钮
    for name, model in AVAILABLE_MODELS.items():
        # 在当前选中的模型旁边添加标记
        current = '✓ ' if context.user_data.get('model') == name else ''
        keyboard.append([InlineKeyboardButton(f"{current}{model}", callback_data=f"model_{name}")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text('请选择要使用的 AI 模型：', reply_markup=reply_markup)

@whitelist_required
async def persona_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /persona command."""
    if not PERSONAS:
        await update.message.reply_text("No personas are configured for this bot.")
        return

    if not context.user_data.get('persona'):
        context.user_data['persona'] = DEFAULT_PERSONA_KEY

    keyboard: list[list[InlineKeyboardButton]] = []
    current_persona = context.user_data.get('persona')
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
    
    # 记录清除前的历史长度
    history_length = len(context.user_data.get('conversation_history', []))
    
    # 清除对话历史
    context.user_data['conversation_history'] = []
    
    # 记录清除操作
    logger.info(f"用户 {username}({user_id}) 执行/forget命令 - 清除了{history_length}轮对话历史")
    conversation_logger.info(f"[用户 {username}({user_id})] 执行/forget命令 - 清除了{history_length}轮对话历史")
    
    await update.message.reply_text("🧹 已清除所有对话历史，我们可以重新开始对话了！")

@whitelist_required
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理按钮回调"""
    query = update.callback_query
    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"
    
    await query.answer()

    if query.data.startswith('model_'):
        model_name = query.data[6:]  # 移除 'model_' 前缀
        if model_name in AVAILABLE_MODELS:
            old_model = context.user_data.get('model', DEFAULT_MODEL)
            context.user_data['model'] = model_name
            new_model = AVAILABLE_MODELS[model_name]
            
            logger.info(f"用户 {username}({user_id}) 切换模型: {old_model} -> {model_name}")
            conversation_logger.info(f"[用户 {username}({user_id})] 切换模型: {old_model} -> {new_model}")
            
            await query.edit_message_text(f'已切换到模型：{new_model}')
        else:
            logger.warning(f"用户 {username}({user_id}) 尝试选择无效模型: {model_name}")
            await query.edit_message_text('无效的模型选择')

    elif query.data.startswith('persona_'):
        persona_name = query.data[8:]
        if persona_name in PERSONAS:
            old_persona = context.user_data.get('persona', DEFAULT_PERSONA_KEY)
            context.user_data['persona'] = persona_name
            context.user_data['conversation_history'] = []

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

    sections.append(f"User: {prompt}\nAssistant:")
    full_prompt = "\n\n".join(sections)

    logger.info(
        "Calling Ollama - model: %s, history: %s, persona: %s",
        model,
        history_count,
        persona_label,
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
    # 检查消息是否存在且为文本消息
    if not update.message or not update.message.text:
        if update.message:
            # 如果是非文本消息，回复提示
            await update.message.reply_text("抱歉，我只处理文本消息。请发送文字内容。")
        return
    
    user_message = update.message.text
    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"
    
    # 确保用户有选择的模型，如果没有则使用默认模型
    if not context.user_data.get('model'):
        context.user_data['model'] = DEFAULT_MODEL
    
    # 确保用户有对话历史，如果没有则初始化
    if not context.user_data.get('conversation_history'):
        context.user_data['conversation_history'] = []

    if PERSONAS:
        if not context.user_data.get('persona'):
            context.user_data['persona'] = DEFAULT_PERSONA_KEY
        current_persona = context.user_data['persona']
        persona_prompt = PERSONAS.get(current_persona, '')
    else:
        current_persona = None
        persona_prompt = ''

    context.user_data.setdefault('show_thoughts', False)
    show_thoughts = context.user_data['show_thoughts']

    current_model = AVAILABLE_MODELS[context.user_data['model']]
    conversation_history = context.user_data['conversation_history']
    conversation_round = len(conversation_history) + 1
    
    # 记录用户消息
    logger.info(f"用户 {username}({user_id}) 发送消息 - 第{conversation_round}轮对话")
    conversation_logger.info(f"[用户 {username}({user_id})] 第{conversation_round}轮 - 用户: {user_message}")
    
    # 发送"正在思考"消息
    persona_line = f"\nPersona: {current_persona}" if current_persona else ""
    thinking_message = await update.message.reply_text(
        f"Thinking...\nUsing model: {current_model}{persona_line}"
    )

    
    try:
        # 获取 AI 回复和生成时间，传递对话历史
        ai_response, thinking_text, generation_time = await query_ollama(user_message, current_model, conversation_history, persona_prompt)
        
        # 删除"正在思考"消息
        await thinking_message.delete()
        
        # 记录AI回复
        logger.info(f"AI回复生成完成 - 用户: {username}({user_id}), 模型: {current_model}, 时间: {generation_time:.2f}s")
        conversation_logger.info(f"[用户 {username}({user_id})] 第{conversation_round}轮 - AI回复: {ai_response}")
        if thinking_text:
            conversation_logger.info(f"[用户 {username}({user_id})] 第{conversation_round}轮 - AI思考: {thinking_text}")
        
        # 将当前对话添加到历史记录中
        conversation_history.append({
            'user': user_message,
            'assistant': ai_response
        })
        
        # 限制历史记录长度，避免上下文过长
        if len(conversation_history) > 20:  # 保留最近20轮对话
            conversation_history = conversation_history[-20:]
            context.user_data['conversation_history'] = conversation_history
            logger.info(f"用户 {username}({user_id}) 对话历史已限制为20轮")
        
        # 格式化生成时间
        if generation_time < 1:
            time_str = f"{generation_time*1000:.0f}ms"
        else:
            time_str = f"{generation_time:.2f}s"
        
        # 发送 AI 回复，支持长消息分割
        display_thinking = thinking_text if show_thoughts else None
        await send_ai_response(update, ai_response, time_str, display_thinking)
        
        # 记录对话完成
        conversation_logger.info(f"[用户 {username}({user_id})] 第{conversation_round}轮对话完成 - 历史记录长度: {len(conversation_history)}")
        
    except Exception as e:
        logger.error(f"处理消息时出错: {str(e)}")
        conversation_logger.error(f"[用户 {username}({user_id})] 第{conversation_round}轮对话出错: {str(e)}")
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
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # 添加错误处理器
    application.add_error_handler(error_handler)

    # 添加处理器
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("thoughts", thoughts_command))
    application.add_handler(CommandHandler("model", model_command))
    application.add_handler(CommandHandler("persona", persona_command))
    application.add_handler(CommandHandler("forget", forget_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # 启动机器人
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == '__main__':
    main() 
