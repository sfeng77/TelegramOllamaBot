import os
import json
import logging
import time
import aiohttp
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler

# 配置日志
import logging.handlers

# 创建日志目录
import os
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 配置根日志器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 控制台输出
        logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, 'bot.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
    ]
)

# 创建专门的对话日志器
conversation_logger = logging.getLogger('conversation')
conversation_handler = logging.handlers.RotatingFileHandler(
    os.path.join(log_dir, 'conversations.log'),
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
conversation_formatter = logging.Formatter('%(asctime)s - %(message)s')
conversation_handler.setFormatter(conversation_formatter)
conversation_logger.addHandler(conversation_handler)
conversation_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
OLLAMA_API_URL = "http://localhost:11434/api/generate"

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

    logger.info(f"用户 {username}({user_id}) 执行/start命令")
    conversation_logger.info(f"[用户 {username}({user_id})] 执行/start命令 - 初始化对话")

    welcome_message = """
欢迎使用 AI 聊天机器人！

这个机器人使用 Ollama 来回答您的问题。
当前使用的模型是：{}

命令列表：
/start - 显示此帮助信息
/help - 获取帮助
/model - 选择 AI 模型
/forget - 忘记之前的对话内容
""".format(AVAILABLE_MODELS[context.user_data['model']])
    await update.message.reply_text(welcome_message)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理 /help 命令"""
    help_text = """
使用说明：

1. 直接发送文字消息即可与 AI 对话
2. 机器人会记住对话上下文，提供更连贯的对话体验
3. 使用 /model 命令可以切换不同的 AI 模型
4. 使用 /forget 命令可以清除对话历史，重新开始
5. 当前支持的模型：
{}
6. 如果遇到问题，请尝试重新发送消息
7. 如需重新开始对话，请使用 /start 命令
""".format('\n'.join(f'   - {name}: {model}' for name, model in AVAILABLE_MODELS.items()))
    await update.message.reply_text(help_text)

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

async def query_ollama(prompt: str, model: str, context_history: list = None) -> tuple[str, float]:
    """向 Ollama API 发送请求，返回回复和生成时间"""
    start_time = time.time()
    
    # 构建包含上下文的完整提示
    if context_history and len(context_history) > 0:
        # 将历史对话转换为上下文格式
        context_text = "\n".join([f"用户: {item['user']}\n助手: {item['assistant']}" for item in context_history[-10:]])  # 只保留最近10轮对话
        full_prompt = f"以下是之前的对话历史：\n{context_text}\n\n现在用户说：{prompt}\n请根据上下文回答："
        logger.info(f"发送API请求到Ollama - 模型: {model}, 包含{len(context_history)}轮历史对话")
    else:
        full_prompt = prompt
        logger.info(f"发送API请求到Ollama - 模型: {model}, 无历史对话")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                OLLAMA_API_URL,
                json={
                    "model": model,
                    "prompt": full_prompt,
                    "stream": False
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    end_time = time.time()
                    generation_time = end_time - start_time
                    response_text = result.get('response', '抱歉，我现在无法回答这个问题。')
                    logger.info(f"Ollama API响应成功 - 模型: {model}, 响应时间: {generation_time:.2f}s, 响应长度: {len(response_text)}字符")
                    return response_text, generation_time
                else:
                    end_time = time.time()
                    generation_time = end_time - start_time
                    error_msg = f"API 请求失败，状态码：{response.status}"
                    logger.error(f"Ollama API请求失败 - 模型: {model}, 状态码: {response.status}, 响应时间: {generation_time:.2f}s")
                    return error_msg, generation_time
        except Exception as e:
            logger.error(f"请求 Ollama API 时出错: {str(e)}")
            end_time = time.time()
            generation_time = end_time - start_time
            error_msg = "抱歉，与 AI 模型通信时出现错误。请确保 Ollama 服务正在运行。"
            logger.error(f"Ollama API异常 - 模型: {model}, 错误: {str(e)}, 响应时间: {generation_time:.2f}s")
            return error_msg, generation_time

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理用户消息"""
    user_message = update.message.text
    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"
    
    # 确保用户有选择的模型，如果没有则使用默认模型
    if not context.user_data.get('model'):
        context.user_data['model'] = DEFAULT_MODEL
    
    # 确保用户有对话历史，如果没有则初始化
    if not context.user_data.get('conversation_history'):
        context.user_data['conversation_history'] = []
    
    current_model = AVAILABLE_MODELS[context.user_data['model']]
    conversation_history = context.user_data['conversation_history']
    conversation_round = len(conversation_history) + 1
    
    # 记录用户消息
    logger.info(f"用户 {username}({user_id}) 发送消息 - 第{conversation_round}轮对话")
    conversation_logger.info(f"[用户 {username}({user_id})] 第{conversation_round}轮 - 用户: {user_message}")
    
    # 发送"正在思考"消息
    thinking_message = await update.message.reply_text(
        f"正在思考...\n使用模型：{current_model}"
    )
    
    try:
        # 获取 AI 回复和生成时间，传递对话历史
        ai_response, generation_time = await query_ollama(user_message, current_model, conversation_history)
        
        # 删除"正在思考"消息
        await thinking_message.delete()
        
        # 记录AI回复
        logger.info(f"AI回复生成完成 - 用户: {username}({user_id}), 模型: {current_model}, 时间: {generation_time:.2f}s")
        conversation_logger.info(f"[用户 {username}({user_id})] 第{conversation_round}轮 - AI回复: {ai_response}")
        
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
        
        # 在回复中添加生成时间信息
        final_response = f"{ai_response}\n\n⏱️ 生成时间：{time_str}"
        
        # 发送 AI 回复
        await update.message.reply_text(final_response)
        
        # 记录对话完成
        conversation_logger.info(f"[用户 {username}({user_id})] 第{conversation_round}轮对话完成 - 历史记录长度: {len(conversation_history)}")
        
    except Exception as e:
        logger.error(f"处理消息时出错: {str(e)}")
        conversation_logger.error(f"[用户 {username}({user_id})] 第{conversation_round}轮对话出错: {str(e)}")
        await thinking_message.edit_text("抱歉，处理您的消息时出现错误。请稍后重试。")

def main() -> None:
    """主函数"""
    if not TELEGRAM_BOT_TOKEN:
        logger.error("未设置 TELEGRAM_BOT_TOKEN 环境变量")
        return

    # 创建应用
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # 添加处理器
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("model", model_command))
    application.add_handler(CommandHandler("forget", forget_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # 启动机器人
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main() 