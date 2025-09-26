import os
import json
import logging
import time
import aiohttp
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
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
DEFAULT_MODEL = 'deepseek_large'

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理 /start 命令"""
    # 初始化用户设置
    if not context.user_data.get('model'):
        context.user_data['model'] = DEFAULT_MODEL

    welcome_message = """
欢迎使用 AI 聊天机器人！

这个机器人使用 Ollama 来回答您的问题。
当前使用的模型是：{}

命令列表：
/start - 显示此帮助信息
/help - 获取帮助
/model - 选择 AI 模型
""".format(AVAILABLE_MODELS[context.user_data['model']])
    await update.message.reply_text(welcome_message)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理 /help 命令"""
    help_text = """
使用说明：

1. 直接发送文字消息即可与 AI 对话
2. 使用 /model 命令可以切换不同的 AI 模型
3. 当前支持的模型：
{}
4. 如果遇到问题，请尝试重新发送消息
5. 如需重新开始对话，请使用 /start 命令
""".format('\n'.join(f'   - {name}: {model}' for name, model in AVAILABLE_MODELS.items()))
    await update.message.reply_text(help_text)

async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理 /model 命令"""
    keyboard = []
    # 创建模型选择按钮
    for name, model in AVAILABLE_MODELS.items():
        # 在当前选中的模型旁边添加标记
        current = '✓ ' if context.user_data.get('model') == name else ''
        keyboard.append([InlineKeyboardButton(f"{current}{model}", callback_data=f"model_{name}")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text('请选择要使用的 AI 模型：', reply_markup=reply_markup)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理按钮回调"""
    query = update.callback_query
    await query.answer()

    if query.data.startswith('model_'):
        model_name = query.data[6:]  # 移除 'model_' 前缀
        if model_name in AVAILABLE_MODELS:
            context.user_data['model'] = model_name
            await query.edit_message_text(f'已切换到模型：{AVAILABLE_MODELS[model_name]}')
        else:
            await query.edit_message_text('无效的模型选择')

async def query_ollama(prompt: str, model: str) -> tuple[str, float]:
    """向 Ollama API 发送请求，返回回复和生成时间"""
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                OLLAMA_API_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    end_time = time.time()
                    generation_time = end_time - start_time
                    response_text = result.get('response', '抱歉，我现在无法回答这个问题。')
                    return response_text, generation_time
                else:
                    end_time = time.time()
                    generation_time = end_time - start_time
                    return f"API 请求失败，状态码：{response.status}", generation_time
        except Exception as e:
            logger.error(f"请求 Ollama API 时出错: {str(e)}")
            end_time = time.time()
            generation_time = end_time - start_time
            return "抱歉，与 AI 模型通信时出现错误。请确保 Ollama 服务正在运行。", generation_time

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理用户消息"""
    user_message = update.message.text
    
    # 确保用户有选择的模型，如果没有则使用默认模型
    if not context.user_data.get('model'):
        context.user_data['model'] = DEFAULT_MODEL
    
    current_model = AVAILABLE_MODELS[context.user_data['model']]
    
    # 发送"正在思考"消息
    thinking_message = await update.message.reply_text(
        f"正在思考...\n使用模型：{current_model}"
    )
    
    try:
        # 获取 AI 回复和生成时间
        ai_response, generation_time = await query_ollama(user_message, current_model)
        
        # 删除"正在思考"消息
        await thinking_message.delete()
        
        # 格式化生成时间
        if generation_time < 1:
            time_str = f"{generation_time*1000:.0f}ms"
        else:
            time_str = f"{generation_time:.2f}s"
        
        # 在回复中添加生成时间信息
        final_response = f"{ai_response}\n\n⏱️ 生成时间：{time_str}"
        
        # 发送 AI 回复
        await update.message.reply_text(final_response)
    except Exception as e:
        logger.error(f"处理消息时出错: {str(e)}")
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
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # 启动机器人
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main() 