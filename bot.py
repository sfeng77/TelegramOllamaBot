import os
import json
import logging
import aiohttp
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

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

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理 /start 命令"""
    welcome_message = """
欢迎使用 AI 聊天机器人！

这个机器人使用 deepseek-r1:32b 模型来回答您的问题。
直接发送消息即可开始对话。

命令列表：
/start - 显示此帮助信息
/help - 获取帮助
"""
    await update.message.reply_text(welcome_message)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理 /help 命令"""
    help_text = """
使用说明：

1. 直接发送文字消息即可与 AI 对话
2. AI 会记住对话上下文
3. 如果遇到问题，请尝试重新发送消息
4. 如需重新开始对话，请使用 /start 命令
"""
    await update.message.reply_text(help_text)

async def query_ollama(prompt: str) -> str:
    """向 Ollama API 发送请求"""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                OLLAMA_API_URL,
                json={
                    "model": "deepseek-r1:32b",
                    "prompt": prompt,
                    "stream": False
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('response', '抱歉，我现在无法回答这个问题。')
                else:
                    return f"API 请求失败，状态码：{response.status}"
        except Exception as e:
            logger.error(f"请求 Ollama API 时出错: {str(e)}")
            return "抱歉，与 AI 模型通信时出现错误。请确保 Ollama 服务正在运行。"

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理用户消息"""
    user_message = update.message.text
    
    # 发送"正在思考"消息
    thinking_message = await update.message.reply_text("正在思考...")
    
    try:
        # 获取 AI 回复
        ai_response = await query_ollama(user_message)
        
        # 删除"正在思考"消息
        await thinking_message.delete()
        
        # 发送 AI 回复
        await update.message.reply_text(ai_response)
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
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # 启动机器人
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main() 