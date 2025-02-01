# Telegram Ollama Bot

这是一个基于 Telegram 的聊天机器人，使用 Ollama 的 deepseek-r1:32b 模型来进行对话。

## 环境要求

- Python 3.8+
- Ollama（需要预先安装并运行）
- Telegram Bot Token（需要从 @BotFather 获取）

## 安装步骤

1. 克隆此仓库
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 确保 Ollama 已安装并运行，且已下载 deepseek-r1:32b 模型：
   ```bash
   ollama pull deepseek-r1:32b
   ```
4. 创建 `.env` 文件并添加你的 Telegram Bot Token：
   ```
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   ```
5. 运行机器人：
   ```bash
   python bot.py
   ```

## 使用方法

1. 在 Telegram 中找到你的机器人并开始对话
2. 直接发送消息即可与 AI 进行对话
3. 使用 /start 命令查看使用说明
4. 使用 /help 命令获取帮助 