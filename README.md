# Telegram Ollama Bot

这是一个基于 Telegram 的聊天机器人，使用 Ollama 的多个模型来进行对话。支持在多个模型之间切换，包括 deepseek-r1:32b、deepseek-r1:1.5b、llama2 等。

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
3. 确保 Ollama 已安装并运行，且已下载所需模型：
   ```bash
   ollama pull deepseek-r1:32b
   ollama pull deepseek-r1:1.5b
   ollama pull llama2
   ollama pull mistral
   ollama pull neural-chat
   ```
4. 创建 `.env` 文件并添加你的 Telegram Bot Token：
   ```
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   ```
5. 设置机器人命令菜单（通过 @BotFather）：
   - 发送 `/mybots` 给 @BotFather
   - 选择你的机器人
   - 点击 "Edit Bot"
   - 选择 "Edit Commands"
   - 复制并粘贴 `commands.txt` 中的内容

6. 运行机器人：
   ```bash
   python bot.py
   ```

## 使用方法

1. 在 Telegram 中找到你的机器人并开始对话
2. 使用命令：
   - `/start` - 开始使用机器人并查看欢迎信息
   - `/help` - 获取帮助和使用说明
   - `/model` - 选择要使用的 AI 模型
3. 直接发送消息即可与 AI 对话

## 支持的模型

- deepseek-r1:32b - 大型模型，理解能力强
- deepseek-r1:1.5b - 小型模型，响应速度快
- llama2 - 通用型模型
- mistral - 新一代开源模型
- neural-chat - 对话风格自然

## 注意事项

1. 确保 Ollama 服务正在运行
2. 首次使用新模型时需要下载，可能需要一些时间
3. 不同模型有不同的特点，可以根据需求选择 