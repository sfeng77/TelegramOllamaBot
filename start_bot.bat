@echo off
cd /d "%~dp0"

:: 启动 Ollama 服务
call start_ollama.bat
if errorlevel 1 (
    echo 错误：无法启动 Ollama 服务！
    pause
    exit /b 1
)

:: 检查虚拟环境是否存在
if not exist "venv\Scripts\activate.bat" (
    echo 错误：虚拟环境未找到！
    echo 请确保已经安装了所有依赖。
    pause
    exit /b 1
)

:: 检查 Python 文件是否存在
if not exist "bot.py" (
    echo 错误：bot.py 未找到！
    pause
    exit /b 1
)

:: 激活虚拟环境并运行机器人
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo 错误：无法激活虚拟环境！
    pause
    exit /b 1
)

:: 等待 Ollama 服务完全启动
timeout /t 5

:: 启动机器人
python bot.py
if errorlevel 1 (
    echo 错误：机器人启动失败！
    pause
    exit /b 1
) 