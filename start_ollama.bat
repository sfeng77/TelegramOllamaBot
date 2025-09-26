@echo off
echo 正在启动 Ollama 服务...

:: 检查 Ollama 是否已经在运行
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo Ollama 服务已在运行
) else (
    echo 启动 Ollama 服务...
    start "" "C:\Program Files\Ollama\ollama.exe" serve
    :: 等待服务启动
    timeout /t 5
)

:: 检查模型是否已下载
echo 检查 deepseek-r1:1.5b 模型...
ollama list | find "deepseek-r1:1.5b" >NUL
if errorlevel 1 (
    echo 正在下载 deepseek-r1:1.5b 模型...
    ollama pull deepseek-r1:1.5b
) else (
    echo deepseek-r1:1.5b 模型已存在
) 