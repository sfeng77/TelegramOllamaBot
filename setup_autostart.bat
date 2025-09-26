@echo off
echo 正在设置开机自动启动...

:: 获取当前脚本的完整路径
set "SCRIPT_PATH=%~dp0"
set "VBS_PATH=%SCRIPT_PATH%run_bot_hidden.vbs"
set "CREATE_SHORTCUT_VBS=%SCRIPT_PATH%create_shortcut.vbs"

:: 检查必要文件是否存在
if not exist "%VBS_PATH%" (
    echo 错误：run_bot_hidden.vbs 未找到！
    pause
    exit /b 1
)

:: 使用 VBScript 创建快捷方式
cscript //nologo "%CREATE_SHORTCUT_VBS%"
if errorlevel 1 (
    echo 错误：无法创建快捷方式！
    pause
    exit /b 1
)

echo 设置完成！机器人将在下次开机时自动启动。

:: 询问是否立即启动机器人
set /p "START_NOW=是否要立即启动机器人？(Y/N) "
if /i "%START_NOW%"=="Y" (
    start "" "%VBS_PATH%"
)

pause 