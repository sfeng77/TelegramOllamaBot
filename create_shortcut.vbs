Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' 获取脚本路径
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
targetVBS = scriptDir & "\run_bot_hidden.vbs"

' 获取启动文件夹路径
startupFolder = WshShell.SpecialFolders("Startup")
shortcutPath = startupFolder & "\TelegramOllamaBot.lnk"

' 创建快捷方式
Set shortcut = WshShell.CreateShortcut(shortcutPath)
shortcut.TargetPath = targetVBS
shortcut.WorkingDirectory = scriptDir
shortcut.WindowStyle = 7  ' 最小化窗口

' 设置管理员权限
With CreateObject("Shell.Application")
    .ShellExecute "cmd.exe", "/c echo Set oShell = CreateObject(""Shell.Application"") > """ & scriptDir & "\temp.vbs""", "", "runas", 0
    .ShellExecute "cmd.exe", "/c echo oShell.ShellExecute """ & targetVBS & """, """", """", ""runas"", 1 >> """ & scriptDir & "\temp.vbs""", "", "runas", 0
End With

shortcut.Save

WScript.Echo "快捷方式已创建：" & shortcutPath

' 清理对象
Set shortcut = Nothing
Set WshShell = Nothing
Set fso = Nothing 