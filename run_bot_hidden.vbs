Dim WshShell, fso, scriptDir
Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
WshShell.CurrentDirectory = scriptDir
WshShell.Run chr(34) & scriptDir & "\start_bot.bat" & chr(34), 0
Set WshShell = Nothing
Set fso = Nothing 