' launch.vbs — Windows launcher for SETS-WARP (no console window)
' Used by the desktop / Start Menu shortcut created by the installer.
' Double-clicking sets_warp.bat directly also works but shows a cmd window.

Set fso   = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = scriptDir

' 0 = hidden window, False = don't wait for process to finish
shell.Run "cmd /c sets_warp.bat", 0, False
