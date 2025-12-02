' 静默运行Python脚本
Set ws = CreateObject("Wscript.Shell")
ws.Run "pythonw "".\create.py""", 0, False
