@echo off
chcp 65001 >nul

:: 请求管理员权限
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"
if %errorlevel% neq 0 (
    powershell -Command "Start-Process '%~s0' -Verb RunAs"
    exit /b
)

cd /D "D:\Blog\leuco-blog\"
set /p msg="输入提交信息: "

echo 正在执行Git操作...
git add .
git commit -m "%msg%"
git push -f origin main

echo 完成！
pause