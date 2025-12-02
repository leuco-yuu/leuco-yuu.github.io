@echo off
:: 检查管理员权限
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"
if %errorlevel% neq 0 (
    echo 请求管理员权限...
    :: 重新以管理员身份运行
    powershell -Command "Start-Process '%~s0' -Verb RunAs"
    exit /b
)

:: 设置变量
set REPO_PATH=D:\Blog\leuco-blog\
set BRANCH=main

:: 获取提交信息
set /p COMMIT_MSG="请输入提交信息: "
if "%COMMIT_MSG%"=="" (
    echo 提交信息不能为空！
    pause
    exit /b
)

:: 执行Git操作
echo 正在执行Git操作...
echo.

:: 切换到仓库目录
cd /D "%REPO_PATH%"
if %errorlevel% neq 0 (
    echo 错误：无法切换到目录 %REPO_PATH%
    pause
    exit /b
)

:: 检查是否是Git仓库
git status >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误：%REPO_PATH% 不是Git仓库或Git未安装
    pause
    exit /b
)

:: 执行Git命令
echo 1. 添加所有更改...
git add .
if %errorlevel% neq 0 (
    echo 错误：git add 失败
    pause
    exit /b
)

echo 2. 提交更改...
git commit -m "%COMMIT_MSG%"
if %errorlevel% neq 0 (
    echo 注意：提交可能失败（没有更改或已有未提交的更改）
    echo 将继续尝试推送...
)

echo 3. 推送到远程仓库...
git push origin %BRANCH%
if %errorlevel% neq 0 (
    echo 错误：推送失败！
    echo 请检查网络连接或远程仓库权限
    pause
    exit /b
)

echo.
echo ✓ Git操作完成！
echo 提交信息: "%COMMIT_MSG%"
echo 仓库位置: "%REPO_PATH%"
echo 分支: %BRANCH%

pause