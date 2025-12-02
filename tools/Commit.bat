@echo off
chcp 65001 >nul
:: 设置UTF-8编码支持中文

:: ==== 自动请求管理员权限 ====
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"
if %errorlevel% neq 0 (
    echo 正在请求管理员权限...
    echo 请在弹出的UAC窗口中点击"是"
    echo.
    timeout /t 2 /nobreak >nul
    powershell -Command "Start-Process '%~s0' -Verb RunAs"
    exit /b
)

:: ==== 主程序开始 ====
cls
echo ========================================
echo     Git自动提交工具 (管理员模式)
echo ========================================
echo 当前用户: %USERNAME%
echo 权限级别: 管理员
echo ========================================
echo.

:: 设置变量
set REPO_PATH=D:\Blog\leuco-blog\
set BRANCH=main

echo 仓库路径: %REPO_PATH%
echo 目标分支: %BRANCH%
echo.

:: 获取提交信息
:input_commit
set /p COMMIT_MSG="请输入提交信息: "
if "%COMMIT_MSG%"=="" (
    echo 错误: 提交信息不能为空！
    goto input_commit
)

:: 执行Git操作
echo.
echo 正在切换到仓库目录...
cd /D "%REPO_PATH%"
if %errorlevel% neq 0 (
    echo 错误: 无法访问目录 %REPO_PATH%
    pause
    exit /b
)

:: 检查Git状态
echo [0/4] 检查Git状态...
git status
if %errorlevel% neq 0 (
    echo 错误: 当前目录不是Git仓库或Git未安装
    pause
    exit /b
)

echo.
echo [1/4] 正在拉取远程更新...
git pull origin %BRANCH%
if %errorlevel% neq 0 (
    echo 警告: 拉取时可能有冲突，继续本地操作...
)

echo [2/4] 正在添加所有更改...
git add .
if %errorlevel% neq 0 (
    echo 错误: git add 执行失败
    pause
    exit /b
)

echo [3/4] 正在提交更改...
git commit -m "%COMMIT_MSG%"
if %errorlevel% neq 0 (
    echo 注意: 可能没有需要提交的更改
)

echo [4/4] 正在推送到远程仓库...
echo 请稍候，正在推送...
git push origin %BRANCH%
if %errorlevel% neq 0 (
    echo.
    echo ========================================
    echo           推送失败，尝试强制推送
    echo ========================================
    echo 注意: 强制推送可能会覆盖他人提交
    echo 请确认你知道自己在做什么！
    echo.
    set /p CONFIRM="是否强制推送? (y/n): "
    if /i "%CONFIRM%"=="y" (
        echo 正在执行强制推送...
        git push -f origin %BRANCH%
        if %errorlevel% neq 0 (
            echo 错误: 强制推送也失败！
            pause
            exit /b
        )
        echo 强制推送成功！
    ) else (
        echo 已取消推送，请手动处理冲突
        pause
        exit /b
    )
)

echo.
echo ========================================
echo          提交成功！
echo ========================================
echo 提交信息: %COMMIT_MSG%
echo 仓库位置: %REPO_PATH%
echo 目标分支: %BRANCH%
echo 提交时间: %date% %time%
echo ========================================
echo.
echo 按任意键退出...
pause >nul