@echo off
chcp 65001 >nul
:: 设置UTF-8编码支持中文

:: ==== 自动请求管理员权限 ====
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"
if %errorlevel% neq 0 (
    powershell -Command "Start-Process '%~s0' -Verb RunAs"
    exit /b
)

:: ==== 主程序开始 ====
cls
echo ========================================
echo     Git智能提交工具 (管理员模式)
echo ========================================
echo.

:: 设置变量
set REPO_PATH=D:\Blog\leuco-blog\
set BRANCH=main

echo 仓库: %REPO_PATH%
echo 分支: %BRANCH%
echo.

:: 获取提交信息
set "COMMIT_MSG="
set /p COMMIT_MSG="请输入提交信息: "
if "%COMMIT_MSG%"=="" (
    echo 使用默认提交信息: Auto commit %date% %time%
    set COMMIT_MSG=Auto commit %date% %time%
)

:: 切换到仓库目录
cd /D "%REPO_PATH%" 2>nul || (
    echo 错误: 仓库目录不存在!
    pause
    exit /b
)

:: 智能Git流程
echo.
echo 开始执行Git流程...

:: 1. 先尝试拉取更新（但不合并，避免冲突）
echo [1] 获取远程更新...
git fetch origin
if %errorlevel% neq 0 (
    echo 警告: 无法获取远程更新
)

:: 2. 检查本地是否有未提交的更改
git diff --quiet
if %errorlevel% equ 0 (
    echo 没有检测到本地更改
    goto push_only
)

:: 3. 添加并提交本地更改
echo [2] 暂存所有更改...
git add .

echo [3] 提交更改...
git commit -m "%COMMIT_MSG%"

:: 4. 如果有远程更新，尝试合并
echo [4] 尝试合并远程更新...
git pull --rebase origin %BRANCH%
if %errorlevel% neq 0 (
    echo.
    echo 存在冲突，正在处理...
    echo 选项:
    echo  1) 放弃本地修改，使用远程版本
    echo  2) 强制推送本地版本（覆盖远程）
    echo  3) 手动解决冲突
    echo.
    :retry_choice
    set /p CHOICE="请选择 (1/2/3): "
    
    if "%CHOICE%"=="1" (
        echo 放弃本地修改...
        git stash
        git pull origin %BRANCH%
    ) else if "%CHOICE%"=="2" (
        echo 强制推送本地版本...
        git push -f origin %BRANCH%
        goto success
    ) else if "%CHOICE%"=="3" (
        echo 请手动解决冲突后再次运行此脚本
        echo 命令提示:
        echo  git mergetool    - 使用合并工具
        echo  git status       - 查看状态
        echo  git add .        - 添加解决后的文件
        echo  git rebase --continue - 继续合并
        pause
        exit /b
    ) else (
        goto retry_choice
    )
)

:push_only
:: 5. 推送到远程
echo [5] 推送到远程仓库...
git push origin %BRANCH%
if %errorlevel% neq 0 (
    echo 推送失败，可能是网络问题或权限不足
    pause
    exit /b
)

:success
echo.
echo ========================================
echo          操作完成！
echo ========================================
echo 提交信息: %COMMIT_MSG%
echo 时间: %date% %time%
echo ========================================
echo.
echo 按任意键退出...
pause >nul