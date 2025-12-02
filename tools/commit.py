import tkinter as tk
from tkinter import simpledialog
import subprocess
import os


def git_commit_and_push():
    # 创建Tkinter根窗口但不显示
    root = tk.Tk()
    root.withdraw()

    # 弹出输入对话框
    information = simpledialog.askstring("Git提交", "请输入提交信息：")

    if information is None or information.strip() == "":
        print("提交已取消或提交信息为空")
        return

    # 关闭Tkinter窗口
    root.destroy()

    # 切换到指定目录
    target_dir = r"D:\Blog\leuco-blog"

    try:
        # 检查目录是否存在
        if not os.path.exists(target_dir):
            print(f"错误：目录不存在 - {target_dir}")
            return

        # 切换到目标目录
        original_dir = os.getcwd()
        os.chdir(target_dir)

        print(f"切换到目录: {target_dir}")

        # 执行git add .
        print("正在执行: git add .")
        result_add = subprocess.run(["git", "add", "."],
                                    capture_output=True,
                                    text=True,
                                    encoding='utf-8')

        if result_add.returncode != 0:
            print(f"git add 错误: {result_add.stderr}")
            os.chdir(original_dir)
            return

        print("git add 完成")

        # 执行git commit -m "information"
        print(f"正在执行: git commit -m \"{information}\"")
        result_commit = subprocess.run(["git", "commit", "-m", information],
                                       capture_output=True,
                                       text=True,
                                       encoding='utf-8')

        if result_commit.returncode != 0:
            print(f"git commit 错误: {result_commit.stderr}")
            os.chdir(original_dir)
            return

        print("git commit 完成")

        # 执行git push origin main
        print("正在执行: git push origin main")
        result_push = subprocess.run(["git", "push", "origin", "main"],
                                     capture_output=True,
                                     text=True,
                                     encoding='utf-8')

        if result_push.returncode != 0:
            print(f"git push 错误: {result_push.stderr}")
        else:
            print("git push 成功")

        # 返回原始目录
        os.chdir(original_dir)

    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        os.chdir(original_dir)  # 确保返回原始目录


if __name__ == "__main__":
    git_commit_and_push()
    input("按Enter键退出...")