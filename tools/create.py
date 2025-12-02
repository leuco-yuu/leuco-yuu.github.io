import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import shutil
from datetime import datetime
import re
import webbrowser
import subprocess
import platform


class BlogPostCreator:
    def __init__(self, root):
        self.root = root
        self.root.title("新建博文")
        self.root.geometry("600x560")  # 窗口大小已减小，因为隐藏了一个字段

        # 存储图片信息
        self.image_path = None
        self.image_filename = None
        # 存储创建的文件路径
        self.created_file_path = None

        # 设置工作路径
        self.working_dir = r"D:\Blog\leuco-blog\content\post"

        # 初始化字段变量字典
        self.vars = {}

        # 创建UI
        self.create_widgets()

    def create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # 创建左侧标签和右侧输入框
        self.fields = [
            '标题', '日期', '最后修改', '草稿', '作者', '描述',
            '关键词', '分类', '标签', '所属系列', '封面图片', '封面图片说明',
            '公式', '别名'
        ]

        # 创建字段名到变量的映射
        self.field_to_var = {
            '标题': 'title',
            '日期': 'date',
            '最后修改': 'lastmod',
            '草稿': 'draft',
            '作者': 'author',
            '描述': 'description',
            '关键词': 'keywords',
            '分类': 'categories',
            '标签': 'tags',
            '所属系列': 'series',
            '封面图片': 'image',
            '封面图片说明': 'image_caption',
            '公式': 'math',
            '别名': 'aliases'
        }

        # 设置当前时间
        current_time = datetime.now()
        date_str = current_time.strftime("%Y-%m-%dT%H:%M:%S+08:00")
        lastmod_str = current_time.strftime("%Y-%m-%dT%H:%M:%S+08:00")

        row_index = 0
        for i, field in enumerate(self.fields):
            # 隐藏image_caption字段
            if field == '封面图片说明':
                continue

            # 标签
            label = ttk.Label(main_frame, text=f"{field}:", width=15, anchor=tk.E)
            label.grid(row=row_index, column=0, padx=5, pady=5, sticky=tk.E)

            var_name = self.field_to_var[field]

            if var_name in ['draft', 'math']:
                # 下拉选择框
                var = tk.StringVar()
                combo = ttk.Combobox(main_frame, textvariable=var, state="readonly", width=40)

                if var_name == 'draft':
                    combo['values'] = ('false', 'true')
                    var.set('false')
                else:  # math
                    combo['values'] = ('true', 'false')
                    var.set('true')

                combo.grid(row=row_index, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
                self.vars[var_name] = var

            elif var_name == 'image':
                # 初始化image和image_caption的变量
                self.vars['image'] = tk.StringVar()
                self.vars['image_caption'] = tk.StringVar()

                # 图片上传按钮和显示框
                frame = ttk.Frame(main_frame)
                frame.grid(row=row_index, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
                frame.columnconfigure(1, weight=1)

                btn = ttk.Button(frame, text="上传图片", command=self.upload_image)
                btn.grid(row=0, column=0, padx=(0, 5))

                self.image_label = ttk.Label(frame, text="未选择图片", width=30)
                self.image_label.grid(row=0, column=1, sticky=tk.W)

            elif var_name in ['date', 'lastmod']:
                # 日期字段，自动填充
                var = tk.StringVar()
                if var_name == 'date':
                    var.set(date_str)
                else:
                    var.set(lastmod_str)

                entry = ttk.Entry(main_frame, textvariable=var, width=42)
                entry.grid(row=row_index, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
                self.vars[var_name] = var

            else:
                # 跳过已经处理过的字段
                if var_name in ['image', 'image_caption'] and var_name in self.vars:
                    continue

                # 普通输入框
                if var_name == 'author':
                    var = tk.StringVar(value='leuco')
                else:
                    var = tk.StringVar()

                entry = ttk.Entry(main_frame, textvariable=var, width=42)
                entry.grid(row=row_index, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))

                if var_name == 'title':
                    # 标题特殊处理，设置焦点
                    entry.focus_set()

                self.vars[var_name] = var

            row_index += 1

        # 添加同步tags和keywords的复选框
        sync_frame = ttk.Frame(main_frame)
        sync_frame.grid(row=row_index, column=0, columnspan=2, pady=5)
        row_index += 1

        self.sync_var = tk.BooleanVar(value=True)
        sync_check = ttk.Checkbutton(sync_frame, text="标签与关键词同步",
                                     variable=self.sync_var,
                                     command=self.on_sync_changed)
        sync_check.grid(row=0, column=0, sticky=tk.W)

        # 添加事件绑定，当keywords变化时自动更新tags（如果启用同步）
        self.vars['keywords'].trace_add('write', self.on_keywords_changed)

        # 创建按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row_index, column=0, columnspan=2, pady=20)

        # 创建按钮
        create_btn = ttk.Button(button_frame, text="创建博文", command=self.create_post)
        create_btn.pack(side=tk.LEFT, padx=10)

        cancel_btn = ttk.Button(button_frame, text="取消", command=self.root.quit)
        cancel_btn.pack(side=tk.LEFT, padx=10)

    def on_sync_changed(self):
        """同步选项改变时的处理"""
        if self.sync_var.get():
            # 如果启用同步，用keywords的值更新tags
            keywords_value = self.vars['keywords'].get()
            self.vars['tags'].set(keywords_value)

    def on_keywords_changed(self, *args):
        """keywords变化时的处理"""
        if self.sync_var.get():
            # 如果启用同步，自动更新tags
            keywords_value = self.vars['keywords'].get()
            self.vars['tags'].set(keywords_value)

    def upload_image(self):
        """上传图片文件"""
        filetypes = [
            ("图片文件", "*.jpg *.jpeg *.png *.gif *.bmp"),
            ("所有文件", "*.*")
        ]

        filename = filedialog.askopenfilename(
            title="选择图片",
            filetypes=filetypes
        )

        if filename:
            self.image_path = filename
            self.image_filename = os.path.basename(filename)
            self.image_label.config(text=self.image_filename)

            # 设置image和image_caption字段
            self.vars['image'].set(self.image_filename)

            # 移除扩展名作为caption
            name_without_ext = os.path.splitext(self.image_filename)[0]
            self.vars['image_caption'].set(name_without_ext)

    def open_created_file(self):
        """打开创建的博文文件"""
        if self.created_file_path and os.path.exists(self.created_file_path):
            try:
                system = platform.system()

                if system == 'Windows':
                    # Windows系统：用默认程序打开
                    os.startfile(self.created_file_path)
                elif system == 'Darwin':  # macOS
                    # macOS系统
                    subprocess.call(['open', self.created_file_path])
                else:  # Linux
                    # Linux系统
                    subprocess.call(['xdg-open', self.created_file_path])

            except Exception as e:
                messagebox.showerror("错误", f"打开文件失败：{str(e)}")
        else:
            messagebox.showwarning("警告", "文件不存在或路径无效")

    def open_folder(self):
        """打开创建的文件夹"""
        if self.created_file_path and os.path.exists(self.created_file_path):
            folder_path = os.path.dirname(self.created_file_path)
            try:
                system = platform.system()

                if system == 'Windows':
                    # Windows系统：用资源管理器打开文件夹并选中文件
                    subprocess.Popen(f'explorer /select,"{self.created_file_path}"')
                elif system == 'Darwin':  # macOS
                    # macOS系统
                    subprocess.call(['open', '-R', self.created_file_path])
                else:  # Linux
                    # Linux系统
                    subprocess.call(['xdg-open', folder_path])

            except Exception as e:
                messagebox.showerror("错误", f"打开文件夹失败：{str(e)}")
        else:
            messagebox.showwarning("警告", "文件夹不存在或路径无效")

    def create_post(self):
        """创建博文"""
        # 检查标题是否为空
        title = self.vars['title'].get().strip()
        if not title:
            messagebox.showerror("错误", "标题不能为空！")
            return

        try:
            # 清理标题，用于文件夹名
            clean_title = re.sub(r'[<>:"/\\|?*]', '-', title)  # 移除非法字符
            clean_title = clean_title.replace(' ', '-')

            # 创建文件夹
            current_time = datetime.now().strftime("%Y%m%d%H%M%S")
            folder_name = f"{clean_title}-{current_time}"
            folder_path = os.path.join(self.working_dir, folder_name)

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            else:
                messagebox.showwarning("警告", f"文件夹已存在：{folder_path}")

            # 复制图片
            if self.image_path and self.image_filename:
                dest_image_path = os.path.join(folder_path, self.image_filename)
                try:
                    shutil.copy2(self.image_path, dest_image_path)
                except Exception as e:
                    messagebox.showwarning("警告", f"图片复制失败：{str(e)}")

            # 创建Markdown文件
            md_content = self.generate_markdown_content()
            md_file_path = os.path.join(folder_path, "index.md")
            self.created_file_path = md_file_path  # 保存文件路径

            with open(md_file_path, 'w', encoding='utf-8') as f:
                f.write(md_content)

            # 显示操作选项对话框
            self.show_success_dialog(md_file_path)

        except Exception as e:
            messagebox.showerror("错误", f"创建失败：{str(e)}")

    def show_success_dialog(self, file_path):
        """显示成功对话框并提供选项"""
        # 创建自定义对话框
        dialog = tk.Toplevel(self.root)
        dialog.title("创建成功")
        dialog.geometry("400x200")
        dialog.transient(self.root)  # 设置为父窗口的临时窗口
        dialog.grab_set()  # 模态对话框

        # 使对话框居中
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'400x200+{x}+{y}')

        # 添加内容
        msg_frame = ttk.Frame(dialog, padding="20")
        msg_frame.pack(fill=tk.BOTH, expand=True)

        # 显示成功消息
        success_label = ttk.Label(msg_frame, text="博文创建成功！", font=("Arial", 12, "bold"))
        success_label.pack(pady=(0, 10))

        # 显示文件路径
        path_label = ttk.Label(msg_frame, text=f"位置：\n{file_path}",
                               wraplength=350, justify=tk.LEFT)
        path_label.pack(pady=(0, 20))

        # 按钮框架
        btn_frame = ttk.Frame(msg_frame)
        btn_frame.pack()

        # 打开文件按钮
        open_file_btn = ttk.Button(btn_frame, text="打开博文",
                                   command=lambda: [self.open_created_file(), dialog.destroy()])
        open_file_btn.grid(row=0, column=0, padx=5)

        # 打开文件夹按钮
        open_folder_btn = ttk.Button(btn_frame, text="打开文件夹",
                                     command=lambda: [self.open_folder(), dialog.destroy()])
        open_folder_btn.grid(row=0, column=1, padx=5)

        # 关闭按钮
        close_btn = ttk.Button(btn_frame, text="关闭",
                               command=lambda: [dialog.destroy(), self.root.quit()])
        close_btn.grid(row=0, column=2, padx=5)

        # 等待对话框关闭
        self.root.wait_window(dialog)

    def generate_markdown_content(self):
        """生成Markdown内容"""
        lines = ["+++"]

        # 处理各个字段
        field_mapping = {
            'title': 'title',
            'date': 'date',
            'lastmod': 'lastmod',
            'draft': 'draft',
            'author': 'author',
            'description': 'description',
            'keywords': 'keywords',
            'categories': 'categories',
            'tags': 'tags',
            'series': 'series',
            'aliases': 'aliases',
            'image': 'image',
            'image_caption': 'image_caption',
            'math': 'math'
        }

        for field_name, yaml_key in field_mapping.items():
            if field_name not in self.vars:
                continue

            value = self.vars[field_name].get().strip()

            if not value:
                continue

            # 特殊处理布尔值
            if field_name in ['draft', 'math']:
                lines.append(f'{yaml_key}={value}')

            # 处理列表类型（用分号分隔）
            elif field_name in ['keywords', 'categories', 'tags', 'series']:
                # 分割字符串，支持中文和英文分号
                items = re.split(r'[;；]', value)
                items = [item.strip() for item in items if item.strip()]

                if items:
                    if len(items) == 1:
                        lines.append(f'{yaml_key}= "{items[0]}"')
                    else:
                        items_str = '["' + '", "'.join(items) + '"]'
                        lines.append(f'{yaml_key}= {items_str}')

            # 处理aliases（可能多个，用分号分隔）
            elif field_name == 'aliases':
                aliases = re.split(r'[;；]', value)
                aliases = [alias.strip() for alias in aliases if alias.strip()]

                if aliases:
                    if len(aliases) == 1:
                        lines.append(f'{yaml_key}= "{aliases[0]}"')
                    else:
                        aliases_str = '["' + '", "'.join(aliases) + '"]'
                        lines.append(f'{yaml_key}= {aliases_str}')

            # 普通字符串
            else:
                lines.append(f'{yaml_key}= "{value}"')

        lines.append("+++\n")
        return "\n".join(lines)


def main():
    root = tk.Tk()
    app = BlogPostCreator(root)

    # 使窗口居中
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')

    root.mainloop()


if __name__ == "__main__":
    main()