+++
title= "【攻防实践】基础环境搭建"
date= "2026-01-12T16:58:42+08:00"
lastmod= "2026-01-12T16:58:42+08:00"
draft=false
author= "leuco"
description= "【0】一些基础的应用服务环境搭建"
keywords= ["内网", "文件共享系统", "Flask", "Python"]
categories= ["网络空间安全", "攻防实践"]
tags= ["内网", "文件共享系统", "Flask", "Python"]
math=true
image="cover.png"

+++

# 基础环境搭建

{{<toc>}}

## kali linux内网文件共享系统搭建

### 项目结构

```text
FileShare/
├── app.py              # 主程序
├── uploads/           # 上传文件存储目录
├── downloads/         # 下载文件存储目录
└── templates/
│   └── index.html    # 网页模板
├── start.sh  # 启动脚本
└── stop.sh  # 结束脚本
```

### 安装Flask

```bash
┌──(kali㉿kali)-[~/Desktop]
└─$ sudo apt install python3-flask
```

### app.py

```python
#!/usr/bin/env python3
from flask import Flask, render_template, request, send_from_directory, jsonify, redirect, url_for
import os
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import mimetypes

app = Flask(__name__)

# 配置
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['DOWNLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'downloads')
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB 文件大小限制
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'

# 支持所有文件类型，但可以设置一些危险类型的黑名单
app.config['DANGEROUS_EXTENSIONS'] = {
    'exe', 'msi', 'bat', 'cmd', 'sh', 'bin', 
    'dmg', 'pkg', 'app', 'jar', 'js', 'vbs',
    'ps1', 'py', 'php', 'pl', 'cgi', 'sh', 'bash'
}

# 确保目录存在
for folder in [app.config['UPLOAD_FOLDER'], app.config['DOWNLOAD_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

def is_dangerous_file(filename):
    """检查文件是否为潜在危险类型"""
    if '.' not in filename:
        return False
    
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in app.config['DANGEROUS_EXTENSIONS']

def get_file_info(path):
    """获取文件信息"""
    try:
        stats = os.stat(path)
        file_info = {
            'size': stats.st_size,
            'modified': datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'created': datetime.fromtimestamp(stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 添加文件类型信息
        mime_type, _ = mimetypes.guess_type(path)
        if mime_type:
            file_info['type'] = mime_type
        else:
            # 根据扩展名判断
            ext = os.path.splitext(path)[1].lower().lstrip('.')
            common_types = {
                'zip': 'application/zip',
                'rar': 'application/x-rar-compressed',
                '7z': 'application/x-7z-compressed',
                'tar': 'application/x-tar',
                'gz': 'application/gzip',
                'bz2': 'application/x-bzip2',
                'xz': 'application/x-xz',
                'pdf': 'application/pdf',
                'doc': 'application/msword',
                'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'xls': 'application/vnd.ms-excel',
                'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'ppt': 'application/vnd.ms-powerpoint',
                'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                'mp4': 'video/mp4',
                'avi': 'video/x-msvideo',
                'mkv': 'video/x-matroska',
                'mp3': 'audio/mpeg',
                'wav': 'audio/wav',
                'flac': 'audio/flac',
                'jpg': 'image/jpeg',
                'jpeg': 'image/jpeg',
                'png': 'image/png',
                'gif': 'image/gif',
                'bmp': 'image/bmp',
                'txt': 'text/plain',
                'py': 'text/x-python',
                'js': 'application/javascript',
                'html': 'text/html',
                'css': 'text/css',
                'json': 'application/json',
                'xml': 'application/xml'
            }
            file_info['type'] = common_types.get(ext, 'application/octet-stream')
        
        return file_info
    except Exception as e:
        return {'size': 0, 'modified': 'Unknown', 'created': 'Unknown', 'type': 'Unknown'}

def get_readable_size(size_in_bytes):
    """将字节转换为可读的大小"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.2f} PB"

def is_safe_filename(filename):
    """检查文件名是否安全"""
    # 防止路径遍历攻击
    forbidden_patterns = ['..', '/', '\\']
    for pattern in forbidden_patterns:
        if pattern in filename:
            return False
    return True

@app.route('/')
def index():
    """主页"""
    try:
        # 获取上传文件列表
        upload_files = []
        for filename in sorted(os.listdir(app.config['UPLOAD_FOLDER']), key=lambda x: os.path.getmtime(os.path.join(app.config['UPLOAD_FOLDER'], x)), reverse=True):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(filepath):
                info = get_file_info(filepath)
                upload_files.append({
                    'name': filename,
                    'safe': not is_dangerous_file(filename),
                    **info
                })
        
        # 获取下载文件列表
        download_files = []
        for filename in sorted(os.listdir(app.config['DOWNLOAD_FOLDER']), key=lambda x: os.path.getmtime(os.path.join(app.config['DOWNLOAD_FOLDER'], x)), reverse=True):
            filepath = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
            if os.path.isfile(filepath):
                info = get_file_info(filepath)
                download_files.append({
                    'name': filename,
                    'safe': not is_dangerous_file(filename),
                    **info
                })
        
        return render_template('index.html', 
                             upload_files=upload_files,
                             download_files=download_files,
                             upload_folder=app.config['UPLOAD_FOLDER'],
                             download_folder=app.config['DOWNLOAD_FOLDER'],
                             get_readable_size=get_readable_size)
    
    except Exception as e:
        return f"Error loading page: {str(e)}", 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传"""
    try:
        # 检查是否有文件
        if 'file' not in request.files:
            return jsonify({'error': '没有选择文件'}), 400
        
        file = request.files['file']
        
        # 如果没有选择文件
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        # 检查文件名安全性
        if not is_safe_filename(file.filename):
            return jsonify({'error': '文件名不安全，包含非法字符'}), 400
        
        # 安全处理文件名
        filename = secure_filename(file.filename)
        if not filename:
            filename = 'uploaded_file'
        
        # 处理无扩展名的情况
        if '.' not in filename:
            # 尝试从Content-Type推断扩展名
            content_type = file.content_type
            if content_type:
                ext = mimetypes.guess_extension(content_type)
                if ext:
                    filename = filename + ext
        
        # 避免文件名冲突
        base, extension = os.path.splitext(filename)
        counter = 1
        original_filename = filename
        while os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
            filename = f"{base}_{counter}{extension}"
            counter += 1
        
        # 保存文件
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # 获取文件信息
        info = get_file_info(file_path)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'original_filename': original_filename,
            'message': '文件上传成功',
            'info': info,
            'is_dangerous': is_dangerous_file(filename),
            'warning': '警告：此文件类型可能存在安全风险' if is_dangerous_file(filename) else None
        })
    
    except Exception as e:
        return jsonify({'error': f'上传失败: {str(e)}'}), 500

@app.route('/download/<path:filename>')
def download_file(filename):
    """下载文件"""
    try:
        # 安全验证
        if not is_safe_filename(filename):
            return "文件名不安全", 400
        
        filepath = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return "文件不存在", 404
        
        # 设置下载时显示原始文件名
        as_attachment = True
        download_filename = filename
        
        # 对于某些文件类型，可以在浏览器中预览
        preview_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.gif', '.txt', '.html', '.css', '.js', '.json', '.xml'}
        if any(filename.lower().endswith(ext) for ext in preview_extensions):
            as_attachment = False
        
        return send_from_directory(app.config['DOWNLOAD_FOLDER'], 
                                 filename, 
                                 as_attachment=as_attachment,
                                 download_name=download_filename)
    
    except Exception as e:
        return f"下载失败: {str(e)}", 500

@app.route('/preview/<path:filename>')
def preview_file(filename):
    """预览文件（仅适用于文本和图片）"""
    try:
        if not is_safe_filename(filename):
            return "文件名不安全", 400
        
        filepath = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return "文件不存在", 404
        
        # 检查文件大小（限制预览大小）
        if os.path.getsize(filepath) > 10 * 1024 * 1024:  # 10MB
            return "文件太大，无法预览", 400
        
        # 根据文件类型返回不同的预览
        mime_type, encoding = mimetypes.guess_type(filepath)
        if mime_type:
            if mime_type.startswith('image/'):
                # 图片文件直接返回
                return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename)
            elif mime_type.startswith('text/'):
                # 文本文件读取内容
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                return f'<pre style="white-space: pre-wrap; word-wrap: break-word;">{content}</pre>'
            elif mime_type == 'application/pdf':
                # PDF文件嵌入显示
                return f'''
                <iframe src="/download/{filename}" width="100%" height="800px">
                    您的浏览器不支持PDF预览，请<a href="/download/{filename}">下载</a>查看
                </iframe>
                '''
        
        return f"不支持预览此文件类型，请<a href='/download/{filename}'>下载</a>查看"
    
    except Exception as e:
        return f"预览失败: {str(e)}", 500

@app.route('/delete_upload/<path:filename>', methods=['DELETE'])
def delete_upload_file(filename):
    """删除上传的文件"""
    try:
        if not is_safe_filename(filename):
            return jsonify({'error': '文件名不安全'}), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({'success': True, 'message': '文件已删除'})
        return jsonify({'error': '文件不存在'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete_download/<path:filename>', methods=['DELETE'])
def delete_download_file(filename):
    """删除下载的文件"""
    try:
        if not is_safe_filename(filename):
            return jsonify({'error': '文件名不安全'}), 400
        
        filepath = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({'success': True, 'message': '文件已删除'})
        return jsonify({'error': '文件不存在'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_file_list')
def get_file_list():
    """获取文件列表（JSON格式）"""
    try:
        files = []
        for filename in sorted(os.listdir(app.config['DOWNLOAD_FOLDER']), key=lambda x: os.path.getmtime(os.path.join(app.config['DOWNLOAD_FOLDER'], x)), reverse=True):
            filepath = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
            if os.path.isfile(filepath):
                info = get_file_info(filepath)
                files.append({
                    'name': filename,
                    'safe': not is_dangerous_file(filename),
                    **info
                })
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_upload_list')
def get_upload_list():
    """获取上传文件列表（JSON格式）"""
    try:
        files = []
        for filename in sorted(os.listdir(app.config['UPLOAD_FOLDER']), key=lambda x: os.path.getmtime(os.path.join(app.config['UPLOAD_FOLDER'], x)), reverse=True):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(filepath):
                info = get_file_info(filepath)
                files.append({
                    'name': filename,
                    'safe': not is_dangerous_file(filename),
                    **info
                })
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clean_uploads', methods=['POST'])
def clean_uploads():
    """清理所有上传的文件"""
    try:
        count = 0
        total_size = 0
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(filepath):
                total_size += os.path.getsize(filepath)
                os.remove(filepath)
                count += 1
        
        return jsonify({
            'success': True, 
            'message': f'已清理 {count} 个文件，释放 {get_readable_size(total_size)}',
            'count': count,
            'total_size': total_size
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/create_folder', methods=['POST'])
def create_folder():
    """在下载目录创建文件夹"""
    try:
        data = request.get_json()
        folder_name = data.get('folder_name', '').strip()
        
        if not folder_name:
            return jsonify({'error': '文件夹名不能为空'}), 400
        
        # 安全检查
        if not is_safe_filename(folder_name):
            return jsonify({'error': '文件夹名不安全'}), 400
        
        folder_path = os.path.join(app.config['DOWNLOAD_FOLDER'], folder_name)
        
        if os.path.exists(folder_path):
            return jsonify({'error': '文件夹已存在'}), 400
        
        os.makedirs(folder_path, exist_ok=True)
        return jsonify({'success': True, 'message': f'文件夹 "{folder_name}" 创建成功'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 初始化mimetypes
    mimetypes.init()
    
    # 获取本机IP
    import socket
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = '127.0.0.1'
    
    print("=" * 60)
    print("内网文件共享系统")
    print("=" * 60)
    print(f"📁 上传目录: {app.config['UPLOAD_FOLDER']}")
    print(f"📁 下载目录: {app.config['DOWNLOAD_FOLDER']}")
    print(f"🌐 本地访问: http://127.0.0.1:10000")
    print(f"🌐 局域网访问: http://{local_ip}:10000")
    print(f"💾 支持文件大小: 最大 {app.config['MAX_CONTENT_LENGTH'] // (1024*1024*1024)}GB")
    print(f"⚠️  危险文件类型: {', '.join(sorted(app.config['DANGEROUS_EXTENSIONS']))}")
    print("=" * 60)
    
    # 检查是否需要创建虚拟环境
    try:
        import flask
    except ImportError:
        print("❌ Flask未安装，请运行以下命令:")
        print("   1. python3 -m venv venv")
        print("   2. source venv/bin/activate")
        print("   3. pip install flask")
        exit(1)
    
    # 启动服务器
    app.run(host='0.0.0.0', port=10000, debug=True, threaded=True)
```

### index.html

``` html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>内网文件共享系统</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        header {
            background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
            color: white;
            padding: 25px 40px;
            text-align: center;
        }
        
        header h1 {
            font-size: 2.2em;
            margin-bottom: 10px;
        }
        
        .server-info {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 15px;
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        .content {
            display: flex;
            min-height: 600px;
        }
        
        .panel {
            flex: 1;
            padding: 30px;
            border-right: 1px solid #eaeaea;
        }
        
        .panel:last-child {
            border-right: none;
        }
        
        .panel h2 {
            color: #333;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 2px solid #4b6cb7;
            font-size: 1.5em;
        }
        
        .upload-area {
            border: 3px dashed #4b6cb7;
            border-radius: 10px;
            padding: 40px 20px;
            text-align: center;
            background: #f8f9ff;
            margin-bottom: 30px;
            transition: all 0.3s;
            cursor: pointer;
        }
        
        .upload-area:hover {
            background: #eef2ff;
            border-color: #667eea;
        }
        
        .upload-area.drag-over {
            background: #e0e7ff;
            border-color: #764ba2;
            transform: scale(1.02);
        }
        
        .upload-icon {
            font-size: 48px;
            color: #4b6cb7;
            margin-bottom: 15px;
        }
        
        .file-list {
            margin-top: 20px;
        }
        
        .file-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 15px;
            background: #f8f9fa;
            margin-bottom: 10px;
            border-radius: 8px;
            transition: all 0.2s;
        }
        
        .file-item:hover {
            background: #e9ecef;
            transform: translateX(5px);
        }
        
        .file-info {
            flex: 1;
            min-width: 0;
        }
        
        .file-name {
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
            word-break: break-all;
        }
        
        .file-meta {
            font-size: 0.85em;
            color: #666;
            display: flex;
            gap: 15px;
        }
        
        .file-actions {
            display: flex;
            gap: 10px;
        }
        
        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s;
            text-decoration: none;
            display: inline-block;
            text-align: center;
        }
        
        .btn-download {
            background: #28a745;
            color: white;
        }
        
        .btn-download:hover {
            background: #218838;
        }
        
        .btn-delete {
            background: #dc3545;
            color: white;
        }
        
        .btn-delete:hover {
            background: #c82333;
        }
        
        .btn-refresh {
            background: #17a2b8;
            color: white;
            margin-bottom: 15px;
        }
        
        .btn-refresh:hover {
            background: #138496;
        }
        
        .stats {
            display: flex;
            justify-content: space-between;
            margin-top: 20px;
            padding-top: 15px;
            border-top: 1px solid #eee;
            color: #666;
            font-size: 0.9em;
        }
        
        .empty-message {
            text-align: center;
            color: #999;
            padding: 40px 20px;
            font-style: italic;
        }
        
        .upload-progress {
            margin-top: 20px;
            display: none;
        }
        
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 10px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4b6cb7 0%, #667eea 100%);
            width: 0%;
            transition: width 0.3s;
        }
        
        @media (max-width: 1024px) {
            .content {
                flex-direction: column;
            }
            
            .panel {
                border-right: none;
                border-bottom: 1px solid #eaeaea;
            }
        }
        
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 25px;
            border-radius: 8px;
            color: white;
            font-weight: 500;
            z-index: 1000;
            opacity: 0;
            transform: translateX(100%);
            transition: all 0.3s;
        }
        
        .notification.show {
            opacity: 1;
            transform: translateX(0);
        }
        
        .notification.success {
            background: #28a745;
        }
        
        .notification.error {
            background: #dc3545;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📁 内网文件共享系统</h1>
            <p>拖拽文件到左侧上传，右侧文件可直接下载</p>
            <div class="server-info">
                <div>上传目录：{{ upload_folder }}</div>
                <div>下载目录：{{ download_folder }}</div>
            </div>
        </header>
        
        <div class="content">
            <!-- 左侧上传面板 -->
            <div class="panel">
                <h2>⬆️ 文件上传区</h2>
                
                <div class="upload-area" id="uploadArea">
                    <div class="upload-icon">📁</div>
                    <h3>拖放文件到此处</h3>
                    <p>或点击选择文件</p>
                    <p style="color: #666; margin-top: 10px; font-size: 0.9em;">
                        支持格式：txt, pdf, png, jpg, zip, doc, mp4 等
                    </p>
                    <input type="file" id="fileInput" multiple style="display: none;">
                </div>
                
                <div class="upload-progress" id="uploadProgress">
                    <div>正在上传：<span id="currentFile"></span></div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="progressFill"></div>
                    </div>
                    <div id="progressText">0%</div>
                </div>
                
                <h3>已上传文件</h3>
                <button class="btn btn-refresh" onclick="refreshUploadList()">
                    🔄 刷新列表
                </button>
                
                <div class="file-list" id="uploadFileList">
                    {% if upload_files %}
                        {% for file in upload_files %}
                        <div class="file-item" data-filename="{{ file.name }}">
                            <div class="file-info">
                                <div class="file-name">{{ file.name }}</div>
                                <div class="file-meta">
                                    <span>大小：{{ (file.size / 1024 / 1024)|round(2) if file.size > 1024*1024 else (file.size / 1024)|round(2) }} {{ 'MB' if file.size > 1024*1024 else 'KB' }}</span>
                                    <span>修改：{{ file.modified }}</span>
                                </div>
                            </div>
                            <div class="file-actions">
                                <button class="btn btn-delete" onclick="deleteFile('{{ file.name }}', 'upload')">
                                    🗑️ 删除
                                </button>
                            </div>
                        </div>
                        {% endfor %}
                    {% else %}
                        <div class="empty-message">暂无上传文件</div>
                    {% endif %}
                </div>
                
                <div class="stats">
                    <div>文件总数：{{ upload_files|length }}</div>
                    <div>总大小：{{ upload_files|sum(attribute='size') / 1024 / 1024|round(2) }} MB</div>
                </div>
            </div>
            
            <!-- 右侧下载面板 -->
            <div class="panel">
                <h2>⬇️ 文件下载区</h2>
                
                <button class="btn btn-refresh" onclick="refreshDownloadList()">
                    🔄 刷新列表
                </button>
                
                <div class="file-list" id="downloadFileList">
                    {% if download_files %}
                        {% for file in download_files %}
                        <!-- 修改文件项显示部分 -->
			<div class="file-item" data-filename="{{ file.name }}">
			    <div class="file-info">
				<div class="file-name">
				    {{ file.name }}
				    {% if not file.safe %}
				    <span style="color: #dc3545; font-size: 0.8em; margin-left: 5px;">
					⚠️ 危险文件
				    </span>
				    {% endif %}
				</div>
				<div class="file-meta">
				    <span>大小：{{ get_readable_size(file.size) }}</span>
				    <span>修改：{{ file.modified }}</span>
				    {% if file.type %}
				    <span>类型：{{ file.type }}</span>
				    {% endif %}
				</div>
			    </div>
			    <div class="file-actions">
				{% if file.type and (file.type.startswith('image/') or file.type.startswith('text/') or file.type == 'application/pdf') %}
				<a href="/preview/{{ file.name }}" class="btn btn-preview" target="_blank">
				    👁️ 预览
				</a>
				{% endif %}
				<a href="/download/{{ file.name }}" class="btn btn-download">
				    📥 下载
				</a>
				<button class="btn btn-delete" onclick="deleteFile('{{ file.name }}', 'download')">
				    🗑️ 删除
				</button>
			    </div>
			</div>
                        {% endfor %}
                    {% else %}
                        <div class="empty-message">
                            <p>下载目录为空</p>
                            <p style="margin-top: 10px; font-size: 0.9em;">
                                将文件放入 <code>{{ download_folder }}</code> 目录即可在此显示
                            </p>
                        </div>
                    {% endif %}
                </div>
                
                <div class="stats">
                    <div>文件总数：{{ download_files|length }}</div>
                    <div>总大小：{{ download_files|sum(attribute='size') / 1024 / 1024|round(2) }} MB</div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="notification" id="notification"></div>
    
    <script>
        // 显示通知
        function showNotification(message, type = 'success') {
            const notification = document.getElementById('notification');
            notification.textContent = message;
            notification.className = `notification ${type}`;
            notification.classList.add('show');
            
            setTimeout(() => {
                notification.classList.remove('show');
            }, 3000);
        }
        
        // 刷新上传列表
        async function refreshUploadList() {
            try {
                const response = await fetch('/');
                const text = await response.text();
                const parser = new DOMParser();
                const doc = parser.parseFromString(text, 'text/html');
                const uploadList = doc.getElementById('uploadFileList');
                document.getElementById('uploadFileList').innerHTML = uploadList.innerHTML;
                showNotification('上传列表已刷新', 'success');
            } catch (error) {
                showNotification('刷新失败: ' + error, 'error');
            }
        }
        
        // 刷新下载列表
        async function refreshDownloadList() {
            try {
                const response = await fetch('/get_file_list');
                const data = await response.json();
                
                const downloadList = document.getElementById('downloadFileList');
                if (data.files.length === 0) {
                    downloadList.innerHTML = `
                        <div class="empty-message">
                            <p>下载目录为空</p>
                            <p style="margin-top: 10px; font-size: 0.9em;">
                                将文件放入下载目录即可在此显示
                            </p>
                        </div>
                    `;
                    return;
                }
                
                let html = '';
                data.files.forEach(file => {
                    const size = file.size > 1024*1024 
                        ? (file.size / 1024 / 1024).toFixed(2) + ' MB'
                        : (file.size / 1024).toFixed(2) + ' KB';
                    
                    html += `
                    <div class="file-item" data-filename="${file.name}">
                        <div class="file-info">
                            <div class="file-name">${file.name}</div>
                            <div class="file-meta">
                                <span>大小：${size}</span>
                                <span>修改：${file.modified}</span>
                            </div>
                        </div>
                        <div class="file-actions">
                            <a href="/download/${file.name}" class="btn btn-download">
                                📥 下载
                            </a>
                            <button class="btn btn-delete" onclick="deleteFile('${file.name}', 'download')">
                                🗑️ 删除
                            </button>
                        </div>
                    </div>
                    `;
                });
                
                downloadList.innerHTML = html;
                showNotification('下载列表已刷新', 'success');
            } catch (error) {
                showNotification('刷新失败: ' + error, 'error');
            }
        }
        
        // 删除文件
        async function deleteFile(filename, type) {
            if (!confirm(`确定要删除 ${filename} 吗？`)) {
                return;
            }
            
            try {
                const endpoint = type === 'upload' 
                    ? `/delete_upload/${encodeURIComponent(filename)}`
                    : `/delete_download/${encodeURIComponent(filename)}`;
                
                const response = await fetch(endpoint, {
                    method: 'DELETE'
                });
                
                const data = await response.json();
                
                if (data.success) {
                    showNotification('文件删除成功', 'success');
                    if (type === 'upload') {
                        refreshUploadList();
                    } else {
                        refreshDownloadList();
                    }
                } else {
                    showNotification('删除失败: ' + data.error, 'error');
                }
            } catch (error) {
                showNotification('删除失败: ' + error, 'error');
            }
        }
        
        // 文件上传处理
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const uploadProgress = document.getElementById('uploadProgress');
        const currentFile = document.getElementById('currentFile');
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        
        // 点击上传区域选择文件
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });
        
        // 文件选择变化
        fileInput.addEventListener('change', (e) => {
            handleFiles(e.target.files);
        });
        
        // 拖放上传
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight() {
            uploadArea.classList.add('drag-over');
        }
        
        function unhighlight() {
            uploadArea.classList.remove('drag-over');
        }
        
        // 处理文件放置
        uploadArea.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            handleFiles(files);
        });
        
        // 处理文件上传
        async function handleFiles(files) {
            if (files.length === 0) return;
            
            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                
                // 显示上传进度
                uploadProgress.style.display = 'block';
                currentFile.textContent = file.name;
                progressFill.style.width = '0%';
                progressText.textContent = '0%';
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        progressFill.style.width = '100%';
                        progressText.textContent = '100%';
                        showNotification(`${file.name} 上传成功`, 'success');
                        
                        // 刷新上传列表
                        setTimeout(refreshUploadList, 500);
                    } else {
                        showNotification(`${file.name} 上传失败: ${data.error}`, 'error');
                    }
                } catch (error) {
                    showNotification(`${file.name} 上传失败: ${error}`, 'error');
                }
            }
            
            // 隐藏上传进度条
            setTimeout(() => {
                uploadProgress.style.display = 'none';
            }, 2000);
            
            // 清除文件输入
            fileInput.value = '';
        }
    </script>
</body>
</html>
```

### start.sh

```bash
#!/bin/bash

# 文件共享系统一键启动脚本
# 使用方法: ./start.sh

set -e  # 出错时停止执行

echo "========================================"
echo "  内网文件共享系统 - 启动脚本"
echo "========================================"

# 定义颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 输出带颜色的消息
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否以root运行
if [ "$EUID" -eq 0 ]; then 
    warning "检测到以root用户运行，建议使用普通用户"
    read -p "是否继续? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消"
        exit 1
    fi
fi

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

info "当前目录: $(pwd)"

# 检查必要的文件
if [ ! -f "app.py" ]; then
    error "找不到 app.py 文件"
    exit 1
fi

if [ ! -d "templates" ]; then
    error "找不到 templates 目录"
    exit 1
fi

# 创建必要的目录
mkdir -p uploads downloads logs

# 检查Python环境
info "检查Python环境..."
if ! command -v python3 &> /dev/null; then
    error "未找到 python3，请先安装Python"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
info "Python版本: $PYTHON_VERSION"

# 检查Flask是否已安装
info "检查Flask安装..."
if ! python3 -c "import flask" 2>/dev/null; then
    error "Flask未安装，请先安装Flask: pip install flask"
    exit 1
fi

FLASK_VERSION=$(python3 -c "import flask; print(flask.__version__)" 2>/dev/null || echo "未知")
info "Flask版本: $FLASK_VERSION"

# 检查防火墙（端口改为10000）- 简化检查
info "检查防火墙设置..."
if command -v ufw &> /dev/null; then
    UFW_STATUS=$(sudo ufw status 2>/dev/null || echo "inactive")
    if echo "$UFW_STATUS" | grep -q "active"; then
        warning "检测到UFW防火墙已启用"
        info "检查10000端口..."
        if echo "$UFW_STATUS" | grep -q "10000/tcp"; then
            success "10000端口已开放"
        else
            warning "10000端口未开放"
            read -p "是否开放10000端口? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                sudo ufw allow 10000/tcp
                sudo ufw reload
                success "10000端口已开放"
            else
                warning "如果无法访问，请手动开放端口: sudo ufw allow 10000/tcp"
            fi
        fi
    fi
fi

# 获取本机IP
info "获取本机IP地址..."
IP_ADDRESSES=""

# 尝试多种方法获取IP
if command -v ip &> /dev/null; then
    # 使用ip命令
    IP_ADDRESSES=$(ip -o -4 addr show 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | grep -v '127.0.0.1' | head -5)
elif command -v ifconfig &> /dev/null; then
    # 使用ifconfig命令
    IP_ADDRESSES=$(ifconfig 2>/dev/null | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1' | cut -d' ' -f2 | head -5)
else
    # 最后尝试使用hostname
    IP_ADDRESSES=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -v '^$' | head -5 || echo "")
fi

if [ -z "$IP_ADDRESSES" ]; then
    IP_ADDRESSES="无法获取IP"
    warning "无法获取本机IP地址，只能通过127.0.0.1访问"
else
    info "找到IP地址:"
    for ip in $IP_ADDRESSES; do
        echo "  - $ip"
    done
fi

# 检查是否已在运行
info "检查服务器状态..."
PID_FILE="server.pid"
LOG_FILE="logs/server.log"
ERROR_FILE="logs/error.log"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        warning "服务器已在运行 (PID: $PID)"
        read -p "是否重启服务器? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            info "停止现有服务器..."
            kill "$PID" 2>/dev/null || true
            sleep 2
            if ps -p "$PID" > /dev/null 2>&1; then
                warning "正常停止失败，强制停止..."
                kill -9 "$PID" 2>/dev/null || true
                sleep 1
            fi
            rm -f "$PID_FILE"
            success "服务器已停止"
        else
            info "服务器状态:"
            echo "  PID: $PID"
            echo "  日志文件: $LOG_FILE"
            echo "  访问地址: http://127.0.0.1:10000"
            for ip in $IP_ADDRESSES; do
                if [ "$ip" != "无法获取IP" ]; then
                    echo "           http://$ip:10000"
                fi
            done
            echo ""
            echo "查看日志: tail -f $LOG_FILE"
            echo "停止服务器: ./stop_server.sh 或 kill $PID"
            exit 0
        fi
    else
        info "清理旧的PID文件"
        rm -f "$PID_FILE"
    fi
fi

# 备份旧的日志
if [ -f "$LOG_FILE" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    mv "$LOG_FILE" "logs/server_${TIMESTAMP}.log" 2>/dev/null || true
    info "已备份旧日志: logs/server_${TIMESTAMP}.log"
fi

if [ -f "$ERROR_FILE" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    mv "$ERROR_FILE" "logs/error_${TIMESTAMP}.log" 2>/dev/null || true
    info "已备份旧错误日志: logs/error_${TIMESTAMP}.log"
fi

# 启动服务器
info "正在启动服务器（端口: 10000）..."
echo "启动时间: $(date)" > "$LOG_FILE"
echo "启动目录: $(pwd)" >> "$LOG_FILE"
echo "Python版本: $(python3 --version)" >> "$LOG_FILE"
echo "Flask版本: $FLASK_VERSION" >> "$LOG_FILE"
echo "运行端口: 10000" >> "$LOG_FILE"

# 在后台启动服务器
info "启动命令: python3 app.py"
nohup python3 app.py > "$LOG_FILE" 2>"$ERROR_FILE" &
SERVER_PID=$!

# 保存PID
echo $SERVER_PID > "$PID_FILE"
info "服务器进程ID: $SERVER_PID"

# 等待服务器启动
info "等待服务器启动（最多等待10秒）..."
for i in {1..10}; do
    if ps -p "$SERVER_PID" > /dev/null 2>&1; then
        # 检查端口是否在监听
        if ss -tuln 2>/dev/null | grep -q ":10000 " || netstat -tuln 2>/dev/null | grep -q ":10000 "; then
            success "服务器启动成功，端口10000正在监听"
            break
        fi
        
        if [ $i -eq 10 ]; then
            warning "服务器进程在运行，但端口10000未监听"
        else
            echo -n "."
            sleep 1
        fi
    else
        error "服务器进程已退出"
        break
    fi
done
echo ""

# 检查服务器是否启动成功
if ps -p "$SERVER_PID" > /dev/null 2>&1; then
    success "✅ 服务器启动成功!"
    echo ""
    echo "========================================"
    echo "  服务器信息"
    echo "========================================"
    echo "  PID: $SERVER_PID"
    echo "  状态: 运行中"
    echo "  端口: 10000"
    echo "  日志文件: $LOG_FILE"
    echo "  错误日志: $ERROR_FILE"
    echo ""
    echo "========================================"
    echo "  访问地址"
    echo "========================================"
    echo "  本地访问: http://127.0.0.1:10000"
    if [ "$IP_ADDRESSES" != "无法获取IP" ]; then
        echo "  局域网访问:"
        for ip in $IP_ADDRESSES; do
            echo "          http://$ip:10000"
        done
    fi
    echo ""
    echo "========================================"
    echo "  管理命令"
    echo "========================================"
    echo "  查看实时日志: tail -f $LOG_FILE"
    echo "  查看错误日志: tail -f $ERROR_FILE"
    echo "  停止服务器: ./stop_server.sh 或 kill $SERVER_PID"
    echo "  重启服务器: ./restart_server.sh"
    echo ""
    echo "========================================"
    echo "  文件目录"
    echo "========================================"
    echo "  上传目录: $(pwd)/uploads/"
    echo "  下载目录: $(pwd)/downloads/"
    echo "  日志目录: $(pwd)/logs/"
    echo ""
    
    # 尝试获取服务器信息
    info "测试服务器连接..."
    if command -v curl &> /dev/null; then
        if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:10000 2>/dev/null | grep -q "200\|30"; then
            success "服务器响应正常"
        else
            warning "服务器响应异常，请检查日志"
        fi
    else
        warning "未找到curl，跳过连接测试"
    fi
    
    # 显示最后几行日志
    info "最近日志:"
    if [ -f "$LOG_FILE" ] && [ -s "$LOG_FILE" ]; then
        tail -10 "$LOG_FILE"
    else
        echo "日志文件为空"
    fi
    
else
    error "❌ 服务器启动失败"
    error "请检查日志文件: $LOG_FILE"
    error "错误日志: $ERROR_FILE"
    
    # 显示错误日志
    if [ -f "$ERROR_FILE" ] && [ -s "$ERROR_FILE" ]; then
        echo ""
        echo "错误信息:"
        tail -20 "$ERROR_FILE"
    else
        echo "错误日志文件为空"
    fi
    
    rm -f "$PID_FILE"
    
    # 检查端口是否被占用
    if ss -tuln 2>/dev/null | grep -q ":10000 " || netstat -tuln 2>/dev/null | grep -q ":10000 "; then
        warning "端口10000可能已被其他进程占用"
        info "查看占用端口的进程:"
        if command -v lsof &> /dev/null; then
            sudo lsof -i :10000 || echo "无法查看端口占用情况"
        elif command -v fuser &> /dev/null; then
            sudo fuser 10000/tcp || echo "无法查看端口占用情况"
        fi
    fi
    
    exit 1
fi

# 保持脚本运行（可选）
echo ""
read -p "按回车键退出，或输入 'l' 查看实时日志: " -n 1 -r
echo
if [[ $REPLY =~ ^[Ll]$ ]]; then
    echo "开始查看实时日志 (Ctrl+C 退出)..."
    tail -f "$LOG_FILE"
fi

exit 0
```

### stop.sh

```bash
#!/bin/bash

# 文件共享系统服务结束脚本
# 使用方法: ./stop.sh [选项]

set -e  # 出错时停止执行

echo "========================================"
echo "  内网文件共享系统 - 服务结束脚本"
echo "========================================"

# 定义颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 输出带颜色的消息
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  -h, --help     显示此帮助信息"
    echo "  -f, --force    强制停止（使用kill -9）"
    echo "  -s, --status   仅显示服务器状态，不停止"
    echo "  -v, --verbose  显示详细信息"
    echo "  -a, --all      停止所有相关进程（包括子进程）"
    echo ""
    echo "示例:"
    echo "  $0             正常停止服务器"
    echo "  $0 -f          强制停止服务器"
    echo "  $0 -s          仅查看服务器状态"
    echo "  $0 -v -f       强制停止并显示详细信息"
}

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# 解析命令行参数
FORCE=false
STATUS_ONLY=false
VERBOSE=false
STOP_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -s|--status)
            STATUS_ONLY=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -a|--all)
            STOP_ALL=true
            shift
            ;;
        *)
            error "未知选项: $1"
            echo "使用 $0 -h 查看帮助"
            exit 1
            ;;
    esac
done

# 文件路径
PID_FILE="server.pid"
LOG_FILE="logs/server.log"
ERROR_FILE="logs/error.log"

# 显示详细信息的函数
verbose_info() {
    if [ "$VERBOSE" = true ]; then
        info "$1"
    fi
}

# 检查PID文件是否存在
check_pid_file() {
    if [ ! -f "$PID_FILE" ]; then
        if [ "$STATUS_ONLY" = false ]; then
            warning "未找到PID文件: $PID_FILE"
        fi
        return 1
    fi
    
    PID=$(cat "$PID_FILE" 2>/dev/null)
    if [ -z "$PID" ] || [[ ! "$PID" =~ ^[0-9]+$ ]]; then
        warning "PID文件内容无效"
        return 1
    fi
    
    return 0
}

# 显示服务器状态
show_status() {
    echo "========================================"
    echo "  服务器状态"
    echo "========================================"
    
    # 检查PID文件
    if check_pid_file; then
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "  ✅ 服务器正在运行"
            echo "  PID: $PID"
            
            # 获取进程详细信息
            if command -v ps &> /dev/null; then
                PROCESS_INFO=$(ps -p "$PID" -o pid,ppid,user,cmd --no-headers 2>/dev/null || echo "无法获取进程信息")
                echo "  进程信息:"
                echo "    $PROCESS_INFO"
            fi
            
            # 检查端口占用
            info "检查端口占用情况..."
            if command -v ss &> /dev/null; then
                PORT_INFO=$(sudo ss -tulpn 2>/dev/null | grep ":10000 " || echo "端口未找到或没有权限")
            elif command -v netstat &> /dev/null; then
                PORT_INFO=$(sudo netstat -tulpn 2>/dev/null | grep ":10000 " || echo "端口未找到或没有权限")
            else
                PORT_INFO="无法检查端口（请安装ss或netstat）"
            fi
            echo "  端口10000状态:"
            echo "    $PORT_INFO"
            
            # 显示启动时间
            if [ -f "$LOG_FILE" ]; then
                START_TIME=$(grep "启动时间:" "$LOG_FILE" | head -1 | cut -d: -f2-)
                if [ -n "$START_TIME" ]; then
                    echo "  启动时间:$START_TIME"
                fi
            fi
            
            # 显示日志文件大小
            if [ -f "$LOG_FILE" ]; then
                LOG_SIZE=$(du -h "$LOG_FILE" 2>/dev/null | cut -f1 || echo "未知")
                echo "  日志大小: $LOG_SIZE"
            fi
            
            return 0
        else
            echo "  ❌ 服务器未运行（PID文件存在但进程不存在）"
            echo "  PID: $PID"
            return 1
        fi
    else
        # 检查是否有其他相关进程
        info "检查是否有相关进程在运行..."
        RELATED_PIDS=$(ps aux | grep -E "python.*app\.py|flask" | grep -v grep | awk '{print $2}' | tr '\n' ' ')
        
        if [ -n "$RELATED_PIDS" ]; then
            echo "  ⚠️  发现相关进程（但不是通过本脚本启动）"
            echo "  相关进程PID: $RELATED_PIDS"
            echo "  进程详情:"
            for pid in $RELATED_PIDS; do
                ps -p "$pid" -o pid,user,cmd --no-headers 2>/dev/null || true
            done
            return 2
        else
            echo "  ❌ 服务器未运行"
            return 3
        fi
    fi
}

# 停止服务器
stop_server() {
    if ! check_pid_file; then
        # 尝试查找相关进程
        info "尝试查找文件共享相关进程..."
        PYTHON_PIDS=$(ps aux | grep -E "python.*app\.py" | grep -v grep | awk '{print $2}')
        
        if [ -z "$PYTHON_PIDS" ]; then
            success "没有找到正在运行的文件共享服务器"
            return 0
        fi
        
        warning "未找到PID文件，但发现相关进程: $PYTHON_PIDS"
        read -p "是否停止这些进程? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            info "操作已取消"
            return 0
        fi
        
        # 停止所有找到的进程
        STOPPED_COUNT=0
        for pid in $PYTHON_PIDS; do
            info "停止进程 $pid..."
            if kill $pid 2>/dev/null; then
                verbose_info "已发送停止信号到进程 $pid"
                STOPPED_COUNT=$((STOPPED_COUNT + 1))
            else
                warning "无法停止进程 $pid"
            fi
        done
        
        # 等待进程结束
        info "等待进程结束..."
        for i in {1..5}; do
            REMAINING=$(ps aux | grep -E "python.*app\.py" | grep -v grep | wc -l)
            if [ "$REMAINING" -eq 0 ]; then
                break
            fi
            sleep 1
        done
        
        # 检查是否还有进程运行
        REMAINING_PIDS=$(ps aux | grep -E "python.*app\.py" | grep -v grep | awk '{print $2}')
        if [ -n "$REMAINING_PIDS" ] && [ "$FORCE" = true ]; then
            warning "还有进程在运行，尝试强制停止..."
            for pid in $REMAINING_PIDS; do
                if kill -9 "$pid" 2>/dev/null; then
                    verbose_info "已强制停止进程 $pid"
                fi
            done
        fi
        
        success "已停止 $STOPPED_COUNT 个进程"
        return 0
    fi
    
    # 正常情况：有PID文件
    if ps -p "$PID" > /dev/null 2>&1; then
        info "正在停止服务器 (PID: $PID)..."
        
        # 先尝试正常停止
        if kill "$PID" 2>/dev/null; then
            verbose_info "已发送停止信号到进程 $PID"
            
            # 等待进程结束
            info "等待服务器停止（最多10秒）..."
            for i in {1..10}; do
                if ! ps -p "$PID" > /dev/null 2>&1; then
                    break
                fi
                if [ "$i" -eq 5 ]; then
                    info "服务器仍在运行，继续等待..."
                fi
                sleep 1
            done
        else
            warning "无法发送停止信号到进程 $PID"
            FORCE=true
        fi
        
        # 检查是否还需要强制停止
        if ps -p "$PID" > /dev/null 2>&1; then
            if [ "$FORCE" = true ]; then
                warning "服务器仍在运行，尝试强制停止..."
                if kill -9 "$PID" 2>/dev/null; then
                    success "已强制停止服务器 (PID: $PID)"
                else
                    error "无法强制停止服务器"
                    return 1
                fi
            else
                warning "服务器仍在运行，请使用 -f 选项强制停止"
                read -p "是否强制停止? (y/N): " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    if kill -9 "$PID" 2>/dev/null; then
                        success "已强制停止服务器 (PID: $PID)"
                    else
                        error "无法强制停止服务器"
                        return 1
                    fi
                else
                    info "操作已取消"
                    return 0
                fi
            fi
        else
            success "服务器已正常停止 (PID: $PID)"
        fi
        
        # 清理PID文件
        if [ -f "$PID_FILE" ]; then
            rm -f "$PID_FILE"
            verbose_info "已删除PID文件: $PID_FILE"
        fi
        
        # 如果需要，停止所有相关进程
        if [ "$STOP_ALL" = true ]; then
            info "查找并停止所有相关子进程..."
            CHILD_PIDS=$(pgrep -P "$PID" 2>/dev/null || echo "")
            if [ -n "$CHILD_PIDS" ]; then
                for child_pid in $CHILD_PIDS; do
                    kill "$child_pid" 2>/dev/null && verbose_info "已停止子进程: $child_pid"
                done
            fi
        fi
        
    else
        warning "服务器未运行（PID文件存在但进程不存在）"
        # 清理无效的PID文件
        if [ -f "$PID_FILE" ]; then
            rm -f "$PID_FILE"
            info "已清理无效的PID文件"
        fi
    fi
    
    return 0
}

# 主逻辑
if [ "$STATUS_ONLY" = true ]; then
    show_status
    exit $?
fi

# 显示当前状态
show_status
echo ""

# 确认操作
if [ "$FORCE" = false ]; then
    if check_pid_file && ps -p "$PID" > /dev/null 2>&1; then
        read -p "确定要停止服务器吗? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            info "操作已取消"
            exit 0
        fi
    fi
fi

# 执行停止操作
if stop_server; then
    echo ""
    success "✅ 操作完成"
    
    # 显示最终状态
    if [ "$VERBOSE" = true ]; then
        echo ""
        show_status
    fi
else
    error "❌ 停止服务器时出错"
    exit 1
fi

exit 0
```











