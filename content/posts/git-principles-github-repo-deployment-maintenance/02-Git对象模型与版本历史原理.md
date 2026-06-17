# Git 对象模型与版本历史原理

### 内容寻址与对象数据库

Git 的核心是对象数据库。`.git/objects` 目录中保存了 Git 的对象。对象通过内容计算哈希值，哈希值既是对象的标识符，也是完整性校验的一部分。传统 Git 仓库默认使用 SHA-1；新版本 Git 也支持初始化 SHA-256 仓库，但 SHA-256 仓库与 SHA-1 仓库目前不能直接互操作，因此生产环境选择对象格式时要谨慎。

可以通过如下命令观察对象生成过程：

```bash
echo "hello git" > message.txt
git hash-object message.txt
git hash-object -w message.txt
```

不带 `-w` 时，Git 只计算对象 ID；带 `-w` 时，Git 将内容写入对象数据库。对象写入后，即使工作区文件被删除，对象仍可通过对象 ID 找回，前提是对象尚未被垃圾回收清理。

### 四类核心对象

Git 常见对象类型包括 blob、tree、commit 和 tag。

blob 保存文件内容，不保存文件名。两个路径不同但内容完全一致的文件可以复用同一个 blob。tree 保存目录结构，记录路径名、权限、对象类型和对象 ID。commit 保存一次提交的元数据，包含 tree、父提交、作者、提交者、时间和提交说明。tag 对象主要用于附注标签，保存标签名、目标对象、打标签者、日期和说明，也可以带签名。

查看对象类型和内容：

```bash
git cat-file -t HEAD
git cat-file -p HEAD
git cat-file -t HEAD^{tree}
git cat-file -p HEAD^{tree}
```

一个提交对象并不直接保存每个文件的差异，而是指向一棵 tree。Git 在需要显示差异时，会比较两个 tree 中的路径和 blob。

### 提交是有向无环图

Git 历史是一个有向无环图。普通提交通常有一个父提交；第一次提交没有父提交；合并提交通常有两个或更多父提交。分支历史的“线性”只是特殊情况，真实协作项目通常包含并行开发、合并、回退和标签。

可以用以下命令查看历史图：

```bash
git log --oneline --graph --decorate --all
```

参数含义如下：

| 参数 | 作用 |
| --- | --- |
| `--oneline` | 每个提交压缩为一行，显示短哈希和标题 |
| `--graph` | 用 ASCII 图显示父子关系 |
| `--decorate` | 显示分支、标签、HEAD 等引用 |
| `--all` | 显示所有引用可达历史，而非只显示当前分支 |

### 分支只是可移动引用

分支不是代码副本，而是 `.git/refs/heads/` 下的一个引用，指向某个提交。当当前分支产生新提交时，该分支引用向前移动。`HEAD` 通常是一个符号引用，指向当前分支；在 detached HEAD 状态下，`HEAD` 直接指向某个提交或标签，而不是某个分支。

查看当前 HEAD：

```bash
git symbolic-ref --short HEAD
git rev-parse HEAD
```

创建分支本质上是创建一个指向当前提交的新引用：

```bash
git branch feature/login
```

切换分支后再提交，移动的是切换后的分支引用：

```bash
git switch feature/login
git commit -m "feat: implement login form"
```

### 标签是发布点标记

标签通常用于标记版本发布点。轻量标签类似一个固定引用，直接指向某个对象；附注标签是完整对象，包含打标签者、时间、说明，且可以签名。正式发布建议使用附注标签：

```bash
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

列出标签：

```bash
git tag
git tag -l "v1.*"
```

删除本地标签和远程标签：

```bash
git tag -d v1.0.0
git push origin :refs/tags/v1.0.0
```

如果项目已经公开发布，不应随意移动已发布标签。移动标签会破坏用户对版本的可验证性，也会影响包管理器、CI/CD 和部署系统。

### 引用命名空间

Git 引用通常位于以下命名空间：

| 引用 | 示例 | 含义 |
| --- | --- | --- |
| 本地分支 | `refs/heads/main` | 本地可提交分支 |
| 远程跟踪分支 | `refs/remotes/origin/main` | 最近一次 fetch 得到的远程分支状态 |
| 标签 | `refs/tags/v1.0.0` | 版本标记 |
| HEAD | `HEAD` | 当前检出位置 |

远程跟踪分支不是远程仓库上的真实分支，而是本地对远程分支状态的缓存。执行 `git fetch` 会更新这些缓存；执行 `git push` 才会把本地分支或标签推送到远程仓库。

### packfile 与垃圾回收

Git 对象最初可能以松散对象形式保存。随着对象增多，Git 会将对象压缩成 packfile，以减少磁盘占用并提升传输效率。可以手动触发垃圾回收：

```bash
git gc
```

查看仓库大小和对象情况：

```bash
git count-objects -vH
```

需要注意，`git gc` 可能清理不可达对象。误删提交后，如果 reflog 仍保留引用，通常可以恢复；如果对象已经不可达且被清理，恢复难度会显著上升。

### reflog 是本地安全网

`reflog` 记录本地引用移动历史。例如提交、重置、变基、合并、切换分支都会留下记录。误执行 `reset --hard` 后，可以先查看 reflog：

```bash
git reflog
```

然后基于历史位置恢复：

```bash
git switch -c rescue HEAD@{1}
```

reflog 是本地机制，不会随普通 push 同步到远程。因此它适合挽救本地误操作，不应替代远程备份和分支保护策略。

### 参考资料

- Git 对象与底层命令：https://git-scm.com/book/en/v2/Git-Internals-Git-Objects
- Git init 文档：https://git-scm.com/docs/git-init
- Git tag 文档：https://git-scm.com/docs/git-tag
- Git Book 标签章节：https://git-scm.com/book/en/v2/Git-Basics-Tagging
