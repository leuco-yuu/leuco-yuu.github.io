# Git 与 Gitee

### Gitee 的定位

Gitee 是面向国内开发者和企业的 Git 代码托管与研发协作平台。它与 GitHub 一样，底层围绕 Git 仓库提供托管、协作和项目治理能力；同时也提供更本土化的访问体验、企业研发管理、Gitee Pages、Gitee IDE、微信/钉钉集成、代码质量分析、轻量级 Pull Request 等能力。

从 Git 角度看，同一个本地仓库可以同时推送到 GitHub 和 Gitee。平台差异主要体现在权限、项目管理、CI/CD、镜像、Release、Pages、组织/企业能力和国内访问体验。

### 创建 Gitee 仓库的参数

Gitee 新建仓库时常见参数包括：

| 参数 | 说明 | 建议 |
| --- | --- | --- |
| 仓库名称 | 仓库 URL 与显示名称基础 | 小写、短横线、语义明确 |
| 仓库介绍 | 简短说明项目用途 | 对外仓库必须清晰 |
| 所属空间 | 个人、组织或企业 | 团队项目放组织/企业空间 |
| 可见性 | 开源或私有 | 根据代码属性选择 |
| 初始化 README | 创建默认说明文件 | 已有本地仓库时不要勾选 |
| 初始化 .gitignore | 根据语言生成忽略规则 | 新项目可选择模板 |
| 初始化 LICENSE | 添加许可证 | 开源项目必须明确 |
| 分支设置 | 默认分支、保护分支等 | 主分支建议设为 `main` |
| 成员权限 | 管理员、开发者、观察者等 | 最小权限原则 |

如果从本地已有项目推送到 Gitee：

```bash
git remote add gitee git@gitee.com:org/demo-service.git
git push -u gitee main
```

如果同一仓库同时维护 GitHub 和 Gitee：

```bash
git remote add origin git@github.com:org/demo-service.git
git remote add gitee git@gitee.com:org/demo-service.git

git push origin main
git push gitee main
```

也可以给同一个远程配置多个 push URL：

```bash
git remote set-url origin git@github.com:org/demo-service.git
git remote set-url --add --push origin git@github.com:org/demo-service.git
git remote set-url --add --push origin git@gitee.com:org/demo-service.git
```

之后执行：

```bash
git push origin main
```

会推送到配置的多个 push URL。此方式适合双平台同步，但要注意某个平台失败时需要检查两端状态是否一致。

### Gitee 与 GitHub 仓库镜像

Gitee 支持 GitHub 仓库快速导入和同步相关能力。常见场景包括：

| 场景 | 方案 |
| --- | --- |
| 国内用户访问 GitHub 慢 | 在 Gitee 创建镜像仓库 |
| 开源项目国内分发 | GitHub 作为主仓库，Gitee 作为国内镜像 |
| 企业从 GitHub 迁移 | 使用导入功能后逐步切换远程地址 |
| 双平台展示 | 同步 README、Release 说明、标签 |

本地手工同步示例：

```bash
git clone --mirror git@github.com:org/demo-service.git
cd demo-service.git
git remote set-url --push origin git@gitee.com:org/demo-service.git
git push --mirror
```

`--mirror` 会同步所有引用，包括分支、标签和其他 refs。它适合迁移和镜像，但不适合随意在普通工作仓库使用，因为它可能删除目标端多出的引用。

### Gitee Pull Request 与轻量级 PR

Gitee 支持 Fork + Pull Request 模式，也提供轻量级 Pull Request。轻量级 PR 的目标是降低开源贡献门槛，贡献者无需传统 fork 流程也可以更快速提交变更。团队可以根据仓库开放程度选择模式。

标准分支 PR 流程：

```bash
git fetch gitee
git switch -c feature/order-query gitee/main
git commit -m "feat(order): add query endpoint"
git push -u gitee feature/order-query
```

随后在 Gitee 页面创建 Pull Request，填写目标分支、标题、说明、审查人、关联 Issue 等。

### Gitee 保护分支

Gitee 提供保护分支能力，用于控制重要分支的推送、合并和评审流程。推荐保护：

| 分支 | 策略 |
| --- | --- |
| `main` | 禁止直接推送，必须 PR，要求评审通过 |
| `release/*` | 限制推送成员，禁止强推，要求发布负责人审核 |
| `hotfix/*` | 允许指定维护者快速处理，但保留审查记录 |

在 Gitee 中设置保护分支时，应重点关注：

| 设置项 | 说明 |
| --- | --- |
| 分支模式 | 具体分支或通配模式 |
| 推送权限 | 谁可以向该分支推送 |
| 合并权限 | 谁可以合并 Pull Request |
| 评审要求 | 是否必须评审、评审人数 |
| 强推限制 | 是否禁止 force push |
| 删除限制 | 是否禁止删除分支 |

保护分支适合与 CodeOwners、CI/CD、WebHook 一起使用，把主线写入权限从“人治”变为“规则治理”。

### Gitee Release

Release 是面向用户的版本发布入口。Git 标签用于标记历史位置，Release 用于展示版本说明、上传制品、组织下载入口。

本地标签流程：

```bash
git switch main
git pull --ff-only gitee main
git tag -a v1.0.0 -m "Release v1.0.0"
git push gitee v1.0.0
```

在 Gitee 页面创建发行版时，通常需要选择标签、填写发行版标题、版本说明，并上传构建产物。建议 Release 说明包含：

| 内容 | 说明 |
| --- | --- |
| 版本号 | 遵循语义化版本，如 `v1.2.0` |
| 发布日期 | 使用明确日期 |
| 新增功能 | 面向用户描述能力变化 |
| 修复问题 | 关联 Issue 或缺陷编号 |
| 兼容性说明 | 是否破坏兼容、是否需迁移数据 |
| 安装升级方式 | 包下载、镜像、命令、配置变更 |
| 校验信息 | SHA256、签名或校验文件 |

### Gitee Pages 与文档站点

Gitee Pages 可用于托管静态页面。适用场景包括项目文档、组件演示、API 文档、博客和官网。常见静态站点生成器包括 VitePress、VuePress、Docusaurus、Hexo、Hugo、Jekyll。

静态文档仓库建议结构：

```text
docs/
  index.md
  guide/
  api/
package.json
README.md
```

发布文档前，应在仓库 README 中明确文档源目录、构建命令和部署分支。

### 参考资料

- Gitee 官方站点：https://gitee.com
- Gitee 帮助中心：https://help.gitee.com
- Gitee 功能对比：https://gitee.com/contrast
- Gitee 新建仓库帮助：https://gitee.com/help/articles/4169
- Gitee 保护分支帮助：https://gitee.com/help/articles/4239
- Gitee Release 帮助：https://gitee.com/help/articles/4328
- Gitee GitHub 导入与同步帮助：https://gitee.com/help/articles/4284
