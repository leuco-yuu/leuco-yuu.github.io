# 标签版本发布与 Release 实践

### 标签、版本号与 Release 的关系

Git 标签是版本历史中的标记。Release 是托管平台围绕标签提供的发布页面、版本说明和制品下载入口。标签偏底层，Release 偏产品交付。

发布链路通常是：

```text
提交历史 -> 版本标签 -> CI 构建 -> Release 说明 -> 上传制品 -> 用户下载或部署
```

如果项目是库，Release 可能对应包管理器版本；如果项目是服务，Release 可能对应一次可部署镜像；如果项目是桌面软件，Release 通常包含安装包和校验文件。

### 语义化版本

常用版本格式为：

```text
MAJOR.MINOR.PATCH
```

含义：

| 部分 | 何时增加 | 示例 |
| --- | --- | --- |
| MAJOR | 破坏兼容性 | `2.0.0` |
| MINOR | 向后兼容的新功能 | `1.3.0` |
| PATCH | 向后兼容的问题修复 | `1.3.2` |

预发布版本：

```text
1.4.0-alpha.1
1.4.0-beta.1
1.4.0-rc.1
```

Git 标签通常加 `v` 前缀：

```text
v1.4.0
v1.4.0-rc.1
```

团队应统一是否使用 `v` 前缀。不要在同一项目中混用 `1.0.0` 和 `v1.0.0`。

### 创建标签

附注标签：

```bash
git tag -a v1.0.0 -m "Release v1.0.0"
```

轻量标签：

```bash
git tag v1.0.0
```

正式发布建议使用附注标签。查看标签：

```bash
git tag
git tag -l "v1.*"
git show v1.0.0
```

给历史提交打标签：

```bash
git tag -a v1.0.0 <commit> -m "Release v1.0.0"
```

推送单个标签：

```bash
git push origin v1.0.0
```

推送所有标签：

```bash
git push origin --tags
```

不建议在大型仓库无筛选地频繁 `--tags`，因为可能推送临时标签或本地测试标签。

### 发布前检查清单

发布前建议执行：

```bash
git switch main
git pull --ff-only origin main
git status -sb
git log --oneline --decorate -n 10
npm test
npm run build
```

检查项：

| 检查项 | 说明 |
| --- | --- |
| 工作区干净 | `git status` 无未提交改动 |
| 主线最新 | 已同步远程主分支 |
| CI 通过 | 测试、构建、扫描通过 |
| CHANGELOG 已更新 | 版本说明完整 |
| 版本号已更新 | package、manifest、文档一致 |
| 迁移脚本已验证 | 数据库或配置变更可回滚 |
| 制品可复现 | 构建命令清晰，依赖锁定 |

### GitHub Release 发布流程

页面发布流程：

1. 进入仓库 Releases 页面。
2. 点击 Draft a new release。
3. 选择已有标签，或输入新标签。
4. 若创建新标签，选择 Target 分支或提交。
5. 填写 Release title。
6. 填写版本说明或生成发布说明。
7. 上传二进制制品、压缩包、校验文件。
8. 根据需要勾选 Pre-release。
9. 发布 Release。

CLI 示例：

```bash
gh release create v1.2.0 \
  --target main \
  --title "v1.2.0" \
  --notes-file CHANGELOG.md \
  dist/demo-service-linux-amd64.tar.gz \
  dist/checksums.txt
```

如果标签尚不存在，GitHub CLI 可基于 target 创建标签。生产项目建议先在本地创建并签名标签，再发布 Release。

### Gitee Release 发布流程

Gitee 的发行版同样围绕标签组织。建议流程：

```bash
git switch main
git pull --ff-only gitee main
git tag -a v1.2.0 -m "Release v1.2.0"
git push gitee v1.2.0
```

随后在 Gitee 仓库页面进入发行版，基于标签创建 Release，填写标题、说明并上传产物。若项目同时在 GitHub 和 Gitee 发布，应确保两个平台的标签、Release 标题、说明和附件版本一致。

### 自动化 Release

GitHub Actions 示例：当推送 `v*` 标签时构建并发布 Release。

```yaml
name: release

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - run: npm ci
      - run: npm test
      - run: npm run build
      - run: tar -czf demo-service.tar.gz dist
      - uses: softprops/action-gh-release@v2
        with:
          files: demo-service.tar.gz
```

自动化发布时，必须保护标签模式 `v*`，否则任何能推送标签的人都可能触发正式发布。

### Release 说明模板

```markdown
### Overview

Summarize the purpose of this release.

### Added

- Add ...

### Changed

- Change ...

### Fixed

- Fix ...

### Breaking changes

- Describe migration steps.

### Upgrade

```bash
example upgrade command
```

### Checksums

- demo-service-linux-amd64.tar.gz: SHA256...
```

Release 说明应面向使用者，而不是只复制提交日志。提交日志适合开发者追踪，Release 说明应强调行为变化、升级方式和风险。

### 回滚与补丁发布

如果发布后发现问题，优先判断是回滚部署还是发布补丁版本。

服务端应用通常先回滚部署到上一稳定版本，同时从 `main` 或对应 `release/*` 拉出 `hotfix/*` 修复：

```bash
git switch -c hotfix/1.2.1 v1.2.0
```

修复后：

```bash
git commit -m "fix: handle empty user profile"
git tag -a v1.2.1 -m "Release v1.2.1"
git push origin hotfix/1.2.1
git push origin v1.2.1
```

再通过 PR 把 hotfix 合回 `main`，避免后续版本丢失修复。

### 参考资料

- Git tag 文档：https://git-scm.com/docs/git-tag
- Git Book 标签章节：https://git-scm.com/book/en/v2/Git-Basics-Tagging
- GitHub Release 介绍：https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases
- GitHub 管理 Release：https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository
- GitHub 自动生成发布说明：https://docs.github.com/en/repositories/releasing-projects-on-github/automatically-generated-release-notes
- Gitee Release 帮助：https://gitee.com/help/articles/4328
- Semantic Versioning：https://semver.org
