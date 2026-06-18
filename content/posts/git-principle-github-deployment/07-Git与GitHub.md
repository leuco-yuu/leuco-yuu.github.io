# Git 与代码托管平台

## Git 与 Github

### GitHub 不是 Git

**GitHub 是基于 Git 的代码托管与协作平台**。

- Git 负责版本对象、分支、标签和传输协议；
- GitHub 在此基础上提供仓库页面、组织、权限、Issue、Pull Request、代码审查、Actions、Packages、Pages、Release、安全告警、规则集和保护分支等能力。

在工程实践中，应把 GitHub 看作团队协作层。一个提交能否进入主分支，往往不取决于本地 Git 是否允许，而取决于 GitHub 仓库规则是否允许。分支保护、必需评审、必需状态检查、CODEOWNERS、签名提交、线性历史等，都是平台治理能力。

### 创建 GitHub 仓库的参数

在 GitHub 创建仓库时，常见参数包括：

| 参数 | 说明 | 建议 |
| --- | --- | --- |
| Owner | 仓库归属，可以是个人或组织 | 团队项目放组织下 |
| Repository name | 仓库名 | 小写、短横线、语义明确 |
| Description | 简短描述 | 写清项目用途 |
| Visibility | Public 或 Private | 开源用 Public，内部项目用 Private |
| README | 初始化 README | 新项目可勾选；已有本地仓库不要勾选 |
| .gitignore | 初始化忽略文件 | 根据语言选择模板 |
| License | 开源许可证 | 开源项目必须明确 |
| Template | 是否作为模板仓库 | 脚手架项目可启用 |

**如果已有本地仓库准备推送到 GitHub，创建远程仓库时不要初始化 README、LICENSE、`.gitignore`**，否则本地首推时容易出现无共同历史或需要先合并远程初始化提交的情况。

本地推送到 GitHub：

```bash
git init -b main
git add .
git commit -m "chore: initial commit"
git remote add origin git@github.com:org/demo-service.git
git push -u origin main
```

### GitHub Pull Request 流程

标准 Pull Request 流程：

```bash
git fetch origin
git switch -c feature/user-profile origin/main
git add src/user-profile.ts
git commit -m "feat(user): add profile endpoint"
git push -u origin feature/user-profile
```

在 GitHub 页面创建 Pull Request 时，需要关注：

| 字段 | 说明 |
| --- | --- |
| base repository | 目标仓库 |
| base branch | 目标分支，通常为 `main` |
| head repository | 来源仓库，fork 模式下是个人仓库 |
| compare branch | 来源分支 |
| title | PR 标题，应概括变更 |
| description | 背景、方案、测试、风险、关联 Issue |
| reviewers | 审核人 |
| assignees | 负责人 |
| labels | 类型、优先级、模块 |
| milestone | 版本或迭代 |

推荐 PR 描述模板：

```markdown
### 背景

说明为什么需要这个变更。

### 方案

说明主要实现路径和关键取舍。

### 测试

- [ ] 单元测试通过
- [ ] 本地启动验证
- [ ] 已检查兼容性

### 风险

说明潜在影响和回滚方式。
```

### GitHub 保护分支

GitHub 可以为指定分支或分支模式创建保护规则。常用规则包括：

| 规则 | 作用 |
| --- | --- |
| Require a pull request before merging | 合并前必须经过 PR |
| Require approvals | 要求指定数量审批 |
| Dismiss stale pull request approvals | 新提交后使旧审批失效 |
| Require review from Code Owners | 涉及 CODEOWNERS 路径时要求所有者审批 |
| Require status checks to pass | 要求 CI、测试、扫描等状态通过 |
| Require branches to be up to date | 合并前要求基于最新目标分支 |
| Require linear history | 禁止产生 merge commit |
| Require signed commits | 要求提交签名 |
| Do not allow bypassing | 管理员也不能绕过规则 |
| Restrict who can push | 限制可推送人员、团队或应用 |

典型主分支规则：

```text
Branch name pattern: main
Require a pull request before merging: enabled
Required approvals: 1 or 2
Require status checks: build, test, lint
Require conversation resolution: enabled
Block force pushes: enabled
Block deletions: enabled
```

### CODEOWNERS

`CODEOWNERS` 用于指定路径负责人。文件位置通常为 `.github/CODEOWNERS`、根目录 `CODEOWNERS` 或 `docs/CODEOWNERS`。

示例：

```text
* @org/core-team
/docs/ @org/docs-team
/src/payments/ @org/payment-team
*.sql @org/db-team
```

当保护分支开启 code owner 审查要求时，修改匹配路径的 PR 必须获得对应负责人审批。CODEOWNERS 适合大型仓库、核心模块、合规敏感目录和数据库迁移脚本。

### GitHub Actions 与 Git

GitHub Actions 可以在 push、pull_request、release、workflow_dispatch 等事件触发。示例：

```yaml
name: ci

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - run: npm ci
      - run: npm test
```

如果分支保护要求 `test` job 通过，那么 PR 未通过测试时不能合并。这使 Git 提交历史与自动化质量门禁绑定。

### GitHub Release

GitHub Release 基于 Git 标签。标签标记历史中的某个点，Release 在平台层提供版本说明、下载入口、二进制附件和自动生成发布说明等能力。

本地创建标签并推送：

```bash
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

GitHub 页面发布 Release 时通常填写：

| 字段 | 说明 |
| --- | --- |
| Tag | 现有标签或新标签 |
| Target | 新标签指向的分支或提交 |
| Previous tag | 生成发布说明的比较起点 |
| Release title | 发布标题 |
| Description | 发布说明 |
| Assets | 上传二进制包、安装包、校验文件等 |
| Pre-release | 标记为预发布 |
| Latest release | 是否作为最新版本显示 |

GitHub CLI 发布示例：

```bash
gh release create v1.0.0 \
  --title "v1.0.0" \
  --notes-file CHANGELOG.md \
  dist/demo-service-linux-amd64.tar.gz
```

## Git 与 Gitee

### Gitee 的定位

Gitee 是面向国内开发者和企业的 Git 代码托管与研发协作平台。它与 GitHub 一样，底层围绕 Git 仓库提供托管、协作和项目治理能力；同时也提供更本土化的访问体验、企业研发管理、Gitee Pages、Gitee IDE、微信/钉钉集成、代码质量分析、轻量级 Pull Request 等能力。

从 Git 角度看，同一个本地仓库可以同时推送到 GitHub 和 Gitee。平台差异主要体现在权限、项目管理、CI/CD、镜像、Release、Pages、组织/企业能力和国内访问体验。

### 创建 Gitee 仓库的参数

Gitee 新建仓库时常见参数包括：

| 参数              | 说明                     | 建议                    |
| ----------------- | ------------------------ | ----------------------- |
| 仓库名称          | 仓库 URL 与显示名称基础  | 小写、短横线、语义明确  |
| 仓库介绍          | 简短说明项目用途         | 对外仓库必须清晰        |
| 所属空间          | 个人、组织或企业         | 团队项目放组织/企业空间 |
| 可见性            | 开源或私有               | 根据代码属性选择        |
| 初始化 README     | 创建默认说明文件         | 已有本地仓库时不要勾选  |
| 初始化 .gitignore | 根据语言生成忽略规则     | 新项目可选择模板        |
| 初始化 LICENSE    | 添加许可证               | 开源项目必须明确        |
| 分支设置          | 默认分支、保护分支等     | 主分支建议设为 `main`   |
| 成员权限          | 管理员、开发者、观察者等 | 最小权限原则            |

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

| 分支        | 策略                                       |
| ----------- | ------------------------------------------ |
| `main`      | 禁止直接推送，必须 PR，要求评审通过        |
| `release/*` | 限制推送成员，禁止强推，要求发布负责人审核 |
| `hotfix/*`  | 允许指定维护者快速处理，但保留审查记录     |

在 Gitee 中设置保护分支时，应重点关注：

| 设置项   | 说明                    |
| -------- | ----------------------- |
| 分支模式 | 具体分支或通配模式      |
| 推送权限 | 谁可以向该分支推送      |
| 合并权限 | 谁可以合并 Pull Request |
| 评审要求 | 是否必须评审、评审人数  |
| 强推限制 | 是否禁止 force push     |
| 删除限制 | 是否禁止删除分支        |

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

| 内容         | 说明                         |
| ------------ | ---------------------------- |
| 版本号       | 遵循语义化版本，如 `v1.2.0`  |
| 发布日期     | 使用明确日期                 |
| 新增功能     | 面向用户描述能力变化         |
| 修复问题     | 关联 Issue 或缺陷编号        |
| 兼容性说明   | 是否破坏兼容、是否需迁移数据 |
| 安装升级方式 | 包下载、镜像、命令、配置变更 |
| 校验信息     | SHA256、签名或校验文件       |

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

## Gitee 与 GitHub 仓库镜像同步

Gitee 支持 GitHub 仓库快速导入和同步相关能力。常见场景包括：

| 场景                   | 方案                                  |
| ---------------------- | ------------------------------------- |
| 国内用户访问 GitHub 慢 | 在 Gitee 创建镜像仓库                 |
| 开源项目国内分发       | GitHub 作为主仓库，Gitee 作为国内镜像 |
| 企业从 GitHub 迁移     | 使用导入功能后逐步切换远程地址        |
| 双平台展示             | 同步 README、Release 说明、标签       |

本地手工同步示例：

```bash
git clone --mirror git@github.com:org/demo-service.git
cd demo-service.git
git remote set-url --push origin git@gitee.com:org/demo-service.git
git push --mirror
```

`--mirror` 会同步所有引用，包括分支、标签和其他 refs。它适合迁移和镜像，但不适合随意在普通工作仓库使用，因为它可能删除目标端多出的引用。

