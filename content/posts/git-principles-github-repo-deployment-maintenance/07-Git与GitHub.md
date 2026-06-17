# Git 与 GitHub

### GitHub 不是 Git

GitHub 是基于 Git 的代码托管与协作平台。Git 负责版本对象、分支、标签和传输协议；GitHub 在此基础上提供仓库页面、组织、权限、Issue、Pull Request、代码审查、Actions、Packages、Pages、Release、安全告警、规则集和保护分支等能力。

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

如果已有本地仓库准备推送到 GitHub，创建远程仓库时不要初始化 README、LICENSE、`.gitignore`，否则本地首推时容易出现无共同历史或需要先合并远程初始化提交的情况。

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

### 参考资料

- GitHub 创建仓库：https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository
- GitHub 添加本地仓库：https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github
- GitHub 保护分支：https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches
- GitHub CODEOWNERS：https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners
- GitHub Release：https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases
- GitHub 管理 Release：https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository
