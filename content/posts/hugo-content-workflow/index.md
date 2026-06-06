---
title: "Hugo 内容工作流：从草稿到发布"
date: 2026-05-18T09:15:00+08:00
lastmod: 2026-05-18T09:15:00+08:00
draft: false
slug: hugo-content-workflow
homepage: true
series_order: 1
description: "用 Hugo Narrow 组织文章、项目、分类、标签和系列的一套写作工作流。"
summary: "把 Hugo 的内容目录、front matter 和 GitHub Pages 发布流程串起来，减少写作之外的摩擦。"
tags:
  - Hugo
  - GitHub Pages
  - 写作系统
  - 静态站点
categories:
  - 写作系统
  - Hugo
series:
  - 博客建设
cover: cover.jpg
---

Hugo 的好处是内容和呈现分离。只要主题约定清楚，写作时就可以专注在 Markdown 和 front matter，不需要每篇文章都碰布局。

这套站点采用 Hugo Narrow，内容主要分成两类：`posts` 写文章，`projects` 放项目。分类、标签和系列都由 front matter 自动生成。

## 新建文章

```powershell
hugo new content posts/article-slug/index.md
```

文章使用 page bundle，可以把配图、附件和 `index.md` 放在同一个目录。封面图使用 `cover` 字段，也可以引用站点静态目录中的图片。

## Front Matter

一篇文章最常用的字段如下：

```yaml
title: "文章标题"
date: 2026-05-18T09:15:00+08:00
draft: false
slug: article-slug
summary: "列表页摘要"
tags:
  - Hugo
categories:
  - 写作系统
series:
  - 博客建设
cover: /images/placeholder/3.jpg
```

`categories` 用来表达主题归属，`tags` 用来标细节关键词，`series` 用来组织连续阅读路径。三者不要承担同一件事，否则后续归档会变得混乱。

## 发布前检查

发布前我会做两步：

```powershell
hugo --gc --minify
hugo server --buildDrafts
```

第一步确认静态构建没有错误，第二步在浏览器里检查封面、目录、代码块、分类链接和系列页。

## 小结

好的博客工作流应该让写作动作轻，发布动作稳。目录结构和 front matter 固定下来之后，后续新增文章就只是把内容放到正确位置。
