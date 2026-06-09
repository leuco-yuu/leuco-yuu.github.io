# 联系方式配置文件说明

## 目录结构
```
data/contact/
├── config.yaml    ← 所有联系方式的文字信息 + 链接配置
└── qr/            ← 放二维码图片，替换后修改 config.yaml 中 qrImage 字段
    └── (你的真实二维码图片，如 qq-qrcode.svg, wechat-qrcode.png)
```

## 如何替换

1. 编辑 `config.yaml`：把 `profileName`、`profileId`、`email`、`copy`、`url` 等字段改为你自己的真实信息
2. 二维码图片：放进 `qr/` 目录，然后在 `config.yaml` 中把对应联系人的 `qrImage` 设为 `"qr/你的文件名.svg"`
3. 头像：所有卡片默认使用站点配置的头像 (`hugo.yaml` 中的 `params.author.avatar`)

## 卡片类型说明

| type | 说明 | 显示内容 |
|------|------|---------|
| qr | 带二维码 | 头像 + 名称 + ID + 二维码 + 按钮 |
| email | 邮箱 | 邮箱地址 + 按钮 |
| profile | 资料页 | 头像 + 名称 + 链接 + 按钮 |

## 按钮配置

每个 `buttons` 数组中的按钮：
- `text`: 按钮文字
- `icon`: copy(复制) / send(发送) / external(外部链接) / chat(聊天)
- `copy`: 如果有此字段，点击按钮复制该文本
- `url`: 如果有此字段，点击按钮跳转到该链接
