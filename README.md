# ArchAI — 建筑设计 AI 生成工作台

## 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动服务
python server.py

# 3. 浏览器打开
http://localhost:8000
```

## 配置 API Key

两种方式任选其一：

**方式1：环境变量**
```bash
export OPENAI_API_KEY=sk-...
python server.py
```

**方式2：启动后在界面输入**
打开 http://localhost:8000，在欢迎页的 API Key 栏输入，点击「开始设计」自动保存到 `config.json`。

---

## 文件结构

```
arch_ai/
├── state_manager.py   # 设计状态树（核心）
├── agent.py           # AI调用：prompt优化 / 图像生成 / 语音转文字
├── server.py          # FastAPI后端
├── requirements.txt
├── config.json        # API Key（自动生成，勿提交到git）
├── sessions/          # 会话JSON文件（自动保存）
└── static/
    ├── index.html     # 前端界面
    └── uploads/       # 生成的图片 + 参考图
```

## 状态树逻辑

```
root（初始需求）
 └── v1（用户第一次修改 + 选图）
      └── v2（继续迭代）  ← 当前
      └── v1b（回退后另起分支）
```

**叶子节点生成图片时，prompt = 从 root 到当前节点路径上所有输入的整合**，
由 GPT-4o Agent 自动优化为英文 DALL-E 3 prompt，不丢失历史设计决策。

## 依赖的API

| 功能 | API |
|------|-----|
| Prompt优化 | GPT-4o |
| 图像生成 | DALL-E 3 |
| 语音转文字 | Whisper-1 |
