<div align="center">

# 🌳 DesignTree

**AI-Powered Architectural Design Studio**

An intelligent image generation tool for architectural design with visual design tree management

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)

</div>

---

![Interface Preview](Readme_image/msedge_vrMMkT4GSY.jpg)

## Features

- **🌳 Visual Design Tree** - Visualize design iteration paths with rollback, branching, and node switching
- **🔄 Multi-Platform Support** - OpenAI, OpenRouter, V3.CM, Volcengine, DeepSeek
- **🎨 Multiple Models** - gptimage2, gpt-image-1, BananaPro, Gemini and more
- **🧠 Style Learning** - Automatically learn style preferences from selected images
- **📤 Reference Images** - Drag-and-drop upload, submitted together with prompts
- **🎤 Voice Input** - Real-time speech-to-text with auto-append
- **🎨 Theme Switching** - Dark, Light, Midnight, Dusk themes
- **💾 State Persistence** - Input, style selection, and tab state preserved after refresh

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server
python server.py

# 3. Open in browser
http://localhost:8000
```

## API Key Configuration

Choose one of the following methods:

**Method 1: Environment Variable**
```bash
export OPENAI_API_KEY=sk-...
python server.py
```

**Method 2: Settings in UI**
Open http://localhost:8000, click the "Settings" button in the top right corner, and enter API keys for each platform.

---

## How It Works

### Design Tree Logic

```
root (initial requirement)
 └── v1 (first modification + selected image)
      ├── v2 (continue iteration)  ← current
      └── v1b (branch after rollback)
```

**Core Principle**: 
- When generating images at a leaf node, the prompt = all inputs from **root → current node path** + selected images
- Content from other branches is **NEVER** mixed in
- Only the path matters, not sibling nodes

### What's Included in Context

| Source | Included? | Notes |
|--------|-----------|-------|
| Path: root → current | ✅ Yes | All user inputs and prompts |
| Selected images on path | ✅ Yes | Images chosen at each node |
| Uploaded images on path | ✅ Yes | Reference images attached to nodes |
| Sibling nodes (v7, v8, v9) | ❌ No | Completely isolated |
| Other branches | ❌ No | No cross-branch contamination |

---

## User Guide

### Design Workflow

1. Enter your architectural design requirements
2. Select the number of images to generate (1/2/4)
3. Click "Generate" and wait for images
4. Click on an image to select your preferred design
5. Continue entering modifications to iterate

### State Tree Panel

The left panel displays the design iteration path:
- **Design Path**: Visual tree structure, click nodes to navigate
- **Prompt Keywords**: Keyword tags for the current path
- **Attached Images**: Reference images uploaded for the current node

### Style Panel

The system automatically learns style preferences:
- Click style tags to toggle selection
- Selection applies automatically (with debounce)
- Unselected styles will be downweighted

---

## File Structure

```
DesignTree/
├── state_manager.py   # Design tree core logic
├── agent.py           # AI calls: prompt optimization / image generation
├── server.py          # FastAPI backend
├── requirements.txt   # Python dependencies
├── config.json        # API keys (auto-generated)
├── sessions/          # Session JSON files (auto-saved)
├── static/
│   ├── index.html     # Frontend interface
│   └── uploads/       # Generated images
└── Readme_image/      # README screenshots
```

## Supported Platforms and Models

| Platform | Image Models | Text Models |
|----------|--------------|-------------|
| V3.CM | gptimage2, gptimage3 | gpt-4o-mini |
| OpenAI | gpt-image-1, gpt-image-2 | gpt-4o |
| OpenRouter | BananaPro, Gemini | Claude, Gemini |
| Volcengine | volcengine-image | - |
| DeepSeek | - | deepseek-chat |

## API Dependencies

| Function | API |
|----------|-----|
| Prompt Optimization | GPT-4o-mini / DeepSeek |
| Image Generation | gptimage2 / OpenAI DALL-E |
| Speech-to-Text | Whisper-1 |

## Tech Stack

- **Backend**: Python + FastAPI
- **Frontend**: Vanilla HTML/CSS/JS (no framework)
- **AI**: OpenAI API / OpenRouter API / V3.CM API

---

<div align="center">

Made with ❤️ for architects

**DesignTree** - Grow your designs like a tree 🌳

</div>
