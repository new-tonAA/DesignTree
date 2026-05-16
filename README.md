# ArchAgent — AI-Powered Architectural Design Studio

An AI image generation tool designed for architectural design, featuring design state tree management, multi-platform model switching, and style preference learning.

![Interface Preview](Readme_image/msedge_vrMMkT4GSY.jpg)

## Features

- **Design State Tree**: Visualize design iteration paths with support for rollback, branching, and node switching
- **Multi-Platform Support**: OpenAI, OpenRouter, V3.CM, Volcengine, DeepSeek
- **Multiple Model Selection**: gptimage2, gpt-image-1, BananaPro, Gemini and more
- **Style Preference Learning**: Automatically learn style preferences from selected images, intelligently recommend style tags
- **Reference Image Upload**: Drag-and-drop support for reference images, submitted together with prompts
- **Voice Input**: Real-time speech-to-text, automatically appended to input
- **Theme Switching**: Dark, Light, Midnight, Dusk themes
- **State Persistence**: Input, style selection, and tab state preserved after page refresh

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

## User Guide

### Design Workflow

1. Enter your architectural design requirements (e.g., "Modern commercial building, glass curtain wall, natural lighting")
2. Select the number of images to generate (1/2/4)
3. Click the "Generate" button and wait for images
4. Click on an image to select your preferred design
5. Continue entering modification requests to iterate based on selected images

### State Tree Panel

The left panel displays the design iteration path:
- **Design Path**: Visual tree structure, click nodes to navigate history
- **Prompt Keywords**: Keyword tags for the current path
- **Attached Images**: Reference images uploaded for the current node

### Style Panel

The system automatically learns style preferences:
- Click style tags to toggle selection
- Selection applies automatically (with debounce)
- Unselected styles will be downweighted

### Selected Images

The bottom right section shows all selected images:
- Click the magnifier to view full-size
- Selected images serve as reference for the next generation

---

## File Structure

```
ArchAgent/
├── state_manager.py   # Design state tree (core logic)
├── agent.py           # AI calls: prompt optimization / image generation / speech-to-text
├── server.py          # FastAPI backend
├── requirements.txt   # Python dependencies
├── config.json        # API keys (auto-generated)
├── sessions/          # Session JSON files (auto-saved)
├── static/
│   ├── index.html     # Frontend interface
│   └── uploads/       # Generated images
└── Readme_image/      # README screenshots
```

## State Tree Logic

```
root (initial requirement)
 └── v1 (first modification + selected image)
      └── v2 (continue iteration)  ← current
      └── v1b (branch after rollback)
```

**Core Principle**: When generating images at a leaf node, prompt = all inputs from root to current node path + selected images. Content from other branches is never mixed in.

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

Made with ❤️ for architects
