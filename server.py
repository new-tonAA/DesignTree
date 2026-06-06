"""
server.py — FastAPI 后端（多用户隔离版）

启动：
  uvicorn server:app --host 0.0.0.0 --port $PORT

浏览器访问 http://localhost:8000

架构说明：
- 每个用户通过 X-User-Id 头标识（浏览器 localStorage 生成 UUID）
- 会话数据、API Keys、平台配置 均按 user_id 隔离
- 图片以 base64 data URL 返回，前端存入 IndexedDB
- 服务端维护图片内存缓存，通过 /api/image/{key} 供前端回退加载
- 适配 Render 等无持久化存储的云平台（文件系统为临时存储）
"""

import os
import json
import tempfile
import time
import base64
import threading
from pathlib import Path
from collections import defaultdict
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import state_manager as sm
import agent

app = FastAPI(title="ArchAI Design Studio")

# ── CORS ──────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 路径 ──────────────────────────────────────
def get_app_dir() -> Path:
    import sys
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent
    else:
        return Path(__file__).parent

APP_DIR = get_app_dir()
STATIC_DIR = APP_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── 多用户状态隔离 ─────────────────────────────

# per-user 数据结构
_user_sessions: dict[str, dict] = {}          # user_id → session dict
_user_api_keys: dict[str, dict] = {}          # user_id → {platform: api_key}
_user_platform_config: dict[str, dict] = {}   # user_id → {platform, image_model, text_platform, text_model}
_user_last_session_path: dict[str, str] = {}  # user_id → last loaded path

# 图片内存缓存（key → data URL），所有用户共享 key 空间（key 含时间戳，不会冲突）
_image_cache: dict[str, str] = {}
_image_cache_lock = threading.Lock()


def _get_user_id(request: Request) -> str:
    """从请求头提取 user_id，如不存在则使用默认值（兼容桌面模式）"""
    uid = request.headers.get("X-User-Id", "").strip()
    if not uid:
        uid = "default"
    return uid


# 从环境变量读取默认 API Keys（部署到 Render 等云平台时使用）
_ENV_DEFAULT_KEYS = {}
for _ek, _ev in [
    ("V3_API_KEY", "v3"),
    ("OPENAI_API_KEY", "openai"),
    ("DEEPSEEK_API_KEY", "deepseek"),
    ("OPENROUTER_API_KEY", "openrouter"),
    ("VOLCENGINE_API_KEY", "volcengine"),
]:
    _val = os.environ.get(_ek, "").strip()
    if _val:
        _ENV_DEFAULT_KEYS[_ev] = _val
if _ENV_DEFAULT_KEYS:
    print(f"[server] Loaded default API keys from env: {list(_ENV_DEFAULT_KEYS.keys())}")


def _get_user_api_keys(user_id: str) -> dict:
    """获取用户的 API Keys，优先用户设置的，回退到环境变量默认值"""
    user_keys = _user_api_keys.get(user_id, {})
    merged = dict(_ENV_DEFAULT_KEYS)  # 以环境变量为基础
    merged.update(user_keys)  # 用户设置覆盖环境变量
    return merged


def _get_user_platform_config(user_id: str) -> dict:
    """获取用户的平台配置"""
    cfg = _user_platform_config.get(user_id, None)
    if cfg is None:
        # 初始化为默认配置
        cfg = {
            "platform": "v3",
            "image_model": "gpt-image-1",
            "text_platform": "v3",
            "text_model": "gpt-4o-mini",
        }
        _user_platform_config[user_id] = cfg
    return cfg


def _get_session(user_id: str) -> dict:
    """获取用户当前会话"""
    return _user_sessions.get(user_id)


def _require_session(user_id: str) -> dict:
    s = _get_session(user_id)
    if not s:
        last_path = _user_last_session_path.get(user_id, "")
        if last_path:
            try:
                s = sm.load_session(last_path, user_id)
                _user_sessions[user_id] = s
                return s
            except Exception:
                pass
        raise HTTPException(400, "请先创建或加载一个项目")
    return s


def _cache_image(key: str, data_url: str) -> None:
    """缓存图片 data URL"""
    with _image_cache_lock:
        _image_cache[key] = data_url


def _get_cached_image(key: str) -> Optional[str]:
    """从缓存获取图片 data URL"""
    with _image_cache_lock:
        return _image_cache.get(key)


# ── 页面 ──────────────────────────────────────

def get_resource_path(relative_path: str) -> Path:
    import sys
    if getattr(sys, 'frozen', False):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).parent
    return base_path / relative_path

@app.get("/", response_class=HTMLResponse)
def index():
    return get_resource_path("static/index.html").read_text(encoding="utf-8")


# ── 图片服务 ──────────────────────────────────

@app.get("/api/image/{key}")
def serve_image(key: str):
    """从内存缓存提供图片（供前端 <img src> 使用）"""
    data_url = _get_cached_image(key)
    if not data_url:
        raise HTTPException(404, "Image not found or expired (server may have restarted)")
    try:
        header, b64_data = data_url.split(",", 1)
        content_type = header.split(":")[1].split(";")[0]
        return Response(content=base64.b64decode(b64_data), media_type=content_type)
    except Exception:
        raise HTTPException(500, "Failed to decode cached image")


# ── 项目管理 ─────────────────────────────────

class NewProjectReq(BaseModel):
    project_name: str

@app.post("/api/project/new")
def new_project(req: NewProjectReq, user_id: str = Depends(_get_user_id)):
    session = sm.new_session(req.project_name, user_id)
    _user_sessions[user_id] = session
    _user_last_session_path[user_id] = session["save_path"]
    return {"ok": True, "save_path": session["save_path"]}

@app.post("/api/project/load")
def load_project(path: str = Form(...), user_id: str = Depends(_get_user_id)):
    try:
        session = sm.load_session(path, user_id)
        _user_sessions[user_id] = session
        _user_last_session_path[user_id] = path
    except Exception as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "project": session["project"]}

@app.get("/api/project/list")
def list_projects(user_id: str = Depends(_get_user_id)):
    return sm.list_sessions(user_id)

class RenameProjectReq(BaseModel):
    path: str
    new_name: str

@app.post("/api/project/rename")
def rename_project(req: RenameProjectReq, user_id: str = Depends(_get_user_id)):
    try:
        s = sm.load_session(req.path, user_id)
        s["project"] = req.new_name
        sm._save(s)
        if _user_sessions.get(user_id, {}).get("save_path") == req.path:
            _user_sessions[user_id]["project"] = req.new_name
        return {"ok": True}
    except Exception as e:
        raise HTTPException(400, str(e))

class DeleteProjectReq(BaseModel):
    path: str

@app.post("/api/project/delete")
def delete_project(req: DeleteProjectReq, user_id: str = Depends(_get_user_id)):
    try:
        if _user_sessions.get(user_id, {}).get("save_path") == req.path:
            _user_sessions.pop(user_id, None)
            _user_last_session_path.pop(user_id, None)
        os.remove(req.path)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(400, str(e))

@app.get("/api/project/state")
def get_state(user_id: str = Depends(_get_user_id)):
    s = _require_session(user_id)
    cur = s["nodes"][s["current_node"]]
    current_selected_urls = sm.get_node_selected_urls(s)
    current_attachments = sm.get_node_attachments(s)
    current_path_node_ids = [n["id"] for n in sm.get_path_to_root(s)]

    # 将图片数据替换为轻量引用（key + 服务端缓存URL）
    current_images = _lightweight_images(cur["images"])
    current_attachments_light = _lightweight_images(current_attachments)

    return {
        "project":       s["project"],
        "current_node":  s["current_node"],
        "current_path_node_ids": current_path_node_ids,
        "tree":          sm.get_tree_for_ui(s),
        "style_summary": sm.get_style_summary(s),
        "style_candidates": sm.get_style_candidates(s),
        "ref_images":    s["reference_images"],
        "save_path":     s["save_path"],
        "history":       _history_for_ui(s),
        "path_tags":     _path_tags(s),
        "current_images": current_images,
        "current_prompt": cur["prompt"],
        "current_selected": cur.get("selected"),
        "current_selecteds": current_selected_urls,
        "current_attached_images": current_attachments_light,
        "current_attachments": current_attachments_light,
        "generating":    cur.get("generating", False),
    }


def _lightweight_images(images: list) -> list:
    """将图片列表中的 data URL 替换为服务端缓存 URL，减少 state API 响应体积"""
    result = []
    for img in images:
        if not isinstance(img, dict) or not img.get("key"):
            result.append(img)
            continue
        key = img["key"]
        # 使用服务端缓存 URL
        cache_url = f"/api/image/{key}"
        light_img = {k: v for k, v in img.items() if k != "url"}
        light_img["url"] = cache_url
        result.append(light_img)
    return result


# ── 核心：生成图片 ───────────────────────────

class GenerateReq(BaseModel):
    user_input: str
    n: int = 4
    parent_node_id: Optional[str] = None
    optimize_prompt: bool = True
    model_memory: Optional[str] = None
    prompt_images: Optional[list[dict]] = None

class PolishReq(BaseModel):
    user_input: str

@app.post("/api/polish_prompt")
def polish_prompt(req: PolishReq, user_id: str = Depends(_get_user_id)):
    s = _require_session(user_id)
    api_keys = _get_user_api_keys(user_id)
    pcfg = _get_user_platform_config(user_id)

    try:
        context = sm.build_context_for_agent(s, req.user_input, None)
    except ValueError as e:
        raise HTTPException(400, str(e))
    try:
        polished = agent.prompt_agent(
            context,
            api_keys=api_keys,
            text_platform=pcfg.get("text_platform"),
            text_model=pcfg.get("text_model"),
        )
        return {"polished_prompt": polished, "optimized": True, "warning": None}
    except Exception as e:
        polished = agent.compose_prompt_from_context(context)
        return {
            "polished_prompt": polished,
            "optimized": False,
            "warning": f"AI润色失败，已切换为路径记忆拼接：{e}",
        }

@app.post("/api/generate")
def generate(req: GenerateReq, user_id: str = Depends(_get_user_id)):
    s = _require_session(user_id)
    api_keys = _get_user_api_keys(user_id)
    pcfg = _get_user_platform_config(user_id)

    if req.parent_node_id and req.parent_node_id not in s["nodes"]:
        raise HTTPException(400, f"节点 {req.parent_node_id} 不存在")
    base_node_id = req.parent_node_id or s["current_node"]

    # 创建节点
    try:
        node = sm.add_node(s, req.user_input, req.parent_node_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    node_id = node["id"]

    # 保存用户上传的附带图片（data URL → 缓存 + key 引用）
    attachments = _save_prompt_attachments(req.prompt_images or [])
    if attachments:
        sm.add_attachments(s, node_id, attachments)

    try:
        context = sm.build_context_for_agent(s, req.user_input, base_node_id)
    except ValueError as e:
        raise HTTPException(400, str(e))

    # 将新节点上传图加入上下文
    path_attachments = list(context.get("path_attachments") or [])
    for att in attachments:
        if isinstance(att, dict) and att.get("url"):
            path_attachments.append({
                "node_id": node_id,
                "url": att.get("url"),
                "name": att.get("name", ""),
            })
    context["path_attachments"] = path_attachments

    if req.model_memory:
        context["model_memory"] = req.model_memory

    prompt_warning = None
    if req.optimize_prompt:
        try:
            prompt = agent.prompt_agent(
                context,
                api_keys=api_keys,
                text_platform=pcfg.get("text_platform"),
                text_model=pcfg.get("text_model"),
            )
        except Exception as e:
            prompt = agent.compose_prompt_from_context(context)
            prompt_warning = f"AI润色失败，已切换为路径记忆拼接：{e}"
            print(f"[warn] prompt_agent failed: {e}")
    else:
        prompt = agent.compose_prompt_from_context(context)

    sm.set_node_prompt(s, node_id, prompt)

    # 逐张生成图片
    def on_image_ready(img_dict):
        if img_dict.get("key") and img_dict.get("url"):
            # 缓存图片到内存
            _cache_image(img_dict["key"], img_dict["url"])
            # 存入 session（存 key + 缓存URL，不存完整 data URL）
            light_img = {
                "key": img_dict["key"],
                "url": f"/api/image/{img_dict['key']}",
                "revised_prompt": img_dict.get("revised_prompt", prompt),
            }
            sm.add_images(s, node_id, [light_img])

    images = agent.generate_images(
        prompt, n=req.n,
        api_keys=api_keys,
        platform_key=pcfg.get("platform"),
        image_model=pcfg.get("image_model"),
        on_image_ready=on_image_ready,
    )

    ok_images = [
        img for img in images
        if isinstance(img, dict) and img.get("key")
    ]
    if not ok_images:
        _cleanup_attachment_cache(attachments)
        sm.remove_node(s, node_id)
        first_error = next(
            (img.get("error") for img in images if isinstance(img, dict) and img.get("error")),
            "未生成任何可用图片"
        )
        raise HTTPException(502, f"本次生成失败：{first_error}")

    s["nodes"][node_id]["generating"] = False
    sm._save(s)

    # 返回完整 data URL 给前端（前端存入 IndexedDB）
    return {
        "node_id": node_id,
        "optimized_prompt": prompt,
        "prompt_optimized": req.optimize_prompt,
        "prompt_warning": prompt_warning,
        "context_path_node_ids": context.get("path_node_ids", []),
        "attachments": attachments,
        "images": [
            {
                "key": img.get("key"),
                "url": img.get("url"),  # 完整 data URL，前端存 IndexedDB
                "revised_prompt": img.get("revised_prompt", prompt),
            }
            for img in ok_images
        ],
    }


# ── 选图 ─────────────────────────────────────

class SelectImageReq(BaseModel):
    image_url: str

@app.post("/api/select_image")
def select_image(req: SelectImageReq, user_id: str = Depends(_get_user_id)):
    s = _require_session(user_id)
    try:
        sm.select_image(s, req.image_url)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "style_summary": sm.get_style_summary(s)}


class StyleSelectReq(BaseModel):
    candidate_keywords: list[str]
    selected_keywords: list[str]

@app.post("/api/style/select")
def style_select(req: StyleSelectReq, user_id: str = Depends(_get_user_id)):
    s = _require_session(user_id)
    sm.apply_style_selection(s, req.candidate_keywords, req.selected_keywords)
    return {
        "ok": True,
        "style_summary": sm.get_style_summary(s),
        "style_candidates": sm.get_style_candidates(s),
    }


# ── 切换节点 ─────────────────────────────────

class SwitchNodeReq(BaseModel):
    node_id: str

@app.post("/api/switch_node")
def switch_node(req: SwitchNodeReq, user_id: str = Depends(_get_user_id)):
    s = _require_session(user_id)
    try:
        sm.switch_node(s, req.node_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "current_node": s["current_node"]}


# ── 语音转文字 ───────────────────────────────

@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...), user_id: str = Depends(_get_user_id)):
    api_keys = _get_user_api_keys(user_id)
    pcfg = _get_user_platform_config(user_id)
    suffix = Path(audio.filename or "audio.webm").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        text = agent.transcribe(
            tmp_path,
            api_keys=api_keys,
            platform_key=pcfg.get("platform"),
        )
    finally:
        os.unlink(tmp_path)
    return {"text": text}


# ── 上传参考图 ───────────────────────────────

@app.post("/api/upload_reference")
async def upload_reference(file: UploadFile = File(...), label: str = Form(""), user_id: str = Depends(_get_user_id)):
    s = _require_session(user_id)
    content = await file.read()
    b64_data = base64.b64encode(content).decode("utf-8")

    # 检测 content type
    import imghdr
    img_type = imghdr.what(None, h=content) or "png"
    content_type = f"image/{img_type}"
    data_url = f"data:{content_type};base64,{b64_data}"

    # 生成 key 并缓存
    ts = int(time.time() * 1000)
    key = f"ref_{ts}"
    _cache_image(key, data_url)

    cache_url = f"/api/image/{key}"
    sm.add_reference_image(s, cache_url, label or file.filename)
    return {"ok": True, "url": cache_url, "key": key, "data_url": data_url}


# ── 保存 API Key ─────────────────────────────

class SetApiKeysReq(BaseModel):
    openai: Optional[str] = None
    openrouter: Optional[str] = None
    v3: Optional[str] = None
    deepseek: Optional[str] = None
    volcengine: Optional[str] = None

@app.post("/api/set_api_keys")
def set_api_keys(req: SetApiKeysReq, user_id: str = Depends(_get_user_id)):
    """设置用户的 API Keys"""
    keys = _user_api_keys.get(user_id, {})
    if req.openai:
        keys["openai"] = agent._sanitize_api_key(req.openai)
    if req.openrouter:
        keys["openrouter"] = agent._sanitize_api_key(req.openrouter)
    if req.v3:
        keys["v3"] = agent._sanitize_api_key(req.v3)
    if req.deepseek:
        keys["deepseek"] = agent._sanitize_api_key(req.deepseek)
    if req.volcengine:
        keys["volcengine"] = agent._sanitize_api_key(req.volcengine)
    _user_api_keys[user_id] = keys
    return {"ok": True}


# ── 平台选择 ───────────────────────────────

class SetPlatformReq(BaseModel):
    platform: Optional[str] = None
    image_model: Optional[str] = None
    text_model: Optional[str] = None
    text_platform: Optional[str] = None

@app.get("/api/platforms")
def get_platforms():
    return agent.get_platforms()

@app.post("/api/set_platform")
def set_platform(req: SetPlatformReq, user_id: str = Depends(_get_user_id)):
    cfg = _get_user_platform_config(user_id)
    try:
        if req.platform:
            if req.platform not in agent.PLATFORMS:
                raise ValueError(f"不支持的平台: {req.platform}")
            cfg["platform"] = req.platform
        if req.image_model:
            normalized = agent._normalize_image_model(cfg["platform"], req.image_model)
            supported = agent.PLATFORMS[cfg["platform"]]["models"].get("image", [])
            if normalized not in supported:
                raise ValueError(f"平台 {cfg['platform']} 不支持图像模型: {normalized}")
            cfg["image_model"] = normalized
        if req.text_platform:
            if req.text_platform not in agent.PLATFORMS:
                raise ValueError(f"不支持的文本平台: {req.text_platform}")
            cfg["text_platform"] = req.text_platform
        if req.text_model:
            text_supported = agent.PLATFORMS[cfg["text_platform"]]["models"].get("text", [])
            if req.text_model not in text_supported:
                raise ValueError(f"平台 {cfg['text_platform']} 不支持文本模型: {req.text_model}")
            cfg["text_model"] = req.text_model
        _user_platform_config[user_id] = cfg
        return {"ok": True, "config": cfg}
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.get("/api/current_platform")
def get_current_platform(user_id: str = Depends(_get_user_id)):
    cfg = _get_user_platform_config(user_id)
    return cfg


# ── 会话备份/恢复 ────────────────────────────

class RestoreSessionReq(BaseModel):
    session_data: dict

@app.post("/api/project/restore")
def restore_session(req: RestoreSessionReq, user_id: str = Depends(_get_user_id)):
    """从前端 IndexedDB 恢复会话到服务端内存"""
    s = req.session_data
    if not isinstance(s, dict) or "nodes" not in s:
        raise HTTPException(400, "无效的会话数据")
    _user_sessions[user_id] = s
    if s.get("save_path"):
        _user_last_session_path[user_id] = s["save_path"]
        # 也保存到磁盘（如果可以的话）
        try:
            sm._save(s)
        except Exception:
            pass
    return {"ok": True}


# ── 辅助格式化 ───────────────────────────────

def _history_for_ui(s: dict) -> list:
    out = []
    for n in s["nodes"].values():
        selected_images = []
        selected_urls = []
        if isinstance(n.get("selected_list"), list):
            selected_urls = [u for u in n.get("selected_list", []) if isinstance(u, str) and u]
        elif n.get("selected"):
            selected_urls = [n.get("selected")]

        if selected_urls:
            by_url = {
                img.get("url"): img
                for img in n.get("images", [])
                if isinstance(img, dict) and img.get("url")
            }
            selected_images = [by_url[u] for u in selected_urls if u in by_url]
        selected_image = selected_images[0] if selected_images else None

        attachments = []
        if isinstance(n.get("attachments"), list):
            attachments = [img for img in n.get("attachments", []) if isinstance(img, dict) and img.get("url")]

        out.append({
            "id":         n["id"],
            "user_input": n["user_input"],
            "prompt":     n["prompt"],
            "selected":   bool(selected_urls),
            "selected_count": len(selected_images),
            "selected_urls": selected_urls,
            "selected_images": selected_images,
            "selected_image": selected_image,
            "images":     len(n["images"]),
            "is_current": n["id"] == s["current_node"],
            "parent":     n["parent"],
            "attachments": attachments,
        })
    return out


def _save_prompt_attachments(prompt_images: list[dict]) -> list[dict]:
    """将前端发来的 data URL 图片转为缓存引用"""
    out: list[dict] = []

    for i, item in enumerate(prompt_images or []):
        if not isinstance(item, dict):
            continue
        raw_url = str(item.get("url") or "").strip()
        if not raw_url:
            continue

        # 已经是缓存 URL
        if raw_url.startswith("/api/image/"):
            key = raw_url.split("/api/image/")[-1]
            out.append({"url": raw_url, "key": key, "name": item.get("name", "")})
            continue

        if not raw_url.startswith("data:image"):
            continue

        # 缓存 data URL
        ts = int(time.time() * 1000)
        key = f"attach_{ts}_{i}"
        _cache_image(key, raw_url)

        cache_url = f"/api/image/{key}"
        out.append({
            "url": cache_url,
            "key": key,
            "name": item.get("name", ""),
        })

    return out


def _cleanup_attachment_cache(attachments: list[dict]) -> None:
    """清理附图缓存"""
    for item in attachments or []:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        if key:
            with _image_cache_lock:
                _image_cache.pop(key, None)


def _path_tags(s: dict) -> list:
    path = sm.get_path_to_root(s)
    cur_id = s["current_node"]
    tags = []
    for node in path:
        words = [w for w in node["user_input"].replace("，", " ").replace(",", " ").split() if len(w) >= 2]
        for w in words[:3]:
            tags.append({"text": w, "is_new": node["id"] == cur_id})
    return tags


if __name__ == "__main__":
    import webbrowser
    import socket

    def find_available_port(start_port=8000, max_attempts=10):
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("0.0.0.0", port))
                    return port
            except OSError:
                continue
        return start_port + max_attempts

    port = find_available_port()

    def open_browser():
        import threading
        time.sleep(1.5)
        webbrowser.open(f"http://localhost:{port}")

    threading.Thread(target=open_browser, daemon=True).start()

    print(f"\n  ArchAI Design Studio → http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
