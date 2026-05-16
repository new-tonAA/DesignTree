"""
server.py — FastAPI 后端

启动：
  cd arch_ai
  python server.py

浏览器访问 http://localhost:8000
"""

import os
import json
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

import state_manager as sm
import agent

app = FastAPI(title="ArchAI Design Studio")

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
(STATIC_DIR / "uploads").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# 内存单会话（桌面应用场景足够）
_session: dict = {}
_last_session_path: str = ""  # 记录上次加载的路径


def _auto_load_latest_session():
    """启动时自动加载最近的会话"""
    global _session, _last_session_path
    sessions = sm.list_sessions()
    if sessions:
        latest = sessions[0]["path"]
        try:
            _session = sm.load_session(latest)
            _last_session_path = latest
            print(f"[info] Auto-loaded session: {latest}")
        except Exception as e:
            print(f"[warn] Failed to auto-load session: {e}")


# 启动时自动加载
_auto_load_latest_session()


def _require_session():
    global _session, _last_session_path
    if not _session:
        # 尝试自动恢复
        if _last_session_path and Path(_last_session_path).exists():
            try:
                _session = sm.load_session(_last_session_path)
                return _session
            except:
                pass
        raise HTTPException(400, "请先创建或加载一个项目")
    return _session


# ── 页面 ─────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


# ── 项目管理 ─────────────────────────────────

class NewProjectReq(BaseModel):
    project_name: str

@app.post("/api/project/new")
def new_project(req: NewProjectReq):
    global _session, _last_session_path
    _session = sm.new_session(req.project_name)
    _last_session_path = _session["save_path"]
    return {"ok": True, "save_path": _session["save_path"]}

@app.post("/api/project/load")
def load_project(path: str = Form(...)):
    global _session, _last_session_path
    try:
        _session = sm.load_session(path)
        _last_session_path = path
    except Exception as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "project": _session["project"]}

@app.get("/api/project/list")
def list_projects():
    return sm.list_sessions()

class RenameProjectReq(BaseModel):
    path: str
    new_name: str

@app.post("/api/project/rename")
def rename_project(req: RenameProjectReq):
    """重命名项目"""
    global _session, _last_session_path
    try:
        s = sm.load_session(req.path)
        s["project"] = req.new_name
        sm._save(s)
        # 如果是当前会话，更新内存
        if _session and _session.get("save_path") == req.path:
            _session["project"] = req.new_name
        return {"ok": True}
    except Exception as e:
        raise HTTPException(400, str(e))

class DeleteProjectReq(BaseModel):
    path: str

@app.post("/api/project/delete")
def delete_project(req: DeleteProjectReq):
    """删除项目"""
    global _session, _last_session_path
    try:
        import os
        # 如果删除的是当前会话，清空内存
        if _session and _session.get("save_path") == req.path:
            _session = {}
            _last_session_path = ""
        os.remove(req.path)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(400, str(e))

@app.get("/api/project/state")
def get_state():
    s = _require_session()
    cur = s["nodes"][s["current_node"]]
    return {
        "project":       s["project"],
        "current_node":  s["current_node"],
        "tree":          sm.get_tree_for_ui(s),
        "style_summary": sm.get_style_summary(s),
        "style_candidates": sm.get_style_candidates(s),
        "ref_images":    s["reference_images"],
        "save_path":     s["save_path"],
        "history":       _history_for_ui(s),
        "path_tags":     _path_tags(s),
        "current_images": cur["images"],
        "current_prompt": cur["prompt"],
        "current_selected": cur["selected"],
    }


# ── 核心：生成图片 ───────────────────────────

class GenerateReq(BaseModel):
    user_input: str
    n: int = 4
    parent_node_id: Optional[str] = None
    optimize_prompt: bool = True
    model_memory: Optional[str] = None

class PolishReq(BaseModel):
    user_input: str

@app.post("/api/polish_prompt")
def polish_prompt(req: PolishReq):
    """润色提示词（不生成图片）"""
    s = _require_session()
    try:
        context = sm.build_context_for_agent(s, req.user_input, None)
    except ValueError as e:
        raise HTTPException(400, str(e))
    try:
        polished = agent.prompt_agent(context)
        return {"polished_prompt": polished, "optimized": True, "warning": None}
    except Exception as e:
        # 余额不足/配额错误等场景下，不中断流程，自动退回本地路径记忆拼接
        polished = agent.compose_prompt_from_context(context)
        return {
            "polished_prompt": polished,
            "optimized": False,
            "warning": f"AI润色失败，已切换为路径记忆拼接：{e}",
        }

@app.post("/api/generate")
def generate(req: GenerateReq):
    s = _require_session()
    if req.parent_node_id and req.parent_node_id not in s["nodes"]:
        raise HTTPException(400, f"节点 {req.parent_node_id} 不存在")

    try:
        context = sm.build_context_for_agent(s, req.user_input, req.parent_node_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    
    # 添加模型记忆到context
    if req.model_memory:
        context["model_memory"] = req.model_memory

    prompt_warning = None
    if req.optimize_prompt:
        try:
            prompt = agent.prompt_agent(context)
        except Exception as e:
            prompt = agent.compose_prompt_from_context(context)
            prompt_warning = f"AI润色失败，已切换为路径记忆拼接：{e}"
            print(f"[warn] prompt_agent failed: {e}")
    else:
        prompt = agent.compose_prompt_from_context(context)

    images = agent.generate_images(
        prompt, n=req.n,
        save_dir=Path(__file__).parent / "static" / "uploads",
    )
    ok_images = [
        img for img in images
        if isinstance(img, dict) and img.get("url")
    ]
    if not ok_images:
        first_error = next(
            (img.get("error") for img in images if isinstance(img, dict) and img.get("error")),
            "未生成任何可用图片"
        )
        raise HTTPException(502, f"本次生成失败，未创建新节点：{first_error}")

    try:
        node = sm.add_node(s, req.user_input, req.parent_node_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    node_id = node["id"]
    sm.set_node_prompt(s, node_id, prompt)
    sm.add_images(s, node_id, images)

    return {
        "node_id": node_id,
        "optimized_prompt": prompt,
        "prompt_optimized": req.optimize_prompt,
        "prompt_warning": prompt_warning,
        "images": s["nodes"][node_id]["images"],
    }


# ── 选图 ─────────────────────────────────────

class SelectImageReq(BaseModel):
    image_url: str

@app.post("/api/select_image")
def select_image(req: SelectImageReq):
    s = _require_session()
    try:
        sm.select_image(s, req.image_url)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "style_summary": sm.get_style_summary(s)}


class StyleSelectReq(BaseModel):
    candidate_keywords: list[str]
    selected_keywords: list[str]


@app.post("/api/style/select")
def style_select(req: StyleSelectReq):
    s = _require_session()
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
def switch_node(req: SwitchNodeReq):
    s = _require_session()
    try:
        sm.switch_node(s, req.node_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "current_node": s["current_node"]}


# ── 语音转文字 ───────────────────────────────

@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    suffix = Path(audio.filename or "audio.webm").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        text = agent.transcribe(tmp_path)
    finally:
        os.unlink(tmp_path)
    return {"text": text}


# ── 上传参考图 ───────────────────────────────

@app.post("/api/upload_reference")
async def upload_reference(file: UploadFile = File(...), label: str = Form("")):
    s = _require_session()
    dest = Path(__file__).parent / "static" / "uploads" / f"ref_{file.filename}"
    dest.write_bytes(await file.read())
    url = f"/static/uploads/ref_{file.filename}"
    sm.add_reference_image(s, url, label or file.filename)
    return {"ok": True, "url": url}


# ── 保存 API Key ─────────────────────────────

class SaveKeyReq(BaseModel):
    api_key: str

@app.post("/api/save_key")
def save_key(req: SaveKeyReq):
    config_path = Path(__file__).parent / "config.json"
    data = json.loads(config_path.read_text()) if config_path.exists() else {}
    data["openai_api_key"] = req.api_key
    config_path.write_text(json.dumps(data))
    os.environ["OPENAI_API_KEY"] = req.api_key
    return {"ok": True}


# ── 平台选择 ───────────────────────────────

class SetPlatformReq(BaseModel):
    platform: str
    image_model: Optional[str] = None
    text_model: Optional[str] = None
    text_platform: Optional[str] = None  # 独立的文本平台

@app.get("/api/platforms")
def get_platforms():
    """获取所有支持的平台和模型列表"""
    return agent.get_platforms()

@app.post("/api/set_platform")
def set_platform(req: SetPlatformReq):
    """设置当前使用的平台和模型"""
    try:
        agent.set_platform(req.platform, req.image_model, req.text_model, req.text_platform)
        return {"ok": True, "config": agent.get_current_config()}
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.get("/api/current_platform")
def get_current_platform():
    """获取当前平台配置"""
    return agent.get_current_config()


# ── 辅助格式化 ───────────────────────────────

def _history_for_ui(s: dict) -> list:
    out = []
    for n in s["nodes"].values():
        selected_image = None
        selected_url = n.get("selected")
        if selected_url:
            for img in n.get("images", []):
                if isinstance(img, dict) and img.get("url") == selected_url:
                    selected_image = img
                    break

        out.append({
            "id":         n["id"],
            "user_input": n["user_input"],
            "prompt":     n["prompt"],
            "selected":   selected_url,
            "selected_image": selected_image,
            "images":     len(n["images"]),
            "is_current": n["id"] == s["current_node"],
            "parent":     n["parent"],
        })
    return out

def _path_tags(s: dict) -> list:
    path    = sm.get_path_to_root(s)
    cur_id  = s["current_node"]
    tags    = []
    for node in path:
        words = [w for w in node["user_input"].replace("，", " ").replace(",", " ").split() if len(w) >= 2]
        for w in words[:3]:
            tags.append({"text": w, "is_new": node["id"] == cur_id})
    return tags


if __name__ == "__main__":
    print("\n  ArchAI Design Studio → http://localhost:8000\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
