"""
state_manager.py — 设计状态树核心管理器

数据结构（无数据库，纯JSON）：
{
  "project":      项目名,
  "created":      创建时间,
  "current_node": 当前活跃节点ID,
  "nodes": {
    "root": {
      "id":          节点ID,
      "parent":      父节点ID / null,
      "children":    [子节点ID, ...],
      "user_input":  用户的原始输入文字,
      "prompt":      经Agent优化后的完整prompt（仅本节点新增部分）,
      "images":      [{url, local_path, timestamp}, ...],
      "selected":    被选中的图片url / null,
      "created_at":  时间戳,
    },
    ...
  },
  "style_weights":    { "关键词": float },  # 自动学习的风格偏好
  "reference_images": [{ "path": ..., "label": ... }],
  "save_path":        会话文件路径
}
"""

import json
import uuid
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


SESSIONS_DIR = Path(__file__).parent / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# 节点
# ─────────────────────────────────────────────

def _new_node(node_id: str, parent_id: Optional[str], user_input: str) -> dict:
    return {
        "id":         node_id,
        "parent":     parent_id,
        "children":   [],
        "user_input": user_input,
        "prompt":     "",          # 由Agent填写
        "images":     [],
        "selected":   None,
        "created_at": datetime.now().isoformat(),
    }


# ─────────────────────────────────────────────
# 会话
# ─────────────────────────────────────────────

def new_session(project_name: str) -> dict:
    """创建全新会话，返回session dict。"""
    session = {
        "project":        project_name,
        "created":        datetime.now().isoformat(),
        "current_node":   "root",
        "nodes":          {"root": _new_node("root", None, project_name)},
        "style_weights":  {},
        "reference_images": [],
        "save_path":      str(SESSIONS_DIR / f"{_slugify(project_name)}_{_ts()}.json"),
    }
    _save(session)
    return session


def load_session(path: str) -> dict:
    """从JSON文件加载会话。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_sessions() -> list[dict]:
    """列出所有已保存的会话（按时间倒序）。"""
    files = sorted(SESSIONS_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)
    result = []
    for f in files:
        try:
            s = load_session(str(f))
            result.append({
                "path":    str(f),
                "project": s["project"],
                "created": s["created"],
                "nodes":   len(s["nodes"]),
            })
        except Exception:
            pass
    return result


# ─────────────────────────────────────────────
# 树操作
# ─────────────────────────────────────────────

def add_node(session: dict, user_input: str, parent_id: Optional[str] = None) -> dict:
    """
    在parent_id下新建子节点，成为当前节点。
    如果parent_id为None，默认接在current_node后面。
    """
    parent_id = parent_id or session["current_node"]
    if parent_id not in session["nodes"]:
        raise ValueError(f"节点 {parent_id} 不存在")
    node_id   = f"v{len(session['nodes'])}"   # 简单递增ID

    node = _new_node(node_id, parent_id, user_input)
    session["nodes"][node_id] = node
    session["nodes"][parent_id]["children"].append(node_id)
    session["current_node"] = node_id
    _save(session)
    return node


def select_image(session: dict, image_url: str) -> None:
    """在当前节点标记选中图片，并更新风格权重。传入空字符串则取消选择。"""
    node = _current(session)
    
    # 空字符串表示取消选择
    if not image_url:
        node["selected"] = None
        _save(session)
        return
    
    valid_urls = {img.get("url") for img in node["images"] if isinstance(img, dict)}
    if image_url not in valid_urls:
        raise ValueError("只能选择当前节点下的图片")
    node["selected"] = image_url
    _update_style_weights(session, image_url, node["prompt"])
    _save(session)


def set_node_prompt(session: dict, node_id: str, prompt: str) -> None:
    """Agent写回优化后的prompt。"""
    session["nodes"][node_id]["prompt"] = prompt
    _save(session)


def add_images(session: dict, node_id: str, images: list[dict]) -> None:
    """将生成的图片列表写入节点。images = [{url, local_path}]"""
    ok_images = [
        img for img in images
        if isinstance(img, dict) and img.get("url")
    ]
    session["nodes"][node_id]["images"].extend(ok_images)
    _save(session)


def switch_node(session: dict, node_id: str) -> None:
    """切换当前活跃节点（用于回退/切换分支）。"""
    if node_id not in session["nodes"]:
        raise ValueError(f"节点 {node_id} 不存在")
    session["current_node"] = node_id
    _save(session)


def add_reference_image(session: dict, path: str, label: str = "") -> None:
    session["reference_images"].append({"path": path, "label": label})
    _save(session)


# ─────────────────────────────────────────────
# Prompt 路径拼接（核心逻辑）
# ─────────────────────────────────────────────

def get_path_to_root(session: dict, node_id: Optional[str] = None) -> list[dict]:
    """
    从node_id向上遍历到root，返回有序节点列表（root在前）。
    这条路径上的所有user_input和prompt就是生成图片的完整上下文。
    """
    node_id = node_id or session["current_node"]
    if node_id not in session["nodes"]:
        raise ValueError(f"节点 {node_id} 不存在")
    path = []
    while node_id is not None:
        node = session["nodes"][node_id]
        path.append(node)
        node_id = node["parent"]
    return list(reversed(path))  # root → ... → current


def build_context_for_agent(session: dict, new_user_input: str, node_id: Optional[str] = None) -> dict:
    """
    为Agent准备完整上下文，供其生成优化后的图像prompt。
    返回dict，直接作为LLM的system+user消息素材。
    """
    path = get_path_to_root(session, node_id)

    history_steps = []
    selected_images = []
    for node in path:
        selected_img = _get_selected_image(node)
        step = {
            "user_input":  node["user_input"],
            "prompt_used": node["prompt"],
            "selected":    node["selected"],
            "selected_image": selected_img,
        }
        history_steps.append(step)
        if selected_img:
            selected_images.append({
                "node_id": node["id"],
                "node_user_input": node["user_input"],
                "url": selected_img.get("url"),
                "local_path": selected_img.get("local_path"),
                "revised_prompt": selected_img.get("revised_prompt"),
            })

    return {
        "history":         history_steps,        # root→current的完整历史
        "new_user_input":  new_user_input,        # 用户本次输入
        "style_weights":   session["style_weights"],
        "reference_images": [r["path"] for r in session["reference_images"]],
        "selected_images": selected_images,
        "project_name":    session["project"],
    }


def get_tree_for_ui(session: dict) -> dict:
    """
    返回适合前端渲染的树结构（嵌套形式）。
    """
    def _build(node_id):
        node = session["nodes"][node_id]
        return {
            "id":         node["id"],
            "label":      node["user_input"][:30] + ("…" if len(node["user_input"]) > 30 else ""),
            "selected":   node["selected"] is not None,
            "images":     len(node["images"]),
            "is_current": node["id"] == session["current_node"],
            "children":   [_build(c) for c in node["children"]],
        }
    return _build("root")


# ─────────────────────────────────────────────
# 风格权重（自动学习）
# ─────────────────────────────────────────────

# 建筑相关的风格关键词列表，用于从prompt中提取
STYLE_KEYWORDS = [
    "现代", "简约", "古典", "新古典", "巴洛克", "工业风", "未来主义",
    "玻璃幕墙", "混凝土", "木材", "钢结构", "砖石",
    "自然采光", "大出挑", "悬挑", "绿化", "屋顶花园",
    "对称", "非对称", "曲线", "直线", "几何",
    "暖色调", "冷色调", "白色", "灰色",
    "黄昏", "夜景", "晴天", "鸟瞰", "透视",
    "高层", "低层", "商业", "住宅", "文化",
]

def _update_style_weights(session: dict, selected_url: str, selected_prompt: str) -> None:
    """
    用户每次选图，从该图的prompt中提取关键词，对应权重 +0.2。
    未被选中图片的权重不变（我们不知道用户为什么不选）。
    """
    weights = session["style_weights"]
    for kw in STYLE_KEYWORDS:
        if kw in selected_prompt:
            weights[kw] = round(weights.get(kw, 0.0) + 0.2, 2)
    session["style_weights"] = weights


def get_style_summary(session: dict, top_n: int = 6) -> list[dict]:
    """返回权重最高的N个风格词，供前端展示。"""
    weights = session["style_weights"]
    sorted_kw = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    return [{"keyword": k, "weight": v} for k, v in sorted_kw[:top_n]]


# ─────────────────────────────────────────────
# 内部工具
# ─────────────────────────────────────────────

def _current(session: dict) -> dict:
    return session["nodes"][session["current_node"]]


def _get_selected_image(node: dict) -> Optional[dict]:
    selected_url = node.get("selected")
    if not selected_url:
        return None
    for img in node.get("images", []):
        if isinstance(img, dict) and img.get("url") == selected_url:
            return img
    return None


def _save(session: dict) -> None:
    path = session["save_path"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)


def _slugify(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text)[:30]


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
