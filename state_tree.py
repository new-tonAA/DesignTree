"""
state_tree.py
设计状态树管理器 —— 所有数据保存为本地 JSON 文件，无需数据库。

核心数据结构：
  每个节点（DesignNode）= 一次用户输入 + 生成的图片 + 用户选中的图
  树的路径 root → ... → current_node = 拼接 prompt 的上下文链
"""

import json
import uuid
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


SESSIONS_DIR = Path(__file__).parent / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# 数据模型
# ─────────────────────────────────────────────

@dataclass
class DesignNode:
    node_id: str
    parent_id: Optional[str]          # root 节点为 None
    user_input: str                    # 用户这一步的原始输入
    assembled_prompt: str             # Agent 整合后的最终 prompt（发给图像模型的）
    images: list[str]                 # 生成图片的 URL / 本地路径列表
    selected_image: Optional[str]     # 用户选中的那张
    created_at: float
    children: list[str] = field(default_factory=list)   # 子节点 id 列表
    label: str = ""                   # 在状态树 UI 中显示的短标签（自动生成）


@dataclass
class DesignSession:
    session_id: str
    project_name: str
    created_at: float
    nodes: dict[str, DesignNode]       # node_id → DesignNode
    root_id: str
    current_node_id: str
    style_weights: dict[str, float]    # 风格偏好权重，自动更新
    reference_images: list[str]        # 参考图路径
    updated_at: float = 0.0


# ─────────────────────────────────────────────
# 序列化（dataclass ↔ dict）
# ─────────────────────────────────────────────

def _node_to_dict(n: DesignNode) -> dict:
    return asdict(n)

def _node_from_dict(d: dict) -> DesignNode:
    return DesignNode(**d)

def _session_to_dict(s: DesignSession) -> dict:
    d = asdict(s)
    # nodes 需要特殊处理（dict of dataclass）
    d["nodes"] = {k: _node_to_dict(v) for k, v in s.nodes.items()}
    return d

def _session_from_dict(d: dict) -> DesignSession:
    nodes = {k: _node_from_dict(v) for k, v in d["nodes"].items()}
    d["nodes"] = nodes
    return DesignSession(**d)


# ─────────────────────────────────────────────
# 持久化
# ─────────────────────────────────────────────

def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"

def save_session(session: DesignSession) -> None:
    session.updated_at = time.time()
    path = _session_path(session.session_id)
    path.write_text(json.dumps(_session_to_dict(session), ensure_ascii=False, indent=2), encoding="utf-8")

def load_session(session_id: str) -> DesignSession:
    path = _session_path(session_id)
    if not path.exists():
        raise FileNotFoundError(f"Session not found: {session_id}")
    return _session_from_dict(json.loads(path.read_text(encoding="utf-8")))

def list_sessions() -> list[dict]:
    """返回所有 session 的摘要信息（用于首页列表）"""
    result = []
    for p in sorted(SESSIONS_DIR.glob("*.json"), key=lambda x: -x.stat().st_mtime):
        try:
            s = _session_from_dict(json.loads(p.read_text(encoding="utf-8")))
            node_count = len(s.nodes)
            result.append({
                "session_id": s.session_id,
                "project_name": s.project_name,
                "created_at": s.created_at,
                "updated_at": s.updated_at,
                "node_count": node_count,
                "current_label": s.nodes[s.current_node_id].label,
            })
        except Exception:
            pass
    return result


# ─────────────────────────────────────────────
# 状态树操作
# ─────────────────────────────────────────────

def create_session(project_name: str, root_input: str, root_prompt: str) -> DesignSession:
    """新建项目，创建根节点"""
    sid = str(uuid.uuid4())[:8]
    root_id = "root"
    root_node = DesignNode(
        node_id=root_id,
        parent_id=None,
        user_input=root_input,
        assembled_prompt=root_prompt,
        images=[],
        selected_image=None,
        created_at=time.time(),
        label=project_name,
    )
    session = DesignSession(
        session_id=sid,
        project_name=project_name,
        created_at=time.time(),
        updated_at=time.time(),
        nodes={root_id: root_node},
        root_id=root_id,
        current_node_id=root_id,
        style_weights={},
        reference_images=[],
    )
    save_session(session)
    return session


def add_images_to_node(session: DesignSession, node_id: str, image_urls: list[str]) -> None:
    """将生成的图片 URL 存入节点"""
    session.nodes[node_id].images = image_urls
    save_session(session)


def select_image(session: DesignSession, node_id: str, image_url: str) -> None:
    """用户选中某张图，同时更新风格权重"""
    node = session.nodes[node_id]
    node.selected_image = image_url
    save_session(session)


def update_style_weights(session: DesignSession, keywords: list[str], delta: float = 0.15) -> None:
    """根据本次 prompt 关键词更新风格权重（选图时调用）"""
    for kw in keywords:
        prev = session.style_weights.get(kw, 0.0)
        session.style_weights[kw] = round(min(1.0, prev + delta), 3)
    save_session(session)


def create_child_node(
    session: DesignSession,
    parent_id: str,
    user_input: str,
    assembled_prompt: str,
    label: str = "",
) -> DesignNode:
    """
    在 parent_id 节点下新建子节点（用户选完图、提出改进后调用）。
    支持分支：parent_id 可以是树中任意已有节点（不必须是当前叶子）。
    """
    node_id = f"v{len(session.nodes)}"
    node = DesignNode(
        node_id=node_id,
        parent_id=parent_id,
        user_input=user_input,
        assembled_prompt=assembled_prompt,
        images=[],
        selected_image=None,
        created_at=time.time(),
        label=label or user_input[:20],
    )
    session.nodes[node_id] = node
    session.nodes[parent_id].children.append(node_id)
    session.current_node_id = node_id
    save_session(session)
    return node


def get_path_to_root(session: DesignSession, node_id: str) -> list[DesignNode]:
    """
    返回从 root 到 node_id 的完整节点路径（含两端）。
    这条路径上所有节点的 user_input 就是 prompt 的历史上下文。
    """
    path = []
    current = node_id
    while current is not None:
        node = session.nodes[current]
        path.append(node)
        current = node.parent_id
    path.reverse()
    return path


def get_tree_for_ui(session: DesignSession) -> dict:
    """
    将状态树序列化为前端渲染用的嵌套结构。
    """
    def build(node_id: str) -> dict:
        node = session.nodes[node_id]
        return {
            "node_id": node.node_id,
            "label": node.label,
            "user_input": node.user_input,
            "image_count": len(node.images),
            "selected": node.selected_image is not None,
            "is_current": node.node_id == session.current_node_id,
            "children": [build(c) for c in node.children],
        }
    return build(session.root_id)


def add_reference_image(session: DesignSession, image_path: str) -> None:
    if image_path not in session.reference_images:
        session.reference_images.append(image_path)
    save_session(session)
