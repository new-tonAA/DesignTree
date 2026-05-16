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
import re
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
    _update_style_weights(session, node, image_url)
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

STYLE_ALIASES = {
    "现代": ["现代", "现代风", "modern", "contemporary"],
    "简约": ["简约", "极简", "minimal", "minimalist"],
    "古典": ["古典", "classic", "classical"],
    "新古典": ["新古典", "neoclassical", "neo classical"],
    "巴洛克": ["巴洛克", "baroque"],
    "工业风": ["工业风", "industrial"],
    "未来主义": ["未来主义", "futuristic", "future"],
    "玻璃幕墙": ["玻璃幕墙", "curtain wall", "glass facade"],
    "混凝土": ["混凝土", "concrete"],
    "木材": ["木材", "木质", "wood"],
    "钢结构": ["钢结构", "steel structure", "steel frame"],
    "砖石": ["砖石", "brick", "masonry"],
    "自然采光": ["自然采光", "natural light", "daylight"],
    "大出挑": ["大出挑", "large cantilever", "deep overhang"],
    "悬挑": ["悬挑", "cantilever", "overhang"],
    "绿化": ["绿化", "landscape", "greenery"],
    "屋顶花园": ["屋顶花园", "roof garden", "rooftop garden"],
    "对称": ["对称", "symmetry", "symmetrical"],
    "非对称": ["非对称", "asymmetry", "asymmetrical"],
    "曲线": ["曲线", "curved", "curve"],
    "直线": ["直线", "straight line", "linear"],
    "几何": ["几何", "geometric", "geometry"],
    "暖色调": ["暖色调", "warm tone", "warm color"],
    "冷色调": ["冷色调", "cool tone", "cool color"],
    "白色": ["白色", "white"],
    "灰色": ["灰色", "gray", "grey"],
    "黄昏": ["黄昏", "dusk", "sunset"],
    "夜景": ["夜景", "night scene", "night"],
    "晴天": ["晴天", "sunny", "clear sky"],
    "鸟瞰": ["鸟瞰", "aerial view", "bird view", "bird's-eye"],
    "透视": ["透视", "perspective"],
    "高层": ["高层", "high-rise", "tower"],
    "低层": ["低层", "low-rise"],
    "商业": ["商业", "commercial"],
    "住宅": ["住宅", "residential"],
    "文化": ["文化", "cultural"],
}

NEGATION_CUES = [
    "不要", "不需要", "不要再", "别", "避免", "拒绝", "去掉", "取消", "不想要",
    "without", "avoid", "no ", "not ",
]

STYLE_WEIGHT_MIN = -2.0
STYLE_WEIGHT_MAX = 2.0
STYLE_DECAY_FACTOR = 0.98
EPS = 0.03


def _update_style_weights(session: dict, node: dict, selected_url: str) -> None:
    """
    自动风格学习（改进版）：
    1) 选中图相关描述做正向学习；
    2) 用户输入中的“不要/避免”做负向学习；
    3) 对未选中的候选图做轻微对比降权；
    4) 所有权重做轻微衰减，避免早期偏好长期锁死。
    """
    weights = session.get("style_weights", {})
    _decay_style_weights(weights)

    selected_img = None
    for img in node.get("images", []):
        if isinstance(img, dict) and img.get("url") == selected_url:
            selected_img = img
            break

    selected_text_parts = [
        node.get("user_input", ""),
        node.get("prompt", ""),
        (selected_img or {}).get("revised_prompt", ""),
    ]
    selected_text = " ".join(x for x in selected_text_parts if x)
    selected_scores = _extract_style_scores(selected_text)

    # 选中图正向强化；若描述里本身带否定，也会得到负向强化
    for kw, score in selected_scores.items():
        if score > 0:
            _bump_style_weight(weights, kw, 0.24 * min(score, 3))
        elif score < 0:
            _bump_style_weight(weights, kw, -0.24 * min(abs(score), 3))

    # 用户明确“不要/避免”的风格应强力降权，优先级高于一般正向
    user_scores = _extract_style_scores(node.get("user_input", ""))
    for kw, score in user_scores.items():
        if score < 0:
            _bump_style_weight(weights, kw, -0.40 * min(abs(score), 3))
        elif score > 0:
            _bump_style_weight(weights, kw, 0.12 * min(score, 2))

    # 对比学习：未选中图中的独有风格，做轻微降权
    other_text = " ".join(
        img.get("revised_prompt", "")
        for img in node.get("images", [])
        if isinstance(img, dict) and img.get("url") and img.get("url") != selected_url
    )
    other_scores = _extract_style_scores(other_text)
    selected_positive = {k for k, v in selected_scores.items() if v > 0}
    other_positive = {k for k, v in other_scores.items() if v > 0}
    for kw in selected_positive - other_positive:
        _bump_style_weight(weights, kw, 0.08)
    for kw in other_positive - selected_positive:
        _bump_style_weight(weights, kw, -0.06)

    session["style_weights"] = weights


def apply_style_selection(session: dict, candidate_keywords: list[str], selected_keywords: list[str]) -> None:
    """
    用户手动风格选择：
    - 勾选 = 要（正向加权）
    - 未勾选 = 不要（负向加权）
    """
    candidate = [kw for kw in candidate_keywords if kw in STYLE_KEYWORDS]
    if not candidate:
        return

    selected = {kw for kw in selected_keywords if kw in candidate}
    weights = session.get("style_weights", {})
    _decay_style_weights(weights)

    for kw in candidate:
        if kw in selected:
            _bump_style_weight(weights, kw, 0.45)
        else:
            _bump_style_weight(weights, kw, -0.30)

    session["style_weights"] = weights
    _save(session)


def get_style_summary(session: dict, top_n: int = 6) -> list[dict]:
    """按绝对权重返回风格偏好摘要（包含“要”和“不要”）。"""
    weights = session.get("style_weights", {})
    sorted_kw = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
    out = []
    for k, v in sorted_kw:
        if abs(v) < EPS:
            continue
        out.append({
            "keyword": k,
            "weight": round(v, 2),
            "tendency": "prefer" if v >= 0 else "avoid",
        })
        if len(out) >= top_n:
            break
    return out


def get_style_candidates(session: dict, top_n: int = 14) -> list[dict]:
    """
    返回可供用户勾选的风格候选词：
    综合当前路径文本命中 + 已学习权重。
    """
    weights = session.get("style_weights", {})
    path = get_path_to_root(session)

    score_map: dict[str, float] = {}
    for node in path[-6:]:
        text = " ".join([
            node.get("user_input", ""),
            node.get("prompt", ""),
            (_get_selected_image(node) or {}).get("revised_prompt", ""),
        ])
        for kw, score in _extract_style_scores(text).items():
            score_map[kw] = score_map.get(kw, 0.0) + abs(float(score))

    for kw, w in weights.items():
        score_map[kw] = score_map.get(kw, 0.0) + abs(float(w)) * 1.6

    ranked = [k for k, _ in sorted(score_map.items(), key=lambda x: x[1], reverse=True)]
    defaults = ["现代", "简约", "工业风", "玻璃幕墙", "混凝土", "木材", "自然采光", "夜景", "鸟瞰", "透视", "商业", "住宅"]

    candidates: list[str] = []
    for kw in ranked + defaults + STYLE_KEYWORDS:
        if kw not in STYLE_KEYWORDS or kw in candidates:
            continue
        candidates.append(kw)
        if len(candidates) >= top_n:
            break

    return [{
        "keyword": kw,
        "weight": round(float(weights.get(kw, 0.0)), 2),
        "state": "prefer" if weights.get(kw, 0.0) > EPS else ("avoid" if weights.get(kw, 0.0) < -EPS else "neutral"),
    } for kw in candidates]


def _decay_style_weights(weights: dict, factor: float = STYLE_DECAY_FACTOR) -> None:
    to_delete = []
    for kw, v in weights.items():
        nv = float(v) * factor
        if abs(nv) < EPS:
            to_delete.append(kw)
        else:
            weights[kw] = round(nv, 3)
    for kw in to_delete:
        weights.pop(kw, None)


def _bump_style_weight(weights: dict, keyword: str, delta: float) -> None:
    cur = float(weights.get(keyword, 0.0))
    nxt = max(STYLE_WEIGHT_MIN, min(STYLE_WEIGHT_MAX, cur + float(delta)))
    if abs(nxt) < EPS:
        weights.pop(keyword, None)
    else:
        weights[keyword] = round(nxt, 3)


def _extract_style_scores(text: str) -> dict[str, int]:
    """
    从文本提取风格词并判断正负倾向。
    命中“不要/避免/without/avoid”等否定前缀时记为负分。
    """
    norm = _normalize_text(text)
    if not norm:
        return {}

    scores: dict[str, int] = {}
    for kw in STYLE_KEYWORDS:
        patterns = STYLE_ALIASES.get(kw, [kw])
        score = 0
        for token in patterns:
            token_norm = _normalize_text(token).strip()
            if not token_norm:
                continue
            start = 0
            while True:
                idx = norm.find(token_norm, start)
                if idx < 0:
                    break
                prefix = norm[max(0, idx - 12):idx]
                is_neg = any(cue in prefix for cue in NEGATION_CUES)
                score += -1 if is_neg else 1
                start = idx + len(token_norm)
        if score != 0:
            scores[kw] = score
    return scores


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    s = str(text).lower()
    s = re.sub(r"[，,。.!！？;；:：()（）\[\]{}<>\"'`~\-_/\\]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


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
