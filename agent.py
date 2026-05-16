"""
agent.py — AI调用层

职责：
1. prompt_agent()   : 用户输入 + 历史路径 → 优化后的图像生成prompt
2. generate_images(): 调用图像API生成N张图（支持多平台）
3. transcribe()     : 调用 Whisper API 将音频转文字

支持平台：OpenAI、OpenRouter、V3.CM、火山引擎
"""

import os
import json
import time
import base64
import re
import httpx
from pathlib import Path
from typing import Optional, Dict, Any


_CONFIG_PATH = Path(__file__).parent / "config.json"

# ─────────────────────────────────────────────
# 平台配置（写死在代码中）
# ─────────────────────────────────────────────

PLATFORMS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com",
        "api_key": "",
        "models": {
            "image": ["gpt-image-1", "gpt-image-1.5", "gpt-image-2", "gpt-image-1-mini"],
            "text": ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"],
            "audio": ["whisper-1"]
        }
    },
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api",
        "api_key": "",
        "models": {
            "image": [
                "openai/gpt-5-image-mini",
                "openai/gpt-5-image",
                "openai/gpt-5.4-image-2",
                "google/gemini-2.5-flash-image",
                "google/gemini-3.1-flash-image-preview",
                "google/gemini-3-pro-image-preview",
            ],
            "text": ["anthropic/claude-3.5-sonnet", "google/gemini-pro-1.5", "openai/gpt-4o"]
        }
    },
    "v3": {
        "name": "V3.CM",
        "base_url": "https://api.v3.cm",
        "api_key": "",
        "models": {
            "image": [
                "gpt-image-1",
                "gpt-image-1.5",
                "gpt-image-2",
                "dall-e-3",
                "gemini-3.1-flash-image-preview",
                "gemini-3.1-flash-image-preview-0.5k",
                "gemini-3.1-flash-image-preview-2k",
                "gemini-3.1-flash-image-preview-4k",
                "gemini-3-pro-image-preview",
                "gemini-2.5-flash-image",
                "nano-banana-2",
                "nano-banana-2-0.5k",
                "nano-banana-2-2k",
                "nano-banana-2-4k",
                "nano-banana-pro",
                "qwen-image",
                "stable-diffusion-xl-base-1.0",
            ],
            "text": [
                "gpt-4o",
                "gpt-4-turbo",
                "gpt-4o-mini",
                "gemini-3.1-pro-preview",
            ]
        }
    },
    "volcengine": {
        "name": "火山引擎",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "api_key": "",
        "models": {
            "image": [
                "doubao-seedream-4-0-250828",
                "doubao-seedream-4-5-251128",
                "doubao-seedream-5-0-260128",
                "doubao-seedream-3-0-t2i-250415",
            ],
            "text": ["doubao-pro-32k", "doubao-lite-32k"]
        }
    },
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "api_key": "",
        "models": {
            "image": [],  # DeepSeek 不支持图像生成
            "text": ["deepseek-chat", "deepseek-coder"]
        }
    }
}

# 当前选择的平台和模型（可通过API修改）
_current_platform = "v3"
_current_image_model = "gpt-image-1"
_current_text_platform = "v3"  # 文本模型独立平台（V3.CM 有文本模型）
_current_text_model = "gpt-4o-mini"


def get_platforms() -> Dict[str, Dict[str, Any]]:
    """返回所有平台配置（不包含api_key）"""
    return {
        k: {
            "name": v["name"],
            "models": v["models"]
        }
        for k, v in PLATFORMS.items()
    }


def _model_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").strip().lower())


def _normalize_image_model(platform: str, image_model: Optional[str]) -> Optional[str]:
    if image_model is None:
        return None
    model = str(image_model).strip()
    if not model:
        return model

    supported = PLATFORMS.get(platform, {}).get("models", {}).get("image", [])
    by_key = {_model_key(m): m for m in supported}
    key = _model_key(model)

    # 旧名称兼容（如 gptimage2 / dalle3）
    alias = {
        "gptimage1": "gpt-image-1",
        "gptimage15": "gpt-image-1.5",
        "gptimage2": "gpt-image-2",
        "gptimage1mini": "gpt-image-1-mini",
        "dalle2": "dall-e-2",
        "dalle3": "dall-e-3",
    }
    mapped = alias.get(key, model)
    mapped_key = _model_key(mapped)

    if mapped_key in by_key:
        return by_key[mapped_key]

    # OpenAI 不再支持 DALL-E 系列时，自动映射到 gpt-image-1
    if platform == "openai" and mapped_key in {"dalle2", "dalle3"}:
        return "gpt-image-1"

    return mapped


def set_platform(platform: Optional[str] = None, image_model: Optional[str] = None, text_model: Optional[str] = None, text_platform: Optional[str] = None):
    """设置当前使用的平台和模型"""
    global _current_platform, _current_image_model, _current_text_platform, _current_text_model
    
    # 图像平台设置
    if platform:
        if platform not in PLATFORMS:
            raise ValueError(f"不支持的平台: {platform}")
        _current_platform = platform
        config = PLATFORMS[platform]
        
        if image_model:
            normalized_model = _normalize_image_model(platform, image_model)
            if normalized_model not in config["models"].get("image", []):
                raise ValueError(f"平台 {platform} 不支持图像模型: {normalized_model}")
            _current_image_model = normalized_model
    
    # 文本平台设置（独立）
    if text_platform:
        if text_platform not in PLATFORMS:
            raise ValueError(f"不支持的文本平台: {text_platform}")
        _current_text_platform = text_platform
    
    if text_model:
        text_config = PLATFORMS[_current_text_platform]
        if text_model not in text_config["models"].get("text", []):
            raise ValueError(f"平台 {_current_text_platform} 不支持文本模型: {text_model}")
        _current_text_model = text_model


def get_current_config() -> Dict[str, str]:
    """返回当前配置"""
    return {
        "platform": _current_platform,
        "image_model": _current_image_model,
        "text_platform": _current_text_platform,
        "text_model": _current_text_model
    }


def _get_api_key() -> str:
    """获取当前图像平台的API Key"""
    return _sanitize_api_key(PLATFORMS[_current_platform].get("api_key", ""))


def _get_text_api_key() -> str:
    """获取当前文本平台的API Key"""
    return _sanitize_api_key(PLATFORMS[_current_text_platform].get("api_key", ""))


def _sanitize_api_key(raw: str) -> str:
    """
    清洗 API Key，避免“看起来有效但被判无效令牌”的常见输入问题：
    - 复制了 `Bearer xxx`
    - 包含换行/空格/零宽字符
    - 外层多了一对引号
    """
    s = str(raw or "")
    s = s.replace("\u00A0", " ").replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    s = s.strip()
    if s.lower().startswith("bearer "):
        s = s.split(" ", 1)[1].strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in {"'", '"', "`"}:
        s = s[1:-1].strip()
    s = re.sub(r"\s+", "", s)
    return s


def _mask_api_key(key: str) -> str:
    s = _sanitize_api_key(key)
    if not s:
        return "<empty>"
    if len(s) <= 10:
        return s[:2] + "***"
    return f"{s[:6]}...{s[-4:]}"


# ─────────────────────────────────────────────
# 1. Prompt Agent
# ─────────────────────────────────────────────

_SYSTEM_PROMPT = """你是一名建筑效果图提示词助手。

你的任务很简单：**把用户的需求描述改得更清楚、更具体**。

规则：
1. 用户说什么就改什么，不要添加用户没要求的内容
2. 如果用户说"改成XX"，你就把原来的XX改成新的
3. 如果用户描述模糊，帮他补充具体细节（如材质、视角）
4. 保持简洁，不要写太长，50-100字足够
5. 直接输出润色后的文字，不要解释
6. 绝对不能改变用户核心意图；仅允许“清晰化表达”，不允许“改写需求方向”

示例：
用户输入："把屋顶改成红色"
润色输出："红色屋顶的现代建筑，保持原有建筑结构和周围环境，照片级建筑效果图"
"""


def prompt_agent(context: dict) -> str:
    """
    context 由 state_manager.build_context_for_agent() 生成。
    返回优化后的英文图像prompt字符串。
    使用独立的文本平台（如 DeepSeek）来润色提示词。
    """
    api_key = _get_text_api_key()
    text_platform = _current_text_platform
    config = PLATFORMS[text_platform]

    history_text = _format_history(context["history"])
    weights_text = _format_weights(context["style_weights"])
    selected_text = _format_selected_images(context.get("selected_images", []))
    path_attachments_text = _format_path_attachments(context.get("path_attachments", []))
    model_memory = context.get("model_memory", "")
    memory_text = f"\n模型记忆（永久附加）：{model_memory}" if model_memory else ""

    user_msg = (
        f"项目：{context['project_name']}\n\n"
        f"设计迭代路径（root → 当前）：\n{history_text}\n\n"
        f"路径上的已选图片记忆：\n{selected_text}\n\n"
        f"路径上的用户上传参考图：\n{path_attachments_text}\n\n"
        f"本次新需求：{context['new_user_input']}\n\n"
        f"设计师风格偏好：{weights_text}{memory_text}\n\n"
        f"硬约束：不得改变“本次新需求”的语义方向，只能做清晰化与可视化细节补充。"
    )

    # OpenRouter需要特殊header
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if text_platform == "openrouter":
        headers["HTTP-Referer"] = "https://localhost:8000"
        headers["X-Title"] = "ArchAI"

    endpoint = "/v1/chat/completions"
    if text_platform == "volcengine":
        # 火山引擎 Ark 网关路径为 /api/v3/chat/completions（base_url 已带 /api/v3）
        endpoint = "/chat/completions"
    
    resp = _platform_post(
        endpoint,
        {
            "model": _current_text_model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            "max_tokens": 350,
            "temperature": 0.7,
        },
        api_key,
        text_platform,
        headers,
    )
    return resp["choices"][0]["message"]["content"].strip()


def compose_prompt_from_context(context: dict) -> str:
    """
    不调用LLM时的兜底提示词拼接：
    把 root->当前路径 的设计记忆 + 已选图记忆 + 本次新需求拼成可直接出图的英文prompt。
    """
    history = context.get("history", []) or []
    selected_images = context.get("selected_images", []) or []
    path_attachments = context.get("path_attachments", []) or []
    style_weights = context.get("style_weights", {}) or {}
    new_user_input = (context.get("new_user_input") or "").strip()
    model_memory = (context.get("model_memory") or "").strip()

    memory_inputs = _dedup_text_items(
        [h.get("user_input", "").strip() for h in history if h.get("user_input")],
        max_items=6,
        similarity_threshold=0.82,
    )
    memory_prompts = _dedup_text_items(
        [h.get("prompt_used", "").strip() for h in history if h.get("prompt_used")],
        max_items=4,
        similarity_threshold=0.78,
    )
    selected_revised = _dedup_text_items(
        [x.get("revised_prompt", "").strip() for x in selected_images if x.get("revised_prompt")],
        max_items=4,
        similarity_threshold=0.78,
    )
    attachment_hints = _dedup_text_items(
        [_attachment_hint(x) for x in path_attachments if _attachment_hint(x)],
        max_items=6,
        similarity_threshold=0.9,
    )

    preferred_styles = [
        k for k, v in sorted(style_weights.items(), key=lambda x: x[1], reverse=True)
        if v >= 0.3
    ][:6]
    avoided_styles = [
        k for k, v in sorted(style_weights.items(), key=lambda x: x[1])
        if v <= -0.3
    ][:6]

    parts = []
    # 模型记忆优先添加
    if model_memory:
        parts.append("Model memory (always applied): " + model_memory + ".")
    if memory_inputs:
        parts.append("Design evolution memory from previous nodes: " + " | ".join(memory_inputs) + ".")
    if memory_prompts:
        parts.append("Established visual decisions: " + " | ".join(memory_prompts) + ".")
    if selected_revised:
        parts.append("Selected image references to preserve: " + " | ".join(selected_revised) + ".")
    if attachment_hints:
        parts.append("User-uploaded references on current path: " + " | ".join(attachment_hints) + ".")
    if preferred_styles:
        parts.append("Preferred style cues: " + ", ".join(preferred_styles) + ".")
    if avoided_styles:
        parts.append("Avoid these style cues: " + ", ".join(avoided_styles) + ".")
    if new_user_input:
        parts.append("Primary user request (must remain unchanged): " + new_user_input + ".")

    parts.append(
        "Keep architectural consistency while refining details, and never contradict the primary user request. "
        "Include clear viewpoint, lighting, materials, surroundings, "
        "photorealistic architectural visualization, 8k, professional rendering."
    )
    return " ".join(parts).strip()


def _format_history(history: list) -> str:
    if not history:
        return "  （无历史，这是初始需求）"
    lines = []
    for i, h in enumerate(history):
        lines.append(f"  [{i}] 用户输入: {h['user_input']}")
        if h.get("prompt_used"):
            lines.append(f"      → prompt: {h['prompt_used']}")
        if h.get("selected"):
            lines.append(f"      → 用户选中了一张图继续")
        selected_image = h.get("selected_image")
        if isinstance(selected_image, dict) and selected_image.get("revised_prompt"):
            lines.append(f"      → 该节点选中图的revised_prompt: {selected_image['revised_prompt']}")
    return "\n".join(lines)


def _format_selected_images(selected_images: list) -> str:
    if not selected_images:
        return "  （路径上暂无已选图）"

    lines = []
    for i, img in enumerate(selected_images[-10:]):  # 最多带最近10条，防止上下文过长
        node_id = img.get("node_id", "?")
        node_input = img.get("node_user_input", "")
        url = img.get("url", "")
        revised = img.get("revised_prompt", "")
        lines.append(f"  [{i}] 节点 {node_id} 用户输入: {node_input}")
        if revised:
            lines.append(f"      → revised_prompt: {revised}")
        if url:
            lines.append(f"      → image_url: {url}")
    return "\n".join(lines)


def _format_path_attachments(path_attachments: list) -> str:
    if not path_attachments:
        return "  （路径上暂无用户上传参考图）"

    lines = []
    for i, att in enumerate(path_attachments[-12:]):
        node_id = att.get("node_id", "?")
        hint = _attachment_hint(att)
        lines.append(f"  [{i}] 节点 {node_id}: {hint}")
    return "\n".join(lines)


def _attachment_hint(att: dict) -> str:
    name = str((att or {}).get("name") or "").strip()
    if name:
        return _clean_text_piece(name)
    url = str((att or {}).get("url") or "").strip()
    if not url:
        return ""
    tail = url.rsplit("/", 1)[-1]
    tail = re.sub(r"\.[a-zA-Z0-9]+$", "", tail)
    tail = tail.replace("_", " ").replace("-", " ").strip()
    return _clean_text_piece(tail)


def _format_weights(weights: dict) -> str:
    if not weights:
        return "暂无"
    top = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    return "、".join(f"{k}({v:+.1f})" for k, v in top)


def _dedup_text_items(items: list[str], max_items: Optional[int] = None, similarity_threshold: float = 0.86) -> list[str]:
    """
    文本去重（保守策略）：
    - 完全相同去重
    - 子串高重合去重
    - Jaccard 词集相似度去重
    只去重“补充记忆片段”，不改写文本内容本身。
    """
    kept: list[str] = []
    kept_norm: list[str] = []
    kept_tokens: list[set[str]] = []

    for raw in items:
        text = _clean_text_piece(raw)
        if not text:
            continue
        norm = _normalize_for_similarity(text)
        if not norm:
            continue
        tokens = _tokenize_for_similarity(norm)
        if not tokens:
            continue

        duplicated = False
        for i, prev_norm in enumerate(kept_norm):
            prev_tokens = kept_tokens[i]
            if norm == prev_norm:
                duplicated = True
                break

            shorter = min(len(norm), len(prev_norm))
            longer = max(len(norm), len(prev_norm))
            if shorter > 0 and (norm in prev_norm or prev_norm in norm) and (shorter / longer >= 0.66):
                duplicated = True
                break

            if _jaccard_similarity(tokens, prev_tokens) >= similarity_threshold:
                duplicated = True
                break
            if _overlap_similarity(tokens, prev_tokens) >= 0.9:
                duplicated = True
                break

        if duplicated:
            continue

        kept.append(text)
        kept_norm.append(norm)
        kept_tokens.append(tokens)

    if max_items is not None and max_items > 0:
        return kept[-max_items:]
    return kept


def _clean_text_piece(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().strip("|,.;")


def _normalize_for_similarity(text: str) -> str:
    s = str(text or "").lower().strip()
    s = re.sub(r"[，。！？!?,;:：/\\|()\[\]{}\"'`~]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokenize_for_similarity(norm_text: str) -> set[str]:
    return {tok for tok in norm_text.split(" ") if tok}


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def _overlap_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    min_size = min(len(a), len(b))
    if min_size == 0:
        return 0.0
    return inter / min_size


# ─────────────────────────────────────────────
# 2. 图像生成
# ─────────────────────────────────────────────

def generate_images(
    prompt: str,
    n: int = 4,
    size: str = "1792x1024",
    quality: str = "standard",
    save_dir: Optional[Path] = None,
    on_image_ready: Optional[callable] = None,
) -> list:
    """
    调用图像API生成图片，支持多平台。
    返回 [{url, local_path, revised_prompt}, ...]
    
    on_image_ready: 可选回调函数，每张图片生成后立即调用 (image_dict) -> None
    """
    global _current_image_model

    api_key  = _get_api_key()
    platform = _current_platform
    save_dir = save_dir or (Path(__file__).parent / "static" / "uploads")
    save_dir.mkdir(parents=True, exist_ok=True)

    # OpenRouter需要特殊header
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if platform == "openrouter":
        headers["HTTP-Referer"] = "https://localhost:8000"
        headers["X-Title"] = "ArchAI"

    # 优先当前模型，失败时按平台模型列表逐个回退
    normalized_current = _normalize_image_model(platform, _current_image_model)
    if normalized_current:
        _current_image_model = normalized_current

    model_candidates = [_current_image_model]
    platform_image_models = PLATFORMS[platform].get("models", {}).get("image", [])
    for model in platform_image_models:
        if model not in model_candidates:
            model_candidates.append(model)
    if platform == "openai" and "gpt-image-1" not in model_candidates:
        model_candidates.append("gpt-image-1")
    model_candidates = list(dict.fromkeys(model_candidates))

    # 生成 n 张图，每次独立请求，失败不影响其它张
    results = []
    for i in range(n):
        try:
            item, used_model = _generate_single_image(
                prompt=prompt,
                requested_size=size,
                quality=quality,
                api_key=api_key,
                platform=platform,
                headers=headers,
                model_candidates=model_candidates,
            )
            if used_model != _current_image_model:
                _current_image_model = used_model

            ts = int(time.time() * 1000)
            filename = f"img_{ts}_{i}.png"
            local_path = save_dir / filename

            if "b64_json" in item:
                with open(local_path, "wb") as f:
                    f.write(base64.b64decode(item["b64_json"]))
            elif "url" in item:
                img_url = item["url"]
                if img_url.startswith("data:image"):
                    b64_data = img_url.split(",", 1)[1]
                    with open(local_path, "wb") as f:
                        f.write(base64.b64decode(b64_data))
                else:
                    img_resp = httpx.get(img_url, timeout=60)
                    with open(local_path, "wb") as f:
                        f.write(img_resp.content)
            else:
                raise RuntimeError("未知的图像返回格式")

            img_dict = {
                "url": f"/static/uploads/{filename}",
                "local_path": str(local_path),
                "revised_prompt": item.get("revised_prompt", prompt),
            }
            results.append(img_dict)
            
            # 每张图片生成后立即回调
            if on_image_ready:
                try:
                    on_image_ready(img_dict)
                except Exception as e:
                    print(f"[warn] on_image_ready callback failed: {e}")
                    
        except Exception as e:
            results.append({"url": None, "local_path": None, "error": str(e)})

        if i < n - 1:
            time.sleep(0.5)

    return results


# ─────────────────────────────────────────────
# 3. 语音转文字
# ─────────────────────────────────────────────


def _generate_single_image(
    prompt: str,
    requested_size: str,
    quality: str,
    api_key: str,
    platform: str,
    headers: Dict[str, str],
    model_candidates: list[str],
) -> tuple[dict, str]:
    last_err: Optional[Exception] = None

    for mi, model_name in enumerate(model_candidates):
        size_candidates = _get_size_candidates(platform, model_name, requested_size)

        for si, current_size in enumerate(size_candidates):
            try:
                if platform == "openrouter":
                    item = _generate_openrouter_image(prompt, model_name, current_size, api_key, headers)
                else:
                    item = _generate_openai_compatible_image(
                        prompt=prompt,
                        model_name=model_name,
                        current_size=current_size,
                        quality=quality,
                        api_key=api_key,
                        platform=platform,
                        headers=headers,
                    )
                return item, model_name
            except Exception as e:
                last_err = e
                err_text = str(e)

                if _is_size_unsupported_error(err_text):
                    # 先尝试同模型下其它尺寸，再尝试回退到下一模型
                    if si < len(size_candidates) - 1:
                        continue
                    if mi < len(model_candidates) - 1:
                        break

                if (
                    _is_model_not_found_error(err_text)
                    or _is_model_unavailable_error(err_text, model_name)
                ) and mi < len(model_candidates) - 1:
                    break

                raise

    if last_err:
        raise last_err
    raise RuntimeError("image generation failed with empty response")


def _generate_openai_compatible_image(
    prompt: str,
    model_name: str,
    current_size: str,
    quality: str,
    api_key: str,
    platform: str,
    headers: Dict[str, str],
) -> dict:
    endpoint = "/v1/images/generations"
    if platform == "volcengine":
        endpoint = "/images/generations"

    payload: Dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "n": 1,
        "size": current_size,
    }

    if "gpt-image" in model_name.lower():
        quality_map = {
            "standard": "medium",
            "hd": "high",
        }
        gpt_quality = quality_map.get(quality, quality)
        if gpt_quality in {"low", "medium", "high", "auto"}:
            payload["quality"] = gpt_quality
    elif model_name.lower().startswith("dall-e") and quality and platform in {"openai", "v3"}:
        payload["quality"] = quality

    resp = _platform_post(endpoint, payload, api_key, platform, headers)
    return resp["data"][0]


def _generate_openrouter_image(
    prompt: str,
    model_name: str,
    current_size: str,
    api_key: str,
    headers: Dict[str, str],
) -> dict:
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["image", "text"],
        "image_config": {
            "aspect_ratio": _size_to_openrouter_aspect_ratio(current_size),
        },
    }
    resp = _platform_post("/v1/chat/completions", payload, api_key, "openrouter", headers)
    choices = resp.get("choices") or []
    if not choices:
        raise RuntimeError("openrouter response has no choices")
    message = choices[0].get("message") or {}
    images = message.get("images") or []
    if not images:
        raise RuntimeError("openrouter response has no image output")

    img = images[0]
    url = (
        (img.get("image_url") or {}).get("url")
        or img.get("url")
        or (img.get("imageUrl") or {}).get("url")
    )
    if not url:
        raise RuntimeError("openrouter image output missing url")

    revised_prompt = ""
    content = message.get("content")
    if isinstance(content, str):
        revised_prompt = content
    return {"url": url, "revised_prompt": revised_prompt or prompt}


def _get_size_candidates(platform: str, model_name: str, requested_size: str) -> list[str]:
    sizes: list[str] = []

    def add(v: str):
        if v and v not in sizes:
            sizes.append(v)

    m = model_name.lower()
    add(requested_size)

    if "gpt-image" in m:
        add("1536x1024")
        add("1024x1536")
        add("1024x1024")
        add("auto")

    if "0.5k" in m:
        add("512x512")

    if platform == "v3":
        add("1024x1024")
        add("1536x1024")
        add("1024x1536")
        add("512x512")
        add("auto")
    elif platform in {"openai", "volcengine"}:
        add("1024x1024")
        add("1536x1024")
        add("1024x1536")
    elif platform == "openrouter":
        add("1024x1024")
        add("1536x1024")
        add("1024x1536")
        add("512x512")

    return sizes


def _size_to_openrouter_aspect_ratio(size: str) -> str:
    mapping = {
        "1024x1024": "1:1",
        "512x512": "1:1",
        "1536x1024": "3:2",
        "1024x1536": "2:3",
        "1792x1024": "16:9",
        "1024x1792": "9:16",
    }
    return mapping.get(size, "1:1")

def transcribe(audio_path: str, language: str = "zh") -> str:
    """Whisper API，支持 mp3/wav/webm/m4a（仅OpenAI支持）。"""
    api_key   = _get_api_key()
    platform  = _current_platform
    config    = PLATFORMS[platform]
    
    # 只有OpenAI和部分平台支持语音转文字
    if platform not in ["openai", "v3"]:
        raise RuntimeError(f"平台 {platform} 不支持语音转文字")
    
    mime_map  = {"mp3": "audio/mpeg", "wav": "audio/wav",
                 "webm": "audio/webm", "m4a": "audio/mp4"}
    suffix    = Path(audio_path).suffix.lstrip(".")
    mime_type = mime_map.get(suffix, "audio/webm")

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    base_url = config["base_url"]
    with httpx.Client(timeout=60) as client:
        resp = client.post(
            f"{base_url}/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (Path(audio_path).name, audio_bytes, mime_type)},
            data={"model": "whisper-1", "language": language},
        )
    resp.raise_for_status()
    return resp.json().get("text", "").strip()


# ─────────────────────────────────────────────
# 内部工具
# ─────────────────────────────────────────────

def _is_model_not_found_error(error_text: str) -> bool:
    text = (error_text or "").lower()
    normalized = text.replace("'", "\"")
    return (
        "does not exist" in text
        or "unknown model" in text
        or "model_not_found" in text
        or "unsupported model" in text
        or "not available in your region" in text
        or "invalidendpointormodel.notfound" in text
        or ("invalid_value" in text and "\"param\": \"model\"" in normalized)
    )


def _is_model_unavailable_error(error_text: str, model_name: Optional[str] = None) -> bool:
    text = (error_text or "").lower()
    explicit_hit = (
        "暂无可用渠道" in error_text
        or "无可用渠道" in error_text
        or "no available channel" in text
        or "no channel available" in text
        or "channel unavailable" in text
        or "model unavailable" in text
        or "temporarily unavailable" in text
    )
    if explicit_hit:
        return True

    # 某些网关会返回 "v3 400: ...模型 gptimage2 ..."，不带统一英文错误码
    # 当报错中明确提到当前模型，且是 4xx 请求错误（非余额/鉴权）时，尝试回退到下一个模型。
    model_key = _model_key(model_name or "")
    normalized = _model_key(text)
    model_mentioned = bool(model_key and model_key in normalized)
    if not model_mentioned:
        return False

    hard_fail = (
        "insufficient balance" in text
        or "invalid api key" in text
        or "unauthorized" in text
        or "forbidden" in text
        or "quota" in text
    )
    if hard_fail:
        return False

    return (" 400:" in text or "invalid_request" in text or "bad request" in text)


def _is_size_unsupported_error(error_text: str) -> bool:
    text = (error_text or "").lower()
    normalized = text.replace("'", "\"")
    return (
        "unsupported size" in text
        or "size not supported" in text
        or "size_not_supported" in text
        or "invalid size" in text
        or ("invalid_value" in text and "\"param\": \"size\"" in normalized)
    )


def _platform_post(path: str, payload: dict, api_key: str, platform: str, headers: Optional[Dict] = None) -> dict:
    """统一的平台API POST请求"""
    config = PLATFORMS[platform]
    base_url = config["base_url"]
    
    api_key = _sanitize_api_key(api_key)
    if headers is None:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    
    with httpx.Client(timeout=120) as client:
        resp = client.post(
            f"{base_url}{path}",
            headers=headers,
            json=payload,
        )
    if not (200 <= resp.status_code < 300):
        if resp.status_code == 401:
            raise RuntimeError(
                f"{platform} 401: {resp.text[:400]} (key={_mask_api_key(api_key)})"
            )
        raise RuntimeError(f"{platform} {resp.status_code}: {resp.text[:400]}")
    return resp.json()
