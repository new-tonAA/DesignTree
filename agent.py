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
        "api_key": "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",
        "models": {
            "image": ["gpt-image-1", "gpt-image-1.5", "gpt-image-2", "gpt-image-1-mini"],
            "text": ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"],
            "audio": ["whisper-1"]
        }
    },
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api",
        "api_key": "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",
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
        "api_key": "sk-byNbcxPlyWXrZnU43b7cDa3e4a1449958b24622dC328D955",
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
        "api_key": "ark-bfc41aa0-5b92-4f84-9692-aaab1f823126-93e25",
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
        "api_key": "sk-612908087c0d44c48afcb1b84746b179",  # 用户需要自己填入 API Key
        "models": {
            "image": [],  # DeepSeek 不支持图像生成
            "text": ["deepseek-chat", "deepseek-coder"]
        }
    }
}

# 当前选择的平台和模型（可通过API修改）
_current_platform = "openai"
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


def set_platform(platform: str, image_model: Optional[str] = None, text_model: Optional[str] = None, text_platform: Optional[str] = None):
    """设置当前使用的平台和模型"""
    global _current_platform, _current_image_model, _current_text_platform, _current_text_model
    if platform not in PLATFORMS:
        raise ValueError(f"不支持的平台: {platform}")
    
    _current_platform = platform
    config = PLATFORMS[platform]
    
    if image_model:
        if platform == "openai" and image_model in {"dall-e-3", "dall-e-2"}:
            image_model = "gpt-image-1"
        if image_model not in config["models"].get("image", []):
            raise ValueError(f"平台 {platform} 不支持图像模型: {image_model}")
        _current_image_model = image_model
    elif config["models"].get("image"):
        _current_image_model = config["models"]["image"][0]
    
    # 文本模型可以独立选择平台
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
    return PLATFORMS[_current_platform]["api_key"]


def _get_text_api_key() -> str:
    """获取当前文本平台的API Key"""
    return PLATFORMS[_current_text_platform]["api_key"]


# ─────────────────────────────────────────────
# 1. Prompt Agent
# ─────────────────────────────────────────────

_SYSTEM_PROMPT = """你是一名专业的建筑效果图提示词工程师。

你的任务是对用户的描述进行**轻微润色**，保持用户原意，只做必要的优化。

规则：
1. 保持用户的核心意图不变，只做微调
2. 如果用户描述已经很完整，直接返回原文即可
3. 适当补充：视角、光线、材质等建筑渲染要素（如果用户未提及）
4. 输出80-150字的中文描述，直接输出，不加任何前缀或解释
5. 如果有历史prompt，在原有基础上微调，不要大幅改动
6. 结尾可加上：照片级建筑效果图, 8k分辨率
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

    user_msg = (
        f"项目：{context['project_name']}\n\n"
        f"设计迭代路径（root → 当前）：\n{history_text}\n\n"
        f"路径上的已选图片记忆：\n{selected_text}\n\n"
        f"本次新需求：{context['new_user_input']}\n\n"
        f"设计师风格偏好：{weights_text}"
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
    style_weights = context.get("style_weights", {}) or {}
    new_user_input = (context.get("new_user_input") or "").strip()

    memory_inputs = [h.get("user_input", "").strip() for h in history if h.get("user_input")]
    memory_prompts = [h.get("prompt_used", "").strip() for h in history if h.get("prompt_used")]
    selected_revised = [x.get("revised_prompt", "").strip() for x in selected_images if x.get("revised_prompt")]

    top_styles = sorted(style_weights.items(), key=lambda x: x[1], reverse=True)[:6]
    top_styles = [k for k, v in top_styles if v >= 0.3]

    parts = []
    if memory_inputs:
        parts.append("Design evolution memory from previous nodes: " + " | ".join(memory_inputs[-6:]) + ".")
    if memory_prompts:
        parts.append("Established visual decisions: " + " | ".join(memory_prompts[-3:]) + ".")
    if selected_revised:
        parts.append("Selected image references to preserve: " + " | ".join(selected_revised[-4:]) + ".")
    if top_styles:
        parts.append("Preferred style cues: " + ", ".join(top_styles) + ".")
    if new_user_input:
        parts.append("New refinement request: " + new_user_input + ".")

    parts.append(
        "Keep architectural consistency while refining details. "
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


def _format_weights(weights: dict) -> str:
    if not weights:
        return "暂无"
    top = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:6]
    return "、".join(f"{k}({v:+.1f})" for k, v in top)


# ─────────────────────────────────────────────
# 2. 图像生成
# ─────────────────────────────────────────────

def generate_images(
    prompt: str,
    n: int = 4,
    size: str = "1792x1024",
    quality: str = "standard",
    save_dir: Optional[Path] = None,
) -> list:
    """
    调用图像API生成图片，支持多平台。
    返回 [{url, local_path, revised_prompt}, ...]
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

    # 优先当前模型，OpenAI失败时可回退到 gpt-image-1
    model_candidates = [_current_image_model]
    platform_image_models = PLATFORMS[platform].get("models", {}).get("image", [])
    if platform_image_models:
        default_model = platform_image_models[0]
        if default_model not in model_candidates:
            model_candidates.append(default_model)
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

            results.append({
                "url": f"/static/uploads/{filename}",
                "local_path": str(local_path),
                "revised_prompt": item.get("revised_prompt", prompt),
            })
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

                if _is_model_not_found_error(err_text) and mi < len(model_candidates) - 1:
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
        raise RuntimeError(f"{platform} {resp.status_code}: {resp.text[:400]}")
    return resp.json()
