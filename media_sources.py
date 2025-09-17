# media_sources.py
import os
import io
import json
import math
import hashlib
import random
import pathlib
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw

from moviepy.editor import (
    ImageClip,
    VideoFileClip,
    CompositeVideoClip,
    vfx,
)

# ---------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------
def _log(msg: str):
    print(f"[media_sources] {msg}", flush=True)


# ---------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------
def _get_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).lower() == "true"

def _get_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None else str(v)

# Feature flags / knobs
USE_OPENAI_IMAGES   = _get_bool("USE_OPENAI_IMAGES", False)
USE_STABILITY       = _get_bool("USE_STABILITY_IMAGES", False)
STRICT_VISUALS_ONLY = _get_bool("STRICT_VISUALS_ONLY", True)  # <- default ON per your request
STABILITY_ENGINE    = _get_str("STABILITY_ENGINE", "core")     # "core", "sd3", etc.
OPENAI_IMAGE_MODEL  = _get_str("OPENAI_IMAGE_MODEL", "gpt-image-1")
OPENAI_IMAGE_STYLE  = _get_str("OPENAI_IMAGE_STYLE", "photorealistic")

# Stock source preferences (kept for parity; they fail gracefully if no keys)
USE_PEXELS          = _get_bool("USE_PEXELS", False)
USE_PIXABAY         = _get_bool("USE_PIXABAY", False)
PREFERRED_STOCK     = _get_str("PREFERRED_STOCK_SOURCE", "auto")

# Local media acceptance
LOCAL_MEDIA_DIR     = pathlib.Path("assets/media")
IMAGE_EXTS          = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTS          = {".mp4", ".mov", ".m4v"}

# Pillow ANTIALIAS back-compat
try:
    import PIL.Image as _PILImage
    if not hasattr(_PILImage, "ANTIALIAS"):
        _PILImage.ANTIALIAS = _PILImage.Resampling.LANCZOS
except Exception:
    pass


# ---------------------------------------------------------------------
# Visual prompt handling
# ---------------------------------------------------------------------
def _extract_prompt_from_visuals(visual_ideas: Union[List[Union[str, dict]], str, None]) -> Optional[str]:
    """
    Turn visual_suggestions into a single prompt string. Accepts:
      - ["text", {"prompt": "text"}, ...]
      - "text"
      - None / empty
    Returns a clean prompt string or None if nothing usable.
    """
    if not visual_ideas:
        return None

    if isinstance(visual_ideas, str):
        s = visual_ideas.strip()
        return s or None

    prompts: List[str] = []
    for item in visual_ideas:
        if isinstance(item, str):
            s = item.strip()
            if s:
                prompts.append(s)
        elif isinstance(item, dict):
            # Prefer explicit "prompt" key if present
            if "prompt" in item and isinstance(item["prompt"], str):
                s = item["prompt"].strip()
                if s:
                    prompts.append(s)
            else:
                # Fallback: flatten other key/values into a phrase (best-effort)
                try:
                    txt = ", ".join(f"{k}: {v}" for k, v in item.items())
                    if txt.strip():
                        prompts.append(txt.strip())
                except Exception:
                    pass

    return ", ".join(prompts) if prompts else None


def _combined_prompt_for_models(
    title: str,
    narration: str,
    visual_ideas: Union[List[Union[str, dict]], str, None],
    w: int,
    h: int,
) -> str:
    """
    Build the prompt string sent to image models.
    If STRICT_VISUALS_ONLY is True: use only visual_suggestions (+ fixed style & aspect).
    Else: include title/narration context as lightweight hints.
    """
    aspect = "vertical 9:16" if h >= w else "horizontal 16:9"

    core = _extract_prompt_from_visuals(visual_ideas)
    if not core:
        # If STRICT and no visual suggestions, return empty here;
        # caller will decide to fallback or try another source.
        if STRICT_VISUALS_ONLY:
            return ""

        # Non-strict: synthesize something from title/narration
        pieces = []
        if title:
            pieces.append(f"title: {title}")
        if narration:
            pieces.append(f"narration: {narration[:240]}")  # avoid blowing up the prompt
        core = ", ".join(pieces) if pieces else "simple subject, clean background"

    style_tail = f"— style: {OPENAI_IMAGE_STYLE}; composition: {aspect}; avoid text overlays."
    return f"{core}. {style_tail}"


# ---------------------------------------------------------------------
# Fallback clip (animated gradient)
# ---------------------------------------------------------------------
def _animated_gradient_clip(w: int, h: int, duration: float) -> ImageClip:
    """
    A tiny animated gradient fallback that always returns a valid MoviePy clip.
    (No PIL shape errors; we return a function-based ImageClip via make_frame.)
    """
    def make_frame(t):
        # shift hues over time
        t = float(t)
        phase = (t / max(duration, 0.001)) * 2 * math.pi
        # simple two-color lerp across rows
        col_top = np.array([50 + 50*math.sin(phase), 80, 180 + 50*math.cos(phase)])
        col_bot = np.array([180, 50 + 50*math.cos(phase), 80])
        grad = np.linspace(col_top, col_bot, h).astype(np.uint8)
        frame = np.tile(grad[:, None, :], (1, w, 1))
        return frame

    clip = ImageClip(make_frame(0), ismask=False).set_duration(duration)
    # Convert to dynamically generated frames
    clip = clip.fl_time(lambda t: t, apply_to=[]).set_make_frame(make_frame)
    return clip


def _fallback(reason: str, w: int, h: int, duration: float) -> ImageClip:
    _log(f"FALLBACK reason → {reason}")
    return _animated_gradient_clip(w, h, duration)


# ---------------------------------------------------------------------
# Local media support
# ---------------------------------------------------------------------
def _list_local_media() -> List[pathlib.Path]:
    if not LOCAL_MEDIA_DIR.exists():
        return []
    files = []
    for p in LOCAL_MEDIA_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in (IMAGE_EXTS | VIDEO_EXTS):
            files.append(p)
    return sorted(files)

def _pick_local_media(files: List[pathlib.Path], key: str) -> Optional[pathlib.Path]:
    if not files:
        return None
    # Stable choice per-section using a hash of the title (or key)
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
    return files[h % len(files)]

def _clip_from_local_path(path: pathlib.Path, w: int, h: int, duration: float) -> ImageClip:
    ext = path.suffix.lower()
    if ext in VIDEO_EXTS:
        try:
            v = VideoFileClip(str(path)).without_audio()
            v = v.resize(height=h) if h >= w else v.resize(width=w)
            # ensure duration (loop or trim)
            if v.duration >= duration:
                return v.subclip(0, duration)
            # loop
            loops = int(math.ceil(duration / max(v.duration, 0.1)))
            pieces = []
            for i in range(loops):
                pieces.append(v.copy())
            out = pieces[0]
            for p in pieces[1:]:
                out = out.fx(vfx.concatenate_videoclips, [out, p]) if hasattr(vfx, "concatenate_videoclips") else out
            return out.subclip(0, duration)
        except Exception as e:
            _log(f"[Local media] video load failed: {e}; falling back to image frame.")
            # Fall through to static image fallback
    # Image
    img = Image.open(str(path)).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    return ImageClip(arr).resize((w, h)).set_duration(duration)


# ---------------------------------------------------------------------
# OpenAI Images (optional / off by default)
# ---------------------------------------------------------------------
def _try_openai_image(prompt: str, w: int, h: int, duration: float) -> Optional[ImageClip]:
    """
    Minimal, defensive OpenAI image call (disabled by default).
    If you enable USE_OPENAI_IMAGES and have a valid key, this will try to generate
    one PNG and return it as a clip. Silent if anything goes wrong.
    """
    if not USE_OPENAI_IMAGES:
        return None

    import base64
    import requests

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        _log("[OpenAI] no API key; skipping")
        return None

    try:
        _log("[OpenAI] requesting image")
        # Resize hint; gpt-image-1 supports sizes like 1024x1024/1792x1024/1024x1792, etc.
        size = "1024x1792" if h >= w else "1792x1024"

        resp = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENAI_IMAGE_MODEL,
                "prompt": prompt,
                "size": size,
                "response_format": "b64_json",
            },
            timeout=60,
        )
        if resp.status_code != 200:
            _log(f"[OpenAI] FAIL {resp.status_code}: {resp.text[:300]}")
            return None

        data = resp.json()
        b64 = data["data"][0]["b64_json"]
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        return ImageClip(arr).resize((w, h)).set_duration(duration)
    except Exception as e:
        _log(f"[OpenAI] ERROR {e}")
        return None


# ---------------------------------------------------------------------
# Stability Images
# ---------------------------------------------------------------------
def _try_stability_image(prompt: str, w: int, h: int, duration: float) -> Optional[ImageClip]:
    """
    Stability v2beta stable-image/generate/<engine> call with:
      - Accept: image/*
      - multipart/form-data (forced by a dummy files part)
    """
    if not USE_STABILITY:
        return None

    import requests

    api_key = os.getenv("STABILITY_API_KEY", "")
    if not api_key:
        _log("[Stability] no API key; skipping")
        return None

    try:
        _log(f"[Stability] Requesting {STABILITY_ENGINE}")
        # pick a reasonable aspect
        aspect_ratio = "9:16" if h >= w else "16:9"

        headers = {
            "Authorization": f"Bearer {api_key}",
            # IMPORTANT per your logs: must be 'image/*' or 'application/json'
            "Accept": "image/*",
        }
        data = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "output_format": "png",
        }

        # Force multipart/form-data by including a 'files' part
        resp = requests.post(
            f"https://api.stability.ai/v2beta/stable-image/generate/{STABILITY_ENGINE}",
            headers=headers,
            files={"none": ("", "")},  # enforce multipart/form-data
            data=data,
            timeout=90,
        )

        if resp.status_code == 200:
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            arr = np.array(img, dtype=np.uint8)
            return ImageClip(arr).resize((w, h)).set_duration(duration)
        else:
            _log(f"[Stability] FAIL {resp.status_code}: {resp.text}")
            return None
    except Exception as e:
        _log(f"[Stability] ERROR {e}")
        return None


# ---------------------------------------------------------------------
# Pexels / Pixabay (optional; minimal + defensive)
# ---------------------------------------------------------------------
def _search_pexels_image(query: str) -> Optional[Image.Image]:
    if not USE_PEXELS:
        return None
    import requests
    key = os.getenv("PEXELS_API_KEY", "")
    if not key:
        return None
    try:
        r = requests.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": key},
            params={"query": query, "per_page": 1, "orientation": "portrait"},
            timeout=30,
        )
        if r.status_code != 200:
            _log(f"[Pexels] FAIL {r.status_code}: {r.text[:200]}")
            return None
        js = r.json()
        if js.get("photos"):
            url = js["photos"][0]["src"]["large"]
            img = Image.open(io.BytesIO(requests.get(url, timeout=30).content)).convert("RGB")
            return img
    except Exception as e:
        _log(f"[Pexels] ERROR {e}")
    return None

def _search_pixabay_image(query: str) -> Optional[Image.Image]:
    if not USE_PIXABAY:
        return None
    import requests
    key = os.getenv("PIXABAY_API_KEY", "")
    if not key:
        return None
    try:
        r = requests.get(
            "https://pixabay.com/api/",
            params={"key": key, "q": query, "image_type": "photo", "orientation": "vertical", "per_page": 3},
            timeout=30,
        )
        if r.status_code != 200:
            _log(f"[Pixabay] FAIL {r.status_code}: {r.text[:200]}")
            return None
        js = r.json()
        hits = js.get("hits", [])
        if hits:
            url = hits[0].get("largeImageURL") or hits[0].get("webformatURL")
            if url:
                img = Image.open(io.BytesIO(requests.get(url, timeout=30).content)).convert("RGB")
                return img
    except Exception as e:
        _log(f"[Pixabay] ERROR {e}")
    return None


# ---------------------------------------------------------------------
# Public entry point used by pipeline.py
# ---------------------------------------------------------------------
def get_media_for_section(
    title: str,
    narration: str,
    visual_ideas: Union[List[Union[str, dict]], str, None],
    w: int,
    h: int,
    min_duration: float,
    font_path: Optional[str] = None,        # kept for signature parity; unused here
    use_local_media: Optional[bool] = None, # when None, read env
):
    """
    Returns a MoviePy clip for the section.
    Priority:
      1) Local media (if enabled)
      2) Stability (if enabled)
      3) OpenAI Images (if enabled)
      4) Stock (Pexels/Pixabay) if configured
      5) Animated gradient fallback
    """
    # Decide local-media flag
    use_local = _get_bool("USE_LOCAL_MEDIA", False) if use_local_media is None else bool(use_local_media)
    _log(f"USE_LOCAL_MEDIA={use_local}")

    # Build the model prompt based ONLY on visual suggestions if STRICT_VISUALS_ONLY is True
    prompt = _combined_prompt_for_models(
        title=title,
        narration=("" if STRICT_VISUALS_ONLY else (narration or "")),
        visual_ideas=visual_ideas,
        w=w,
        h=h,
    )

    # If STRICT and we ended up with empty prompt AND no local media, skip models and explain fallback
    if STRICT_VISUALS_ONLY and (not prompt) and not use_local:
        return _fallback("No valid visual_suggestions and local media disabled.", w, h, max(0.1, float(min_duration)))

    # 1) Local media
    if use_local:
        files = _list_local_media()
        _log(f"[Local media] Found {len(files)} file(s) in {LOCAL_MEDIA_DIR}")
        p = _pick_local_media(files, key=f"{title}-{prompt or ''}")
        if p:
            try:
                return _clip_from_local_path(p, w, h, max(0.1, float(min_duration)))
            except Exception as e:
                _log(f"[Local media] failed: {e}")

    # 2) Stability
    clip = _try_stability_image(prompt, w, h, max(0.1, float(min_duration)))
    if clip is not None:
        return clip

    # 3) OpenAI Images
    clip = _try_openai_image(prompt, w, h, max(0.1, float(min_duration)))
    if clip is not None:
        return clip

    # 4) Stock (if enabled) — image only for simplicity
    #    We'll hit either Pexels or Pixabay first depending on PREFERRED_STOCK_SOURCE
    if USE_PEXELS or USE_PIXABAY:
        stock_pref = (PREFERRED_STOCK or "auto").lower()
        order = []
        if stock_pref == "pexels":
            order = ["pexels", "pixabay"]
        elif stock_pref == "pixabay":
            order = ["pixabay", "pexels"]
        else:
            order = ["pexels", "pixabay"]

        img = None
        for src in order:
            if src == "pexels":
                img = _search_pexels_image(prompt or title or "abstract")
                if img is not None:
                    _log("[Pexels] using image")
                    break
            if src == "pixabay":
                img = _search_pixabay_image(prompt or title or "abstract")
                if img is not None:
                    _log("[Pixabay] using image")
                    break

        if img is not None:
            arr = np.array(img.convert("RGB"), dtype=np.uint8)
            return ImageClip(arr).resize((w, h)).set_duration(max(0.1, float(min_duration)))

    # 5) Fallback
    return _fallback("All media sources failed.", w, h, max(0.1, float(min_duration)))
