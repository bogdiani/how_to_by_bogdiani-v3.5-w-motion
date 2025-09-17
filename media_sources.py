# media_sources.py
import os
import io
import requests
import numpy as np
from PIL import Image, ImageDraw
from moviepy.editor import ImageClip, CompositeVideoClip
from helpers.motion import ken_burns, parallax_drift, handheld

USE_LOCAL_MEDIA = os.getenv("USE_LOCAL_MEDIA", "false").lower() == "true"
USE_OPENAI_IMAGES = os.getenv("USE_OPENAI_IMAGES", "false").lower() == "true"
USE_STABILITY_IMAGES = os.getenv("USE_STABILITY_IMAGES", "false").lower() == "true"
STABILITY_ENGINE = os.getenv("STABILITY_ENGINE", "core")
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
OPENAI_IMAGE_STYLE = os.getenv("OPENAI_IMAGE_STYLE", "photorealistic")

def _explain_fallback(reason=""):
    print(f"[media_sources] FALLBACK reason → {reason}")
    img = Image.new("RGB", (1080, 1920), (40, 40, 40))
    d = ImageDraw.Draw(img)
    d.text((50, 50), "No media found", fill=(200, 200, 200))
    arr = np.array(img)
    return ImageClip(arr).set_duration(0.1)

def _apply_motion(clip: ImageClip, w: int, h: int, duration: float) -> ImageClip:
    preset = os.getenv("MOTION_PRESET", "kenburns").lower()
    try:
        if preset == "kenburns":
            return ken_burns(clip, w, h, duration)
        elif preset == "parallax":
            return parallax_drift(clip, w, h, duration)
        elif preset == "handheld":
            return handheld(clip, w, h, duration)
        else:
            return clip
    except Exception as e:
        print(f"[media_sources] Motion effect failed: {e}")
        return clip

def get_media_for_section(title, narration, visual_ideas,
                          w, h, min_duration,
                          font_path=None, use_local_media=False):
    """
    Centralized media fetcher for each section.
    """
    print(f"[media_sources] get_media_for_section → {title}")

    # -- 1. Local media
    if USE_LOCAL_MEDIA or use_local_media:
        folder = "assets/media"
        if os.path.isdir(folder):
            files = [f for f in os.listdir(folder)
                     if f.lower().endswith((".jpg", ".jpeg", ".png", ".mp4"))]
            if files:
                path = os.path.join(folder, files[0])
                print(f"[media_sources] Using local: {path}")
                if path.endswith(".mp4"):
                    from moviepy.editor import VideoFileClip
                    return VideoFileClip(path).subclip(0, min_duration).resize((w, h))
                else:
                    img = Image.open(path).convert("RGB")
                    arr = np.array(img)
                    clip = ImageClip(arr).resize((w, h)).set_duration(min_duration)
                    return _apply_motion(clip, w, h, min_duration)

    # -- 2. Stability AI (if enabled)
    if USE_STABILITY_IMAGES:
        try:
            prompt = None
            if visual_ideas:
                if isinstance(visual_ideas, list):
                    prompt_parts = []
                    for v in visual_ideas:
                        if isinstance(v, dict) and "prompt" in v:
                            prompt_parts.append(v["prompt"])
                        elif isinstance(v, str):
                            prompt_parts.append(v)
                    prompt = ", ".join(prompt_parts)
                elif isinstance(visual_ideas, str):
                    prompt = visual_ideas
            if not prompt:
                raise ValueError("No valid visual_suggestions")

            print(f"[media_sources] [Stability] Requesting {STABILITY_ENGINE} with prompt: {prompt}")
            resp = requests.post(
                f"https://api.stability.ai/v2beta/stable-image/generate/{STABILITY_ENGINE}",
                headers={
                    "Authorization": f"Bearer {os.getenv('STABILITY_API_KEY')}",
                    "Accept": "application/json"
                },
                files={"none": ''},
                data={"prompt": prompt, "output_format": "png"},
                timeout=60
            )
            if resp.status_code == 200:
                out = resp.json()
                if "image" in out:
                    import base64
                    img = Image.open(io.BytesIO(base64.b64decode(out["image"])))
                    arr = np.array(img.convert("RGB"))
                    clip = ImageClip(arr).resize((w, h)).set_duration(min_duration)
                    return _apply_motion(clip, w, h, min_duration)
            else:
                print(f"[media_sources] [Stability] FAIL {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"[media_sources] Stability error: {e}")

    # -- 3. OpenAI Images (if enabled)
    if USE_OPENAI_IMAGES:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            prompt = None
            if visual_ideas:
                if isinstance(visual_ideas, list):
                    prompt_parts = []
                    for v in visual_ideas:
                        if isinstance(v, dict) and "prompt" in v:
                            prompt_parts.append(v["prompt"])
                        elif isinstance(v, str):
                            prompt_parts.append(v)
                    prompt = ", ".join(prompt_parts)
                elif isinstance(visual_ideas, str):
                    prompt = visual_ideas
            if not prompt:
                raise ValueError("No valid visual_suggestions")

            print(f"[media_sources] [OpenAI] Prompt: {prompt}")
            result = client.images.generate(
                model=OPENAI_IMAGE_MODEL,
                prompt=prompt,
                size="1024x1024",
                n=1
            )
            img_url = result.data[0].url
            arr = np.array(Image.open(io.BytesIO(requests.get(img_url, timeout=60).content)))
            clip = ImageClip(arr).resize((w, h)).set_duration(min_duration)
            return _apply_motion(clip, w, h, min_duration)
        except Exception as e:
            print(f"[media_sources] OpenAI error: {e}")

    # -- 4. Fallback
    return _explain_fallback("All media sources failed.")
