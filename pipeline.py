# pipeline.py
import json
import os
import inspect
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# --- Pillow ANTIALIAS compatibility for MoviePy (Pillow 10+ removed ANTIALIAS) ---
try:
    import PIL.Image as _PILImage
    if not hasattr(_PILImage, "ANTIALIAS"):
        _PILImage.ANTIALIAS = _PILImage.Resampling.LANCZOS  # back-compat alias
except Exception:
    pass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from moviepy.editor import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    concatenate_videoclips,
    vfx,
)

# Proper audio fx imports (use .fx(...) not .audio_fx)
from moviepy.audio.fx.audio_fadein import audio_fadein
from moviepy.audio.fx.audio_fadeout import audio_fadeout

from media_sources import get_media_for_section  # we'll wrap this with a compat shim
from tts_engines import synthesize_openai_tts, synthesize_offline_tts


# ==============================
# Utility
# ==============================

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def _get_env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return str(v).lower() == "true"

def _load_json(p: Path) -> Dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _calc_canvas(preset: str) -> Tuple[int, int]:
    preset = (preset or "reels").lower()
    if preset in ("reels", "tiktok", "shorts"):
        return (1080, 1920)
    if preset == "portrait1024":
        return (1024, 1536)
    return (1080, 1920)

def _wrap_text_to_width(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    if not text:
        return []
    words = text.split()
    lines, cur = [], []
    img = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(img)

    def line_w(s: str) -> int:
        return int(draw.textlength(s, font=font))

    for w in words:
        trial = (" ".join(cur + [w])).strip()
        if not trial:
            continue
        if line_w(trial) <= max_width or not cur:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines

def _subtitle_clip(
    text: str,
    w: int,
    h: int,
    duration: float,
    font_path: Optional[str] = None,
    margin_ratio: float = 0.06,
    box_alpha: int = 190,
) -> ImageClip:
    if not text:
        arr = np.zeros((1, 1, 4), dtype=np.uint8)
        return ImageClip(arr, transparent=True).set_duration(duration)

    horiz_margin = int(w * margin_ratio)
    max_text_width = w - 2 * horiz_margin
    base_font_px = max(30, int(w * 0.04))
    if font_path:
        try:
            font = ImageFont.truetype(font_path, base_font_px)
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    lines = _wrap_text_to_width(text, font, max_text_width)
    if not lines:
        lines = [text]

    line_heights = []
    dummy = Image.new("RGB", (10, 10))
    d_draw = ImageDraw.Draw(dummy)
    for ln in lines:
        bbox = font.getbbox(ln or " ")
        line_heights.append(bbox[3] - bbox[1])

    line_spacing = max(12, int(base_font_px * 0.35))
    text_block_h = sum(line_heights) + line_spacing * (len(lines) - 1)

    pad_y = max(16, int(base_font_px * 0.5))
    pad_x = max(18, int(base_font_px * 0.6))
    box_w = w - 2 * horiz_margin
    box_h = text_block_h + pad_y * 2

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    box_left = horiz_margin
    box_top = h - int(h * 0.18) - box_h // 2
    box_right = box_left + box_w
    box_bottom = box_top + box_h
    radius = max(12, int(base_font_px * 0.6))
    draw.rounded_rectangle(
        [box_left, box_top, box_right, box_bottom],
        radius=radius,
        fill=(0, 0, 0, box_alpha)
    )

    cur_y = box_top + pad_y
    for i, ln in enumerate(lines):
        tw = int(d_draw.textlength(ln, font=font))
        tx = box_left + (box_w - tw) // 2
        draw.text((tx, cur_y), ln, font=font, fill=(255, 255, 255, 255),
                  stroke_width=2, stroke_fill=(0, 0, 0, 255))
        cur_y += line_heights[i] + line_spacing

    arr = np.asarray(overlay).astype("uint8")
    return ImageClip(arr, transparent=True).set_duration(duration).set_position(("center", "center"))

def _ensure_mp3(path: Path) -> Path:
    if path.suffix.lower() != ".mp3":
        return path.with_suffix(".mp3")
    return path


# ==============================
# Compatibility shim for media_sources.get_media_for_section
# ==============================

def _compat_get_media_for_section(
    title: str,
    narration: str,
    visual_ideas: Optional[List[str]],
    w: int,
    h: int,
    min_duration: float,
    font_path: Optional[str],
    use_local_media: Optional[bool],
):
    """
    Calls media_sources.get_media_for_section with whatever parameters
    it supports (old or new signature). Also maps 'min_duration'->'duration'
    if the older function name is used.
    """
    sig = inspect.signature(get_media_for_section)
    params = sig.parameters

    # Build a kw dict containing only the names it accepts
    kwargs = {}
    mapping = {
        "title": title,
        "narration": narration,
        "visual_ideas": visual_ideas,
        "w": w,
        "h": h,
        "min_duration": min_duration,
        "duration": min_duration,  # older versions used 'duration'
        "font_path": font_path,
        "use_local_media": use_local_media,
    }

    for name in mapping:
        if name in params:
            kwargs[name] = mapping[name]

    # If neither 'min_duration' nor 'duration' is present, fall back to positional lenient call
    try:
        return get_media_for_section(**kwargs)
    except TypeError as e:
        # Last-ditch: try a minimal positional call based on what exists
        # (title, w, h, duration) is the smallest sensible set
        print(f"[Compat] Direct kwargs call failed ({e}); trying positional fallback.")
        try:
            if {"title", "w", "h", "duration"} <= set(params.keys()):
                return get_media_for_section(title=title, w=w, h=h, duration=min_duration)
            elif {"title", "w", "h"} <= set(params.keys()):
                clip = get_media_for_section(title=title, w=w, h=h)
                # Ensure duration
                return clip.set_duration(min_duration)
            else:
                raise
        except Exception as e2:
            raise RuntimeError(f"media_sources.get_media_for_section signature mismatch: {params.keys()} (error: {e2})")


# ==============================
# Core render
# ==============================

def render_from_json(
    input_json: str | None = None,
    output_path: str = "",
    fps: int = 30,
    preset: str = "reels",
    tts_engine: str = "openai",
    data: dict | None = None,
    tts_mode: str | None = None,      # backward-compat alias
    font_path: str | None = None,     # optional font path
    use_local_media: bool | None = None,  # override env if provided
):
    """
    Render a video from JSON instructions.
    Accepts either input_json path or pre-parsed data.
    """

    # Backward-compatible alias for callers using tts_mode
    if tts_mode is not None and (not tts_engine or tts_engine == "openai"):
        tts_engine = tts_mode

    # Load data
    if data is None:
        if not input_json:
            raise RuntimeError("render_from_json: provide either `data` or `input_json`.")
        data = _load_json(Path(input_json))

    sections: List[Dict] = data.get("sections", [])
    if not sections:
        raise RuntimeError("No sections found in input JSON/data.")

    # Canvas
    w, h = _calc_canvas(preset)
    print(f"[Config] preset={preset} → {w}x{h}")

    # Flags (env with optional override)
    env_use_local = _get_env_bool("USE_LOCAL_MEDIA", False)
    if use_local_media is None:
        use_local_media = env_use_local

    burn_subtitles = _get_env_bool("BURN_SUBTITLES", False)
    xfade_seconds = float(os.getenv("XFADE_SECONDS", "0.4") or "0.4")

    print(f"[Config] USE_LOCAL_MEDIA={use_local_media}")
    print(f"[Config] BURN_SUBTITLES={burn_subtitles}")
    print(f"[Config] XFADE_SECONDS={xfade_seconds}")

    # Paths
    temp_dir = _ensure_dir(Path("temp"))
    voice_dir = _ensure_dir(temp_dir / "voice")
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Font selection
    if font_path:
        font_to_use = font_path
    else:
        candidate = Path("assets/fonts/Inter-SemiBold.ttf")
        font_to_use = str(candidate) if candidate.exists() else None

    # Build clips
    video_clips = []

    for i, sec in enumerate(sections, start=1):
        title = sec.get("title", f"Section {i}")
        narration = sec.get("narration", "")
        visuals = sec.get("visual_suggestions", []) or []
        target_duration = float(sec.get("duration_seconds", 3.0) or 3.0)

        # TTS
        voice_path = _ensure_mp3(voice_dir / f"sec_{i:02d}.mp3")
        tts_ok = False
        try:
            if tts_engine == "openai":
                synthesize_openai_tts(str(voice_path), narration)
                tts_ok = True
            elif tts_engine == "offline":
                synthesize_offline_tts(str(voice_path), narration)
                tts_ok = True
            else:
                try:
                    synthesize_openai_tts(str(voice_path), narration)
                    tts_ok = True
                except Exception as e:
                    print(f"[WARN] OpenAI TTS failed ({e}); trying offline.")
                    synthesize_offline_tts(str(voice_path), narration)
                    tts_ok = True
        except Exception as e:
            print(f"[WARN] TTS failed ({e}); no voice will be attached for '{title}'.")

        voice_clip = None
        voice_dur = 0.0
        if tts_ok and voice_path.exists() and voice_path.stat().st_size > 0:
            try:
                voice_clip = AudioFileClip(str(voice_path))
                voice_dur = float(voice_clip.duration)
                print(f"[TTS] Section {i} '{title}': duration={voice_dur:.2f}s, file={voice_path.name}")
            except Exception as e:
                print(f"[WARN] Could not load TTS audio for '{title}': {e}")

        # Ensure visuals last as long as voice (plus tiny headroom)
        min_dur = max(target_duration, voice_dur + 0.05)

        # Visuals (robust call regardless of media_sources signature)
        media_clip = _compat_get_media_for_section(
            title=title,
            narration=narration,
            visual_ideas=visuals,
            w=w,
            h=h,
            min_duration=min_dur,
            font_path=font_to_use,
            use_local_media=use_local_media,
        )

        # Ensure desired duration
        media_clip = media_clip.set_duration(min_dur)

        # Subtitles
        if burn_subtitles and narration:
            sub = _subtitle_clip(narration, w, h, duration=min_dur, font_path=font_to_use)
            media_clip = CompositeVideoClip([media_clip, sub]).set_duration(min_dur)

        # Attach audio with tiny fades if we actually have it
        if voice_clip is not None:
            a = voice_clip.fx(audio_fadein, 0.06).fx(audio_fadeout, 0.06)
            media_clip = media_clip.set_audio(a)
        else:
            print(f"[Audio] Section {i} '{title}': no voice attached.")

        # Visual fade edges to assist crossfades
        media_clip = media_clip.fx(vfx.fadein, max(0.01, min(0.25, xfade_seconds))).fx(
            vfx.fadeout, max(0.01, min(0.25, xfade_seconds))
        )

        video_clips.append(media_clip)

    if not video_clips:
        raise RuntimeError("No video clips could be constructed from the sections.")

    # Crossfade concatenate
    xfade = max(0.0, float(xfade_seconds))
    if xfade > 0:
        xclips = [video_clips[0]]
        for c in video_clips[1:]:
            xclips.append(c.crossfadein(xfade))
        final = concatenate_videoclips(xclips, method="compose", padding=-xfade)
    else:
        final = concatenate_videoclips(video_clips, method="compose")

    # Export
    final.write_videofile(
        str(out_path),
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        threads=0,
        temp_audiofile=str(Path("temp") / "videoTEMP_MPY_aud.m4a"),
        remove_temp=True,
        verbose=False,
        logger=None,
    )

    # Cleanup
    try:
        for c in video_clips:
            if c.audio is not None and isinstance(c.audio, CompositeAudioClip):
                c.audio.close()
    except Exception:
        pass


# ==============================
# CLI
# ==============================

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Render video from JSON script")
    ap.add_argument("--input", help="Path to JSON script")
    ap.add_argument("--output", required=True, help="Output video path")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--preset", default="reels")
    ap.add_argument("--tts", default="openai")
    ap.add_argument("--font", help="Optional font path")
    ap.add_argument("--use-local-media", dest="use_local_media", action="store_true")
    ap.add_argument("--no-use-local-media", dest="use_local_media", action="store_false")
    ap.set_defaults(use_local_media=None)
    args = ap.parse_args()

    render_from_json(
        input_json=args.input,
        output_path=args.output,
        fps=args.fps,
        preset=args.preset,
        tts_engine=args.tts,
        font_path=args.font,
        use_local_media=args.use_local_media,
    )

if __name__ == "__main__":
    main()
