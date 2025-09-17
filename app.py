import argparse
import json
import os
from pathlib import Path

from pipeline import render_from_json

def load_json(path: str):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"JSON not found: {p.resolve()}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to JSON script")
    ap.add_argument("--output", default="outputs/video.mp4", help="Output MP4 path")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--preset", default="reels", choices=["reels", "tiktok", "shorts"])
    ap.add_argument("--tts", default="auto", choices=["auto", "openai", "offline"])
    return ap.parse_args()

def str2bool(s: str, default=False) -> bool:
    if s is None:
        return default
    return str(s).strip().lower() in {"1","true","yes","y","on"}

def main():
    args = parse_args()
    data = load_json(args.input)

    # Toggle from env (wired by workflow)
    use_local_media = str2bool(os.getenv("USE_LOCAL_MEDIA", "false"))
    # Optional: show a summary so logs are obvious
    print(f"[Config] USE_LOCAL_MEDIA={use_local_media}")
    print(f"[Config] USE_OPENAI_IMAGES={os.getenv('USE_OPENAI_IMAGES')}")
    print(f"[Config] USE_PEXELS={os.getenv('USE_PEXELS')}")
    print(f"[Config] USE_PIXABAY={os.getenv('USE_PIXABAY')}")
    print(f"[Config] PREFERRED_STOCK_SOURCE={os.getenv('PREFERRED_STOCK_SOURCE')}")
    print(f"[Config] BURN_SUBTITLES={os.getenv('BURN_SUBTITLES')}")
    print(f"[Config] XFADE_SECONDS={os.getenv('XFADE_SECONDS')}")

    # Helpful warnings for local media
    media_dir = Path("assets/media")
    if use_local_media:
        if not media_dir.exists():
            print(f"[Local media] WARNING: {media_dir} does not exist. Creating it.")
            media_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Count supported files
            exts = {".jpg",".jpeg",".png",".webp",".bmp",".mp4",".mov",".mkv",".webm"}
            candidates = [p for p in media_dir.rglob("*.*") if p.suffix.lower() in exts]
            print(f"[Local media] Found {len(candidates)} file(s) in {media_dir}")

    render_from_json(
        data=data,
        output_path=args.output,
        fps=args.fps,
        preset=args.preset,
        tts_engine=args.tts,
        font_path=None,
        use_local_media=use_local_media,
    )

if __name__ == "__main__":
    main()
