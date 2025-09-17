
# JSON → Video (Vertical) Generator

Create 9:16 videos (TikTok/Reels/Shorts) from a simple JSON script.

## Features
- 9:16 vertical output (1080×1920)
- Section-based timing (Intro, Steps, Outro)
- Voiceover via **OpenAI TTS** (high quality) or **pyttsx3** (offline fallback)
- Visuals via **Pexels API** (optional) or auto-generated title cards
- Burned-in captions (optional) and section titles
- CLI **and** Streamlit web app

---

## Quick Start (CLI)

1) **Install system dependency**
   - You need [FFmpeg](https://ffmpeg.org/). On macOS: `brew install ffmpeg`. On Windows: use winget/choco. On Linux: your package manager.

2) **Create & activate a virtualenv (recommended)**
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

3) **Install Python deps**
```bash
pip install -r requirements.txt
```

4) **Configure keys (optional, but recommended for best results)**
   - Copy `.env.example` to `.env` and fill in:
     - `OPENAI_API_KEY` for high-quality voiceover
     - `PEXELS_API_KEY` for stock images/videos
```bash
cp .env.example .env
# then edit .env
```

5) **Put your JSON script at** `assets/script.json` (an example is included).

6) **Run the pipeline**
```bash
python app.py --input assets/script.json --output outputs/video.mp4 --fps 30 --preset reels
```
- `--preset reels` sets 1080x1920 and font sizing tuned for mobile.
- The script will:
  - Generate voiceovers per section (OpenAI TTS if key is present, else pyttsx3)
  - Fetch visuals from Pexels (if key) or generate clean title cards
  - Compose everything into a single MP4

7) Find the output in `outputs/video.mp4`.

---

## Web UI (Streamlit)

```bash
streamlit run streamlit_app.py
```
- Upload your JSON, click **Render**, and download the result.
- Uses the same pipeline under the hood.

---

## JSON Format

```json
{
  "topic": "How to Make Pancakes",
  "sections": [
    {
      "title": "Intro",
      "narration": "Exact spoken line here...",
      "visual_suggestions": ["Visual idea 1", "Visual idea 2"],
      "duration_seconds": 6
    },
    {
      "title": "Step 1: ...",
      "narration": "...",
      "visual_suggestions": ["..."],
      "duration_seconds": 12
    }
  ],
  "total_estimated_duration": 74
}
```

> Tip: Keep total duration between 60–120s for best completion rate on Shorts/Reels.

---

## Notes & Tips
- If Pexels finds multiple results, a random best-match is chosen; you can lock in assets by placing your own media in `assets/media/` and referencing them with `--use-local-media`.
- To change fonts/colors, tweak constants in `pipeline.py`.
- If you hit TTS character limits, batch-run `app.py` with `--tts offline` to use pyttsx3.
- For smoother edits, keep each step between 6–15s.

---

## Troubleshooting
- **FFmpeg not found**: Ensure it’s on PATH. `ffmpeg -version` should work.
- **OpenAI auth error**: Check `OPENAI_API_KEY` in `.env`.
- **Pexels empty results**: The pipeline will fall back to title cards automatically.
- **Fonts missing**: The defaults are safe; you can specify a font path with `--font-path`.

Happy creating!
