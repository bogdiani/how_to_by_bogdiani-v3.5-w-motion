# tts_engines.py
import os
from pathlib import Path
from openai import OpenAI

def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    org = os.getenv("OPENAI_ORG_ID") or None
    return OpenAI(api_key=api_key, organization=org)

def synthesize_openai_tts(out_path: str, text: str) -> None:
    """
    OpenAI TTS.
    Env (optional):
      OPENAI_TTS_MODEL = gpt-4o-mini-tts (default)
      OPENAI_TTS_VOICE = alloy (default)
      OPENAI_TTS_FORMAT = mp3 (default)
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    model = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
    voice = os.getenv("OPENAI_TTS_VOICE", "alloy")
    fmt = os.getenv("OPENAI_TTS_FORMAT", "mp3").lower()

    # normalize extension to chosen format
    if out.suffix.lower().lstrip(".") != fmt:
        out = out.with_suffix(f".{fmt}")

    client = _get_openai_client()
    print(f"[TTS] OpenAI: model={model} voice={voice} format={fmt} → {out}")

    # 1) Try streaming API
    try:
        with client.audio.speech.with_streaming_response.create(
            model=model, voice=voice, input=text, format=fmt
        ) as resp:
            stream_to_file = getattr(resp, "stream_to_file", None)
            if callable(stream_to_file):
                stream_to_file(str(out))
            else:
                # iterate bytes
                with open(out, "wb") as f:
                    for chunk in resp.iter_bytes():
                        f.write(chunk)
        size = out.stat().st_size if out.exists() else 0
        if size == 0:
            raise RuntimeError("OpenAI streaming TTS produced an empty file.")
        print(f"[TTS] Wrote {size} bytes → {out}")
        return
    except Exception as e:
        print(f"[TTS] Streaming path not available or failed: {e}; trying non-streaming...")

    # 2) Non-streaming: try response_format first, then format (covers both SDKs)
    try:
        try:
            resp = client.audio.speech.create(model=model, voice=voice, input=text, response_format=fmt)
        except TypeError:
            resp = client.audio.speech.create(model=model, voice=voice, input=text, format=fmt)

        content = getattr(resp, "content", None)
        if content is None:
            read = getattr(resp, "read", None)
            if callable(read):
                content = read()
        if not content:
            raise RuntimeError("OpenAI non-streaming TTS returned no content.")

        with open(out, "wb") as f:
            f.write(content)

        size = out.stat().st_size if out.exists() else 0
        if size == 0:
            raise RuntimeError("OpenAI non-streaming TTS produced an empty file.")
        print(f"[TTS] Wrote {size} bytes → {out}")
    except Exception as e:
        print(f"[TTS][ERROR] OpenAI synthesis failed: {e}")
        raise

def synthesize_offline_tts(out_path: str, text: str, voice_name: str | None = None) -> None:
    """
    Offline TTS via pyttsx3 (optional dependency).
    """
    try:
        import pyttsx3
    except ImportError:
        raise RuntimeError(
            "Offline TTS requested but 'pyttsx3' is not installed. "
            "Use --tts openai or add 'pyttsx3==2.90' to requirements.txt."
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    engine = pyttsx3.init()
    if voice_name:
        for v in engine.getProperty("voices"):
            if voice_name.lower() in (v.id.lower(), v.name.lower()):
                engine.setProperty("voice", v.id)
                break

    wav_path = out.with_suffix(".wav")
    print(f"[TTS] Offline (pyttsx3) → {wav_path}")
    engine.save_to_file(text, str(wav_path))
    engine.runAndWait()

    # Convert to mp3 via ffmpeg if requested
    if out.suffix.lower() == ".mp3":
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path), str(out)],
            check=True
        )
        try:
            wav_path.unlink()
        except Exception:
            pass
    size = out.stat().st_size if out.exists() else 0
    print(f"[TTS] Offline wrote {size} bytes → {out}")
