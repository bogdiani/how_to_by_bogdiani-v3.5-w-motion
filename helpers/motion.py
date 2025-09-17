# helpers/motion.py
import math
from moviepy.editor import ImageClip, CompositeVideoClip, vfx

def ken_burns(img_clip: ImageClip, w: int, h: int, duration: float,
              zoom_start=1.05, zoom_end=1.15,
              pan_start=(0.5, 0.45), pan_end=(0.5, 0.55)):
    """
    Simple Ken Burns (pan + zoom).
    """
    def ease(t):
        x = t / max(duration, 0.001)
        return 0.5 - 0.5 * math.cos(math.pi * x)

    def make_frame(t):
        p = ease(t)
        zoom = zoom_start + (zoom_end - zoom_start) * p
        cx = w * (pan_start[0] + (pan_end[0] - pan_start[0]) * p)
        cy = h * (pan_start[1] + (pan_end[1] - pan_start[1]) * p)

        frame = img_clip.get_frame(0)  # base still
        ch, cw = frame.shape[0:2]
        crop_w, crop_h = int(cw / zoom), int(ch / zoom)
        x0 = int(cx - crop_w / 2)
        y0 = int(cy - crop_h / 2)
        x0 = max(0, min(cw - crop_w, x0))
        y0 = max(0, min(ch - crop_h, y0))
        cropped = frame[y0:y0+crop_h, x0:x0+crop_w]

        from moviepy.video.fx.resize import resize
        return resize(ImageClip(cropped), (w, h)).get_frame(0)

    kb = ImageClip(img_clip.get_frame(0)).set_duration(duration)
    kb = kb.fl(make_frame, apply_to=[])
    return kb


def parallax_drift(img_clip: ImageClip, w: int, h: int, duration: float):
    """
    Creates subtle parallax drift by layering two copies.
    """
    bg = img_clip.resize(1.08)
    fg = img_clip.resize(1.02)

    def pos_bg(t):
        return ("center", h/2 + 10 * math.sin(0.6 * t))

    def pos_fg(t):
        return ("center", h/2 - 10 * math.sin(0.6 * t + math.pi/2))

    bg = bg.set_position(pos_bg)
    fg = fg.set_position(pos_fg)

    return CompositeVideoClip([bg, fg], size=(w, h)).set_duration(duration)


def handheld(img_clip: ImageClip, w: int, h: int, duration: float,
             amp_px=6, amp_deg=0.4):
    """
    Simulates handheld camera wobble.
    """
    def tf(get_frame, t):
        import cv2, math
        frame = get_frame(0)
        dx = amp_px * math.sin(0.9*t) + amp_px * 0.5 * math.sin(1.7*t)
        dy = amp_px * math.cos(0.8*t) + amp_px * 0.5 * math.cos(1.3*t)
        ang = amp_deg * math.sin(0.6*t)

        hgt, wdt = frame.shape[:2]
        M = cv2.getRotationMatrix2D((wdt/2, hgt/2), ang, 1.0)
        shifted = cv2.warpAffine(frame, M, (wdt, hgt),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT101)
        return shifted

    return img_clip.fl(tf, keep_duration=True)
