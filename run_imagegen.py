"""JAY Image Generation Addon — CPU text-to-image / image-to-image.

Generates on the server (thread-capped, single-flight), serves the PNG on a
dedicated HTTP port, tells a PC to open the URL, and auto-deletes the file
after a TTL so storage never grows.

Runtime-configurable (no rebuild): the active model, NSFW/safety, default
steps and img2img strength can all be changed live via the /config endpoint.
Model + NSFW changes reload the pipeline in-place; steps + strength apply per
request. Guidance is bound to each model preset automatically. Config is
persisted to the data volume so restarts keep your last choice.

MQTT contract:
  subscribes: skill/imagegen/request, skill/imagegen/config/set
  publishes:  tts/request, skill/imagegen/state, pc/command/<machine>

HTTP:
  GET  /img/<id>.png   serve a generated image
  GET  /config         current config + available models
  POST /config         update config (JSON body)
  POST /edit           img2img: raw image body, ?prompt=&strength=&steps=

Author: Alex Stan
"""
import os
import io
import re
import json
import time
import uuid
import glob
import logging
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# thread cap BEFORE importing torch so it takes effect
IMAGE_THREADS = int(os.getenv("IMAGE_THREADS", "4"))
os.environ.setdefault("OMP_NUM_THREADS", str(IMAGE_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(IMAGE_THREADS))

import numpy as np
import paho.mqtt.client as mqtt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("imagegen-addon")

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))

# ============================================================================
# MODEL PRESETS — each carries its own guidance + default steps so the UI
# never has to expose guidance. Add a line here to offer another model.
# ============================================================================
MODELS = {
    "dreamshaper": {"label": "DreamShaper v7 (LCM)",
                    "id": "SimianLuo/LCM_Dreamshaper_v7", "guidance": 8.0, "steps": 6},
    "sd15-lcm":    {"label": "SD 1.5 (LCM)",
                    "id": "qiacheng/stable-diffusion-v1-5-lcm", "guidance": 1.0, "steps": 6},
    "sd-turbo":    {"label": "SD-Turbo (fastest)",
                    "id": "stabilityai/sd-turbo", "guidance": 0.0, "steps": 1},
}
DEFAULT_MODEL_KEY = os.getenv("IMAGE_MODEL_KEY", "dreamshaper")
if DEFAULT_MODEL_KEY not in MODELS:
    DEFAULT_MODEL_KEY = "dreamshaper"

SIZE = int(os.getenv("IMAGE_SIZE", "512"))
ALLOWED_SIZES = {512, 768}
HTTP_PORT = int(os.getenv("IMAGE_HTTP_PORT", "8137"))
HOST_ADDR = os.getenv("IMAGE_HOST_ADDR", "100.119.255.57")  # tailnet IP for served URL
TTL_SECONDS = int(os.getenv("IMAGE_TTL_SECONDS", "600"))    # delete PNGs after 10 min
OUT_DIR = os.getenv("IMAGE_OUT_DIR", "/app/data/out")
DEFAULT_PC = os.getenv("DEFAULT_PC", "laptop")
SKILL = os.getenv("IMAGE_SKILL_TOPIC", "imagegen")

# Outpainting / inpainting uses a dedicated inpaint checkpoint (standard SD1.5,
# not LCM) so it needs more steps + normal guidance than the LCM gen models.
INPAINT_MODEL_ID = os.getenv("IMAGE_INPAINT_MODEL", "Lykon/dreamshaper-5-inpainting")
INPAINT_STEPS = int(os.getenv("INPAINT_STEPS", "24"))
INPAINT_GUIDANCE = float(os.getenv("INPAINT_GUIDANCE", "7.5"))
OUTPAINT_CAP = int(os.getenv("OUTPAINT_CAP", "768"))   # max long side of the extended canvas

# ControlNet-lineart (Stage C, experimental): regenerate following the drawn
# linework to preserve style. Loaded on demand; reuses the active gen model's
# components so it only adds the ControlNet weights (~1.5GB) to RAM.
CONTROLNET_MODEL_ID = os.getenv("IMAGE_CONTROLNET_MODEL", "lllyasviel/control_v11p_sd15_lineart")
LINEART_STEPS = int(os.getenv("LINEART_STEPS", "8"))

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================================
# RUNTIME CONFIG — mutable, persisted to the data volume.
# ============================================================================
CONFIG_PATH = os.path.join(os.path.dirname(OUT_DIR.rstrip("/")) or "/app/data", "config.json")
CFG_LOCK = threading.Lock()
CFG = {
    "model_key": DEFAULT_MODEL_KEY,
    "nsfw": os.getenv("ENABLE_NSFW", "false").lower() in ("1", "true", "yes"),
    "steps": None,   # None = use the active model's default
    "strength": float(os.getenv("IMG2IMG_STRENGTH", "0.6")),
    "outpaint_steps": INPAINT_STEPS,
    "outpaint_amount": float(os.getenv("OUTPAINT_AMOUNT", "0.4")),
    "line_lock": os.getenv("LINE_LOCK", "false").lower() in ("1", "true", "yes"),
}


def load_config():
    """Load persisted config over the defaults, if present and valid."""
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            saved = json.load(f)
        for k in CFG:
            if k in saved:
                CFG[k] = saved[k]
        if CFG["model_key"] not in MODELS:
            CFG["model_key"] = DEFAULT_MODEL_KEY
        logger.info("Loaded persisted config: %s", CFG)
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning("config load failed (%s), using defaults", e)


def save_config():
    try:
        tmp = CONFIG_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(CFG, f)
        os.replace(tmp, CONFIG_PATH)
    except Exception as e:
        logger.warning("config save failed: %s", e)


def cur_model():
    return MODELS.get(CFG["model_key"], MODELS[DEFAULT_MODEL_KEY])


def cur_steps():
    return int(CFG["steps"]) if CFG["steps"] else int(cur_model()["steps"])


def cur_guidance():
    return float(cur_model()["guidance"])


GEN_LOCK = threading.Lock()   # single-flight: one generation at a time
_pipe = None
_pipe_lock = threading.Lock()
_img2img_pipe = None
_img2img_lock = threading.Lock()
_inpaint_pipe = None
_inpaint_lock = threading.Lock()
_controlnet_pipe = None
_controlnet_lock = threading.Lock()
_cn_inpaint_pipe = None
_cn_inpaint_lock = threading.Lock()
READY = threading.Event()


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _load_pipe():
    """Load the active model once, keep resident. Heavy import kept lazy."""
    global _pipe
    with _pipe_lock:
        if _pipe is not None:
            return _pipe
        import torch
        torch.set_num_threads(IMAGE_THREADS)
        from diffusers import AutoPipelineForText2Image
        m = cur_model()
        nsfw = bool(CFG["nsfw"])
        logger.info("Loading %s (model=%s, nsfw=%s, threads=%d)...",
                    m["label"], m["id"], nsfw, IMAGE_THREADS)
        t0 = time.time()
        load_kw = dict(torch_dtype=torch.float32)
        if nsfw:
            # Turn OFF the model's built-in checker so images aren't blacked out.
            load_kw["safety_checker"] = None
            load_kw["requires_safety_checker"] = False
        p = AutoPipelineForText2Image.from_pretrained(m["id"], **load_kw)
        p.set_progress_bar_config(disable=True)
        _pipe = p
        READY.set()
        logger.info("Model ready in %.1fs (safety_checker=%s)",
                    time.time() - t0, "off" if nsfw else "on")
        return _pipe


def _load_img2img_pipe():
    """Build an img2img pipeline from the already-loaded model.
    Reuses the same weights — no extra download, minimal extra RAM."""
    global _img2img_pipe
    with _img2img_lock:
        if _img2img_pipe is not None:
            return _img2img_pipe
        base = _load_pipe()  # ensure the base model is loaded first
        from diffusers import AutoPipelineForImage2Image
        logger.info("Building img2img pipeline from loaded model (no extra download)...")
        p = AutoPipelineForImage2Image.from_pipe(base)
        p.set_progress_bar_config(disable=True)
        _img2img_pipe = p
        logger.info("img2img pipeline ready")
        return _img2img_pipe


def reset_pipes():
    """Drop the resident pipelines so the next load picks up new config."""
    global _pipe, _img2img_pipe
    with _pipe_lock:
        _pipe = None
    with _img2img_lock:
        _img2img_pipe = None
    READY.clear()


def reload_async():
    """Reload the pipeline in the background after a model/nsfw change."""
    def _run():
        try:
            reset_pipes()
            _load_pipe()
            publish_state("ready")
        except Exception as e:
            logger.exception("reload failed: %s", e)
            publish_state("error", error=str(e))
    threading.Thread(target=_run, daemon=True).start()


def _r8(x):
    """Round to nearest multiple of 8 (VAE requires it), min 8."""
    return max(8, int(round(x / 8)) * 8)


def fit_dims(w, h, cap=512):
    """Match input aspect ratio, cap the long side at `cap`, snap to /8.
    Keeps total pixels sane so CPU time doesn't blow up on tall/wide inputs."""
    if w >= h:
        out_w, out_h = cap, cap * h / w
    else:
        out_w, out_h = cap * w / h, cap
    return _r8(out_w), _r8(out_h)


def reap_old(_now=None):
    now = _now or time.time()
    for f in glob.glob(os.path.join(OUT_DIR, "*.png")):
        try:
            if now - os.path.getmtime(f) > TTL_SECONDS:
                os.remove(f)
                logger.info("Reaped %s", os.path.basename(f))
        except OSError:
            pass


def reaper_loop():
    while True:
        time.sleep(60)
        try:
            reap_old()
        except Exception as e:
            logger.warning("reaper: %s", e)


def _flagged_nsfw(result):
    """True only when the pipeline's checker flagged the image AND safety is on."""
    if CFG["nsfw"]:
        return False
    flags = getattr(result, "nsfw_content_detected", None)
    return bool(flags and any(flags))


def generate(prompt, steps=None):
    """Text-to-image. Returns (image_id, path) or raises."""
    pipe = _load_pipe()
    import torch
    reap_old()
    n_steps = int(steps) if steps else cur_steps()
    with GEN_LOCK:
        t0 = time.time()
        kwargs = dict(prompt=prompt, num_inference_steps=n_steps,
                      guidance_scale=cur_guidance(), height=SIZE, width=SIZE)
        with torch.no_grad():
            result = pipe(**kwargs)
        img = result.images[0]
        nsfw = _flagged_nsfw(result)
        logger.info("Generated in %.1fs (steps=%d, nsfw=%s, safety=%s)",
                    time.time() - t0, n_steps, nsfw, "ON" if not CFG["nsfw"] else "OFF")
    if nsfw:
        raise ValueError("blocked_nsfw")
    image_id = f"img_{uuid.uuid4().hex[:10]}"
    path = os.path.join(OUT_DIR, image_id + ".png")
    img.save(path)
    return image_id, path


def generate_img2img(prompt, image_bytes, strength=None, steps=None):
    """Image-to-image. Output matches input aspect ratio.
    Returns (image_id, path) or raises."""
    st = CFG["strength"] if strength is None else float(strength)
    st = min(max(st, 0.1), 1.0)
    n_steps = int(steps) if steps else cur_steps()
    # "keep aspect ratio" / full-res request: lift the long-side cap 512 -> 768.
    # (ratio is always matched; this raises resolution. Higher than 768 breaks CPU.)
    _kw = r"\b(keep aspect ratio|full res(?:olution)?|exact size|hi(?:gh)?[- ]?res)\b"
    keep = bool(re.search(_kw, prompt, re.I))
    cleaned = re.sub(_kw, "", prompt, flags=re.I).strip(" ,.")
    prompt = cleaned or prompt
    cap = 768 if keep else 512
    pipe = _load_img2img_pipe()
    import torch
    from PIL import Image
    reap_old()
    init = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    ow, oh = init.size
    w, h = fit_dims(ow, oh, cap)
    init = init.resize((w, h), Image.LANCZOS)
    with GEN_LOCK:
        t0 = time.time()
        kwargs = dict(prompt=prompt, image=init, num_inference_steps=n_steps,
                      guidance_scale=cur_guidance(), strength=st)
        with torch.no_grad():
            result = pipe(**kwargs)
        img = result.images[0]
        nsfw = _flagged_nsfw(result)
        logger.info("img2img in %.1fs (%dx%d from %dx%d, steps=%d, strength=%.2f, nsfw=%s)",
                    time.time() - t0, w, h, ow, oh, n_steps, st, nsfw)
    if nsfw:
        raise ValueError("blocked_nsfw")
    image_id = f"img_{uuid.uuid4().hex[:10]}"
    path = os.path.join(OUT_DIR, image_id + ".png")
    img.save(path)
    return image_id, path


def _load_inpaint_pipe():
    """Lazy-load the inpainting model (separate architecture from the gen models).
    Only loaded the first time an outpaint/inpaint is requested."""
    global _inpaint_pipe
    with _inpaint_lock:
        if _inpaint_pipe is not None:
            return _inpaint_pipe
        import torch
        torch.set_num_threads(IMAGE_THREADS)
        from diffusers import AutoPipelineForInpainting
        logger.info("Loading inpaint model %s (first use; may download)...", INPAINT_MODEL_ID)
        t0 = time.time()
        load_kw = dict(torch_dtype=torch.float32)
        if CFG["nsfw"]:
            load_kw["safety_checker"] = None
            load_kw["requires_safety_checker"] = False
        p = AutoPipelineForInpainting.from_pretrained(INPAINT_MODEL_ID, **load_kw)
        p.set_progress_bar_config(disable=True)
        _inpaint_pipe = p
        logger.info("Inpaint model ready in %.1fs", time.time() - t0)
        return _inpaint_pipe


def _f8(x):
    """Floor to a multiple of 8 (never rounds up past a budget), min 8."""
    return max(8, (int(x) // 8) * 8)


def plan_outpaint(ow, oh, direction, amount, cap=None):
    """Base size, per-side additions, final canvas — all /8, long side capped."""
    cap = cap or OUTPAINT_CAP
    amount = min(max(float(amount), 0.1), 1.0)
    base_cap = 512
    if ow >= oh:
        bw = min(base_cap, _r8(ow)); bh = _r8(bw * oh / ow)
    else:
        bh = min(base_cap, _r8(oh)); bw = _r8(bh * ow / oh)
    l = r = t = b = 0
    if direction == "down":
        b = _r8(bh * amount)
    elif direction == "up":
        t = _r8(bh * amount)
    elif direction == "left":
        l = _r8(bw * amount)
    elif direction == "right":
        r = _r8(bw * amount)
    elif direction == "all":
        l = r = _r8(bw * amount / 2); t = b = _r8(bh * amount / 2)
    else:
        b = _r8(bh * amount)
    nw, nh = bw + l + r, bh + t + b
    longest = max(nw, nh)
    if longest > cap:
        f = cap / longest
        bw, bh = _f8(bw * f), _f8(bh * f)
        l = _f8(l * f) if l else 0
        r = _f8(r * f) if r else 0
        t = _f8(t * f) if t else 0
        b = _f8(b * f) if b else 0
        nw, nh = bw + l + r, bh + t + b
    return dict(bw=bw, bh=bh, l=l, r=r, t=t, b=b, nw=nw, nh=nh)


def generate_outpaint(prompt, image_bytes, direction="down", amount=None, steps=None):
    """Extend the canvas and fill the new area. Original pixels preserved exactly.
    Returns (image_id, path) or raises."""
    amt = CFG["outpaint_amount"] if amount is None else float(amount)
    amt = min(max(amt, 0.1), 1.0)
    n_steps = int(steps) if steps else int(CFG["outpaint_steps"])
    pipe = _load_inpaint_pipe()
    import torch
    from PIL import Image, ImageDraw
    reap_old()
    src = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    ow, oh = src.size
    p = plan_outpaint(ow, oh, direction, amt)
    base = src.resize((p["bw"], p["bh"]), Image.LANCZOS)
    canvas = Image.new("RGB", (p["nw"], p["nh"]), (255, 255, 255))
    canvas.paste(base, (p["l"], p["t"]))
    mask = Image.new("L", (p["nw"], p["nh"]), 255)   # white = fill, black = keep
    ImageDraw.Draw(mask).rectangle(
        [p["l"], p["t"], p["l"] + p["bw"] - 1, p["t"] + p["bh"] - 1], fill=0)
    try:
        mask = pipe.mask_processor.blur(mask, blur_factor=8)  # soften the seam
    except Exception:
        pass
    with GEN_LOCK:
        t0 = time.time()
        with torch.no_grad():
            result = pipe(prompt=prompt, image=canvas, mask_image=mask,
                          num_inference_steps=n_steps, guidance_scale=INPAINT_GUIDANCE,
                          strength=1.0, height=p["nh"], width=p["nw"])
        out = result.images[0].convert("RGB")
        out.paste(base, (p["l"], p["t"]))   # keep original region pixel-exact
        nsfw = _flagged_nsfw(result)
        logger.info("outpaint in %.1fs (%s +%.0f%% -> %dx%d, steps=%d, nsfw=%s)",
                    time.time() - t0, direction, amt * 100, p["nw"], p["nh"], n_steps, nsfw)
    if nsfw:
        raise ValueError("blocked_nsfw")
    image_id = f"img_{uuid.uuid4().hex[:10]}"
    path = os.path.join(OUT_DIR, image_id + ".png")
    out.save(path)
    return image_id, path


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="jay-imagegen-addon")


def _parse_multipart(body, content_type):
    """Minimal multipart/form-data parser -> {field_name: bytes}."""
    m = re.search(r"boundary=([^;]+)", content_type or "")
    if not m:
        return {}
    boundary = m.group(1).strip('"').encode()
    out = {}
    for part in body.split(b"--" + boundary):
        if b"\r\n\r\n" not in part:
            continue
        head, data = part.split(b"\r\n\r\n", 1)
        nm = re.search(r'name="([^"]+)"', head.decode("latin1", "ignore"))
        if nm:
            out[nm.group(1)] = data.rstrip(b"\r\n")
    return out


def generate_inpaint(prompt, image_bytes, mask_bytes, strength=None, steps=None, line_lock=None):
    """Regenerate only the masked region (white=redo, black=keep). The unmasked
    area is composited back pixel-exact. line_lock adds ControlNet-lineart so the
    fill follows the drawing's linework (style-preserving, slower).
    Returns (image_id, path) or raises."""
    lock = CFG["line_lock"] if line_lock is None else bool(line_lock)
    pipe = _load_cn_inpaint_pipe() if lock else _load_inpaint_pipe()
    import torch
    from PIL import Image
    reap_old()
    src = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    ow, oh = src.size
    w, h = fit_dims(ow, oh)
    src = src.resize((w, h), Image.LANCZOS)
    mask = Image.open(io.BytesIO(mask_bytes)).convert("L").resize((w, h), Image.LANCZOS)
    try:
        mask_soft = pipe.mask_processor.blur(mask, blur_factor=8)
    except Exception:
        mask_soft = mask
    st = 1.0 if strength is None else min(max(float(strength), 0.1), 1.0)
    n_steps = int(steps) if steps else INPAINT_STEPS
    with GEN_LOCK:
        t0 = time.time()
        with torch.no_grad():
            if lock:
                control = _lineart_control(src)
                result = pipe(prompt=prompt, image=src, mask_image=mask_soft,
                              control_image=control, num_inference_steps=n_steps,
                              guidance_scale=INPAINT_GUIDANCE, strength=st, height=h, width=w)
            else:
                result = pipe(prompt=prompt, image=src, mask_image=mask_soft,
                              num_inference_steps=n_steps, guidance_scale=INPAINT_GUIDANCE,
                              strength=st, height=h, width=w)
        out = result.images[0].convert("RGB")
        out = Image.composite(out, src, mask)   # keep unmasked area exact
        nsfw = _flagged_nsfw(result)
        logger.info("inpaint in %.1fs (%dx%d, steps=%d, strength=%.2f, linelock=%s, nsfw=%s)",
                    time.time() - t0, w, h, n_steps, st, lock, nsfw)
    if nsfw:
        raise ValueError("blocked_nsfw")
    image_id = f"img_{uuid.uuid4().hex[:10]}"
    path = os.path.join(OUT_DIR, image_id + ".png")
    out.save(path)
    return image_id, path


def _lineart_control(pil):
    """Turn a drawing into a ControlNet lineart map: white lines on black."""
    from PIL import ImageOps
    g = ImageOps.autocontrast(ImageOps.grayscale(pil))
    return ImageOps.invert(g).convert("RGB")


def _load_controlnet_pipe():
    """Lazy-load ControlNet-lineart, reusing the active gen model's components
    so only the ControlNet weights are added to RAM."""
    global _controlnet_pipe
    with _controlnet_lock:
        if _controlnet_pipe is not None:
            return _controlnet_pipe
        base = _load_pipe()
        import torch
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
        logger.info("Loading ControlNet %s (first use; may download ~1.5GB)...", CONTROLNET_MODEL_ID)
        t0 = time.time()
        cn = ControlNetModel.from_pretrained(CONTROLNET_MODEL_ID, torch_dtype=torch.float32)
        p = StableDiffusionControlNetPipeline(
            vae=base.vae, text_encoder=base.text_encoder, tokenizer=base.tokenizer,
            unet=base.unet, scheduler=base.scheduler, controlnet=cn,
            safety_checker=getattr(base, "safety_checker", None),
            feature_extractor=getattr(base, "feature_extractor", None),
            requires_safety_checker=False)
        p.set_progress_bar_config(disable=True)
        _controlnet_pipe = p
        logger.info("ControlNet ready in %.1fs", time.time() - t0)
        return _controlnet_pipe


def _load_cn_inpaint_pipe():
    """Lazy-load ControlNet + inpaint together (style-locked region edits),
    reusing the inpaint model's components so only the ControlNet is added."""
    global _cn_inpaint_pipe
    with _cn_inpaint_lock:
        if _cn_inpaint_pipe is not None:
            return _cn_inpaint_pipe
        base = _load_inpaint_pipe()
        import torch
        from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
        logger.info("Loading ControlNet-inpaint (%s; first use)...", CONTROLNET_MODEL_ID)
        t0 = time.time()
        cn = ControlNetModel.from_pretrained(CONTROLNET_MODEL_ID, torch_dtype=torch.float32)
        p = StableDiffusionControlNetInpaintPipeline(
            vae=base.vae, text_encoder=base.text_encoder, tokenizer=base.tokenizer,
            unet=base.unet, controlnet=cn, scheduler=base.scheduler,
            safety_checker=getattr(base, "safety_checker", None),
            feature_extractor=getattr(base, "feature_extractor", None),
            requires_safety_checker=False)
        p.set_progress_bar_config(disable=True)
        _cn_inpaint_pipe = p
        logger.info("ControlNet-inpaint ready in %.1fs", time.time() - t0)
        return _cn_inpaint_pipe


def generate_lineart(prompt, image_bytes, steps=None):
    """Regenerate an image following its own linework (style-preserving).
    Returns (image_id, path) or raises."""
    pipe = _load_controlnet_pipe()
    import torch
    from PIL import Image
    reap_old()
    src = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    ow, oh = src.size
    w, h = fit_dims(ow, oh)
    src = src.resize((w, h), Image.LANCZOS)
    control = _lineart_control(src)
    n_steps = int(steps) if steps else LINEART_STEPS
    with GEN_LOCK:
        t0 = time.time()
        with torch.no_grad():
            result = pipe(prompt=prompt, image=control, num_inference_steps=n_steps,
                          guidance_scale=cur_guidance(), height=h, width=w)
        img = result.images[0]
        nsfw = _flagged_nsfw(result)
        logger.info("lineart in %.1fs (%dx%d, steps=%d, nsfw=%s)",
                    time.time() - t0, w, h, n_steps, nsfw)
    if nsfw:
        raise ValueError("blocked_nsfw")
    image_id = f"img_{uuid.uuid4().hex[:10]}"
    path = os.path.join(OUT_DIR, image_id + ".png")
    img.save(path)
    return image_id, path


def publish_tts(text, source=None):
    payload = {"text": text}
    if source:
        payload["source"] = source
    client.publish("tts/request", json.dumps(payload), qos=1)


def config_snapshot():
    """Current effective config + model catalog, for the UI."""
    m = cur_model()
    return {
        "models": [{"key": k, "label": v["label"]} for k, v in MODELS.items()],
        "model_key": CFG["model_key"],
        "model_label": m["label"],
        "nsfw": bool(CFG["nsfw"]),
        "steps": cur_steps(),
        "strength": round(float(CFG["strength"]), 2),
        "outpaint_steps": int(CFG["outpaint_steps"]),
        "outpaint_amount": round(float(CFG["outpaint_amount"]), 2),
        "line_lock": bool(CFG["line_lock"]),
        "ready": READY.is_set(),
    }


def publish_state(status, **extra):
    payload = {"skill": SKILL, "timestamp": utc_now_iso(), "status": status, "size": SIZE}
    payload.update(config_snapshot())
    payload["safety"] = not CFG["nsfw"]  # back-compat field
    payload.update(extra)
    client.publish(f"skill/{SKILL}/state", json.dumps(payload), qos=1, retain=True)


def apply_config(patch):
    """Apply a partial config update. Returns (snapshot, reloaded_bool)."""
    reload_needed = False
    with CFG_LOCK:
        if "model_key" in patch and patch["model_key"] in MODELS \
                and patch["model_key"] != CFG["model_key"]:
            CFG["model_key"] = patch["model_key"]
            reload_needed = True
        if "nsfw" in patch:
            nv = bool(patch["nsfw"])
            if nv != CFG["nsfw"]:
                CFG["nsfw"] = nv
                reload_needed = True
        if "steps" in patch and patch["steps"] is not None:
            try:
                CFG["steps"] = max(1, min(12, int(patch["steps"])))
            except (TypeError, ValueError):
                pass
        if "strength" in patch and patch["strength"] is not None:
            try:
                CFG["strength"] = max(0.1, min(1.0, float(patch["strength"])))
            except (TypeError, ValueError):
                pass
        if "outpaint_steps" in patch and patch["outpaint_steps"] is not None:
            try:
                CFG["outpaint_steps"] = max(4, min(40, int(patch["outpaint_steps"])))
            except (TypeError, ValueError):
                pass
        if "outpaint_amount" in patch and patch["outpaint_amount"] is not None:
            try:
                CFG["outpaint_amount"] = max(0.1, min(1.0, float(patch["outpaint_amount"])))
            except (TypeError, ValueError):
                pass
        if "line_lock" in patch and patch["line_lock"] is not None:
            CFG["line_lock"] = bool(patch["line_lock"])
        save_config()
    if reload_needed:
        reload_async()
    return config_snapshot(), reload_needed


def handle_request(req):
    prompt = (req.get("prompt") or req.get("text") or "").strip()
    data = req.get("data") or {}
    if not prompt and data.get("subject"):
        prompt = data["subject"].strip()
    machine = (req.get("machine") or data.get("machine") or "").strip()
    # A trailing "on laptop/desktop" is the target machine, not part of the image.
    _mt = re.search(r"\s+on\s+(desktop|laptop)\s*$", prompt, re.I)
    if _mt:
        if not machine:
            machine = _mt.group(1).lower()
        prompt = prompt[:_mt.start()].strip()
    machine = machine or DEFAULT_PC
    source = req.get("source")
    if not prompt:
        publish_tts("What should I generate an image of?", source)
        return
    publish_state("generating", prompt=prompt, machine=machine)
    publish_tts(f"Generating an image of {prompt}. This takes a moment.", source)
    try:
        image_id, path = generate(prompt)
    except ValueError as ve:
        if str(ve) == "blocked_nsfw":
            publish_state("blocked")
            publish_tts("That image was blocked by the safety filter.", source)
            return
        raise
    except Exception as e:
        logger.exception("generation failed: %s", e)
        publish_state("error", error=str(e))
        publish_tts("Sorry, I couldn't generate that image.", source)
        return
    url = f"http://{HOST_ADDR}:{HTTP_PORT}/img/{image_id}.png"
    client.publish(f"pc/command/{machine}",
                   json.dumps({"action": "open_url", "url": url}), qos=1)
    publish_state("done", url=url, machine=machine, prompt=prompt)
    publish_tts(f"Here's your image on {machine}.", source)
    logger.info("Served %s -> %s (open on %s)", image_id, url, machine)


def on_connect(client, userdata, flags, reason_code, properties):
    logger.info("Connected to MQTT %s:%s", MQTT_HOST, MQTT_PORT)
    client.subscribe(f"skill/{SKILL}/request", qos=1)
    client.subscribe(f"skill/{SKILL}/config/set", qos=1)
    logger.info("Subscribed to skill/%s/request and /config/set", SKILL)
    publish_state("ready" if READY.is_set() else "loading")


def on_message(client, userdata, msg):
    try:
        if msg.topic == f"skill/{SKILL}/request":
            req = json.loads(msg.payload.decode() or "{}")
            threading.Thread(target=handle_request, args=(req,), daemon=True).start()
        elif msg.topic == f"skill/{SKILL}/config/set":
            patch = json.loads(msg.payload.decode() or "{}")
            apply_config(patch)
    except Exception as e:
        logger.exception("on_message: %s", e)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        from urllib.parse import urlparse
        path = urlparse(self.path).path
        if path == "/config":
            self._json(config_snapshot()); return
        if not path.startswith("/img/"):
            self.send_error(404); return
        name = os.path.basename(path[len("/img/"):])
        if not name.endswith(".png") or "/" in name or "\\" in name:
            self.send_error(400); return
        fpath = os.path.join(OUT_DIR, name)
        if not os.path.exists(fpath):
            self.send_error(404, "expired or unknown"); return
        try:
            with open(fpath, "rb") as f:
                data = f.read()
        except OSError:
            self.send_error(404); return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else b""

        if parsed.path == "/config":
            try:
                patch = json.loads(body.decode() or "{}")
            except Exception:
                self._json({"ok": False, "error": "bad json"}, 400); return
            snap, reloaded = apply_config(patch)
            snap["ok"] = True
            snap["reloading"] = reloaded
            self._json(snap); return

        if parsed.path == "/generate":
            q = parse_qs(parsed.query)
            prompt = (q.get("prompt", [""])[0]).strip()
            steps_raw = q.get("steps", [None])[0]
            if not prompt:
                self.send_error(400, "missing prompt"); return
            try:
                image_id, path = generate(prompt, steps_raw)
            except ValueError as ve:
                if str(ve) == "blocked_nsfw":
                    self.send_error(403, "blocked by safety filter"); return
                self.send_error(400, str(ve)); return
            except Exception as e:
                logger.exception("generate failed: %s", e)
                self.send_error(500, "generation failed"); return
            try:
                with open(path, "rb") as f:
                    data = f.read()
            except OSError:
                self.send_error(500); return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            logger.info("Served text2img %s", image_id)
            return

        if parsed.path == "/inpaint":
            ctype = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in ctype:
                self.send_error(400, "expected multipart (image + mask)"); return
            fields = _parse_multipart(body, ctype)
            image_b = fields.get("image") or fields.get("file")
            mask_b = fields.get("mask")
            prompt = (fields.get("prompt", b"").decode("utf-8", "ignore")).strip()
            strength = (fields.get("strength", b"").decode("utf-8", "ignore")).strip() or None
            steps_f = (fields.get("steps", b"").decode("utf-8", "ignore")).strip() or None
            ll_raw = (fields.get("line_lock", b"").decode("utf-8", "ignore")).strip().lower()
            line_lock = None if ll_raw == "" else ll_raw in ("1", "true", "yes", "on")
            if not image_b or not mask_b:
                self.send_error(400, "need image and mask"); return
            if not prompt:
                prompt = "same style"
            try:
                image_id, path = generate_inpaint(prompt, image_b, mask_b, strength, steps_f, line_lock)
            except ValueError as ve:
                if str(ve) == "blocked_nsfw":
                    self.send_error(403, "blocked by safety filter"); return
                self.send_error(400, str(ve)); return
            except Exception as e:
                logger.exception("inpaint failed: %s", e)
                self.send_error(500, "inpaint failed"); return
            try:
                with open(path, "rb") as f:
                    data = f.read()
            except OSError:
                self.send_error(500); return
            url = f"http://{HOST_ADDR}:{HTTP_PORT}/img/{image_id}.png"
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Image-Url", url)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            logger.info("Served inpaint %s -> %s", image_id, url)
            return

        if parsed.path == "/lineart":
            q = parse_qs(parsed.query)
            prompt = (q.get("prompt", [""])[0]).strip()
            steps_raw = q.get("steps", [None])[0]
            if length <= 0:
                self.send_error(400, "no image body"); return
            if not prompt:
                prompt = "clean line art, monochrome, same style"
            try:
                image_id, path = generate_lineart(prompt, body, steps_raw)
            except ValueError as ve:
                if str(ve) == "blocked_nsfw":
                    self.send_error(403, "blocked by safety filter"); return
                self.send_error(400, str(ve)); return
            except Exception as e:
                logger.exception("lineart failed: %s", e)
                self.send_error(500, "lineart failed"); return
            try:
                with open(path, "rb") as f:
                    data = f.read()
            except OSError:
                self.send_error(500); return
            url = f"http://{HOST_ADDR}:{HTTP_PORT}/img/{image_id}.png"
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Image-Url", url)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            logger.info("Served lineart %s -> %s", image_id, url)
            return

        if parsed.path == "/outpaint":
            q = parse_qs(parsed.query)
            prompt = (q.get("prompt", [""])[0]).strip()
            direction = (q.get("direction", ["down"])[0]).strip().lower()
            amount_raw = q.get("amount", [None])[0]   # None -> CFG default
            try:
                amount = float(amount_raw) if amount_raw not in (None, "") else None
            except ValueError:
                amount = None
            steps_raw = q.get("steps", [None])[0]
            if length <= 0:
                self.send_error(400, "no image body"); return
            if not prompt:
                prompt = "extend the image, same style"
            if direction not in ("down", "up", "left", "right", "all"):
                direction = "down"
            try:
                image_id, path = generate_outpaint(prompt, body, direction, amount, steps_raw)
            except ValueError as ve:
                if str(ve) == "blocked_nsfw":
                    self.send_error(403, "blocked by safety filter"); return
                self.send_error(400, str(ve)); return
            except Exception as e:
                logger.exception("outpaint failed: %s", e)
                self.send_error(500, "outpaint failed"); return
            try:
                with open(path, "rb") as f:
                    data = f.read()
            except OSError:
                self.send_error(500); return
            url = f"http://{HOST_ADDR}:{HTTP_PORT}/img/{image_id}.png"
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Image-Url", url)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            logger.info("Served outpaint %s -> %s", image_id, url)
            return

        if parsed.path != "/edit":
            self.send_error(404); return

        q = parse_qs(parsed.query)
        prompt = (q.get("prompt", [""])[0]).strip()
        strength_raw = q.get("strength", [None])[0]
        steps_raw = q.get("steps", [None])[0]
        if length <= 0:
            self.send_error(400, "no image body"); return
        if not prompt:
            self.send_error(400, "missing prompt"); return
        try:
            image_id, path = generate_img2img(prompt, body, strength_raw, steps_raw)
        except ValueError as ve:
            if str(ve) == "blocked_nsfw":
                self.send_error(403, "blocked by safety filter"); return
            self.send_error(400, str(ve)); return
        except Exception as e:
            logger.exception("img2img failed: %s", e)
            self.send_error(500, "generation failed"); return
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            self.send_error(500); return
        url = f"http://{HOST_ADDR}:{HTTP_PORT}/img/{image_id}.png"
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Image-Url", url)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)
        logger.info("Served img2img %s -> %s", image_id, url)


def serve_http():
    httpd = ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), Handler)
    logger.info("Image HTTP server on :%d (serving %s)", HTTP_PORT, OUT_DIR)
    httpd.serve_forever()


if __name__ == "__main__":
    load_config()
    m = cur_model()
    logger.info("JAY Image Generation Addon starting "
                "(model=%s, steps=%d, strength=%.2f, size=%d, threads=%d, safety=%s)",
                m["label"], cur_steps(), CFG["strength"], SIZE, IMAGE_THREADS,
                "ON" if not CFG["nsfw"] else "OFF")
    threading.Thread(target=serve_http, daemon=True).start()
    threading.Thread(target=reaper_loop, daemon=True).start()
    threading.Thread(target=_load_pipe, daemon=True).start()
    _dl = time.time() + 60
    while True:
        try:
            client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
            break
        except (ConnectionRefusedError, OSError):
            if time.time() >= _dl:
                raise
            time.sleep(1)
    client.on_connect = on_connect
    client.on_message = on_message
    client.loop_forever()