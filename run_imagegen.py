"""JAY Image Generation Addon — CPU sd-turbo text-to-image.

Generates on the server (thread-capped, single-flight), serves the PNG on a
dedicated HTTP port, tells a PC to open the URL, and auto-deletes the file
after a TTL so storage never grows.

Safety is ALWAYS ON. There is no toggle. Any image the checker flags is blocked.

This is a JAY addon. It plugs into JAY purely over MQTT:
  subscribes: skill/imagegen/request
  publishes:  tts/request, skill/imagegen/state, pc/command/<machine>

Author: Alex Stan
"""
import os
import io
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
MODEL_ID = os.getenv("IMAGE_MODEL", "stabilityai/sd-turbo")
STEPS = int(os.getenv("IMAGE_STEPS", "4"))
SIZE = int(os.getenv("IMAGE_SIZE", "512"))
ALLOWED_SIZES = {512, 768}
HTTP_PORT = int(os.getenv("IMAGE_HTTP_PORT", "8137"))
HOST_ADDR = os.getenv("IMAGE_HOST_ADDR", "100.119.255.57")  # tailnet IP for served URL
TTL_SECONDS = int(os.getenv("IMAGE_TTL_SECONDS", "600"))    # delete PNGs after 10 min
OUT_DIR = os.getenv("IMAGE_OUT_DIR", "/app/data/out")
DEFAULT_PC = os.getenv("DEFAULT_PC", "laptop")

# Skill topic this addon answers on. Matches skill_topic in addon.manifest.json.
SKILL = os.getenv("IMAGE_SKILL_TOPIC", "imagegen")

os.makedirs(OUT_DIR, exist_ok=True)

GEN_LOCK = threading.Lock()   # single-flight: one generation at a time
_pipe = None
_pipe_lock = threading.Lock()


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _load_pipe():
    """Load sd-turbo once, keep resident. Heavy import kept lazy."""
    global _pipe
    with _pipe_lock:
        if _pipe is not None:
            return _pipe
        import torch
        torch.set_num_threads(IMAGE_THREADS)
        from diffusers import AutoPipelineForText2Image
        logger.info("Loading %s (threads=%d)...", MODEL_ID, IMAGE_THREADS)
        t0 = time.time()
        p = AutoPipelineForText2Image.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        p.set_progress_bar_config(disable=True)
        _pipe = p
        logger.info("Model ready in %.1fs", time.time() - t0)
        return _pipe


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


def generate(prompt):
    """Return (image_id, path) or raise. Safety is always enforced."""
    pipe = _load_pipe()
    import torch
    reap_old()  # prune before each gen so nothing piles up
    with GEN_LOCK:
        t0 = time.time()
        kwargs = dict(prompt=prompt, num_inference_steps=STEPS,
                      guidance_scale=0.0, height=SIZE, width=SIZE)
        with torch.no_grad():
            result = pipe(**kwargs)
        img = result.images[0]
        # Safety is ALWAYS on. If the pipeline flags NSFW, block unconditionally.
        nsfw = False
        flags = getattr(result, "nsfw_content_detected", None)
        if flags and any(flags):
            nsfw = True
        logger.info("Generated in %.1fs (nsfw=%s)", time.time() - t0, nsfw)
    if nsfw:
        raise ValueError("blocked_nsfw")
    image_id = f"img_{uuid.uuid4().hex[:10]}"
    path = os.path.join(OUT_DIR, image_id + ".png")
    img.save(path)
    return image_id, path


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="jay-imagegen-addon")


def publish_tts(text, source=None):
    payload = {"text": text}
    if source:
        payload["source"] = source
    client.publish("tts/request", json.dumps(payload), qos=1)


def publish_state(status, **extra):
    # Safety is a fixed, non-toggleable property of this addon: always true.
    payload = {"skill": SKILL, "timestamp": utc_now_iso(), "status": status,
               "safety": True, "size": SIZE}
    payload.update(extra)
    client.publish(f"skill/{SKILL}/state", json.dumps(payload), qos=1, retain=True)


def handle_request(req):
    prompt = (req.get("prompt") or req.get("text") or "").strip()
    # An addon-matched request carries data from the manifest's named groups.
    data = req.get("data") or {}
    if not prompt and data.get("subject"):
        prompt = data["subject"].strip()
    machine = (req.get("machine") or data.get("machine") or DEFAULT_PC).strip() or DEFAULT_PC
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
    logger.info("Subscribed to skill/%s/request", SKILL)
    publish_state("ready")


def on_message(client, userdata, msg):
    try:
        if msg.topic == f"skill/{SKILL}/request":
            req = json.loads(msg.payload.decode() or "{}")
            threading.Thread(target=handle_request, args=(req,), daemon=True).start()
    except Exception as e:
        logger.exception("on_message: %s", e)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if not self.path.startswith("/img/"):
            self.send_error(404); return
        name = os.path.basename(self.path[len("/img/"):])
        if not name.endswith(".png") or "/" in name or "\\" in name:
            self.send_error(400); return
        path = os.path.join(OUT_DIR, name)
        if not os.path.exists(path):
            self.send_error(404, "expired or unknown"); return
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            self.send_error(404); return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def serve_http():
    httpd = ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), Handler)
    logger.info("Image HTTP server on :%d (serving %s)", HTTP_PORT, OUT_DIR)
    httpd.serve_forever()


if __name__ == "__main__":
    logger.info("JAY Image Generation Addon starting "
                "(model=%s, steps=%d, size=%d, threads=%d, safety=ALWAYS ON)",
                MODEL_ID, STEPS, SIZE, IMAGE_THREADS)
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
