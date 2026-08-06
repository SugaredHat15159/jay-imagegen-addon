# JAY Image Generation Addon

**Author: SugaredHat15159** · A [JAY](https://github.com/SugaredHat15159/Jay) addon.

Generate images from a spoken prompt, entirely on your own hardware. This addon
runs Stability AI's **sd-turbo** on CPU (no GPU required), serves the result over
a local HTTP port, and opens it on the PC of your choice. Generated images
auto-delete after a short TTL so storage never grows.

**The safety filter is always on.** There is no NSFW toggle. Any image the
checker flags is blocked.

---

## What it does

Say something like:

- *"generate an image of a red barn at sunset"*
- *"make a picture of a robot playing chess"*
- *"draw a dog"*
- *"generate an image of a mountain lake on desktop"*

JAY generates the image on the server (~15s at 512px, ~30s at 768px on a modern
CPU), then tells your PC to open the resulting PNG in the browser. The file is
deleted automatically after 10 minutes.

---

## Requirements

- A running [JAY](https://github.com/SugaredHat15159/Jay) server with the addon
  system installed (`addon.sh`, addon loader).
- The JAY PC Agent running on whichever machine should display the image (it
  handles the `open_url` command).
- ~3 GB RAM free during generation and ~2.5 GB disk for the model cache (first
  run downloads sd-turbo from Hugging Face).

---

## Install

From your JAY server root:

```bash
./addon.sh add https://github.com/SugaredHat15159/jay-imagegen-addon
```

That clones this repo into `addons/imagegen/`, builds the container, starts it,
and restarts JAY's NLP router so the image commands work immediately.

Then copy and edit the environment file:

```bash
cd addons/imagegen
cp .env.example .env
# Edit .env — set IMAGE_HOST_ADDR to your JAY tailnet IP, DEFAULT_PC to your machine name
```

Restart the addon to pick up your `.env`:

```bash
cd /srv/jay
./addon.sh rebuild
```

The **first generation** is slow — it downloads the sd-turbo model (~2.5 GB) and
loads it into memory. Subsequent generations are fast and the model stays
resident.

---

## Configuration

All configuration is via `.env` (copy from `.env.example`):

| Variable            | Default                   | Meaning                                            |
|---------------------|---------------------------|----------------------------------------------------|
| `MQTT_HOST`         | `127.0.0.1`               | JAY broker host.                                   |
| `MQTT_PORT`         | `1883`                    | JAY broker port (local anon).                      |
| `IMAGE_MODEL`       | `stabilityai/sd-turbo`    | Diffusers model id.                                |
| `IMAGE_THREADS`     | `4`                       | CPU threads for generation.                        |
| `IMAGE_STEPS`       | `4`                       | Inference steps (sd-turbo is a few-step model).    |
| `IMAGE_SIZE`        | `512`                     | Output size in px (512 or 768).                    |
| `IMAGE_HTTP_PORT`   | `8137`                    | Port the generated PNG is served on.               |
| `IMAGE_HOST_ADDR`   | `100.119.255.57`          | Address PCs reach to fetch the image (tailnet IP). |
| `IMAGE_TTL_SECONDS` | `600`                     | Delete PNGs after this many seconds.               |
| `DEFAULT_PC`        | `laptop`                  | Which PC opens the image if none is named.         |

---

## How it plugs into JAY

This addon follows the JAY addon contract. It communicates purely over MQTT:

- **Subscribes:** `skill/imagegen/request`
- **Publishes:** `tts/request` (spoken status), `skill/imagegen/state` (retained,
  for dashboards), `pc/command/<machine>` (tells the PC agent to open the image).

The voice intents are declared as regex patterns in `addon.manifest.json`, so
JAY's NLP router recognizes image phrases without any changes to core code.

---

## Voice grammar

The addon matches two shapes:

1. **Explicit:** *generate / make / create / render* + *image / picture / photo /
   drawing / painting* + *of X* — e.g. "generate an image of a car".
2. **Bare art verb:** *draw / paint / sketch* + *X* — e.g. "draw a dog".

Either form accepts an optional *on desktop* / *on laptop* suffix to pick the
display target.

Common non-image phrases ("draw the curtains", "draw me a bath") are excluded so
they pass through to the rest of JAY.

---

## License

JAY Personal-Use License — see [LICENSE](LICENSE). Free for personal and
non-commercial use with attribution to **SugaredHat15159**. Commercial use requires
written permission.
