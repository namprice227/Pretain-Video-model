---
name: tiktok-dance-dataset-crawler
description: >-
  Browser-guided TikTok feed curation and dataset filtering for VietH3: opens TikTok
  in the built-in browser so the user can sign into their account and solve any CAPTCHAs,
  scrolls through the feed or Vietnamese dance searches to visually pre-screen clips
  against strict 9:16 tripod-background/anti-glow/physics rules, collects matching URLs,
  and downloads/filters them into training clips. Use when the user asks to browse,
  scroll, crawl, or collect suitable TikTok dance videos for training.
---

# Browser-Guided TikTok Dance Curation (9:16 VietH3)

Curate **9:16 Vietnamese solo dance clips** directly inside a **real signed-in browser session** (not a headless bot). The user signs into their own TikTok account and handles any CAPTCHAs directly in the browser pane, while the `browser` subagent scrolls, visually pre-screens videos against [REQUIREMENTS.md](./references/REQUIREMENTS.md), collects matching URLs, and runs [crawl_and_filter_tiktok.py](./scripts/crawl_and_filter_tiktok.py).

---

## Mandatory Guardrails
- **Human-in-the-loop Login & CAPTCHA:** **NEVER** attempt to solve, click through, wait out, or bypass any TikTok CAPTCHA, puzzle slider, or login wall. Whenever login or a CAPTCHA appears, **stop scrolling immediately**, ask the user to complete it in the browser pane, and resume the **exact same** `browser` subagent via `send_message` once they confirm.
- **Keep the same browser session:** Always pass `Model: "flash"` when invoking the `browser` subagent, leave it idle between steps, and reuse its `conversationId` via `send_message` so the signed-in TikTok session stays open.

---

## Step-by-Step Workflow

### 1. Open TikTok & Let the User Sign In
1. Launch `invoke_subagent(TypeName="browser", Model="flash", Role="TikTok Browser Curator", ...)` instructing it to navigate to `https://www.tiktok.com` and immediately pause if not signed in (or if a login/CAPTCHA modal appears).
2. Ask the user to **sign into their TikTok account** (and solve any initial CAPTCHA) directly in the browser pane, and let you know when they are logged in.
3. Once the user confirms, resume the **exact same** `browser` subagent via `send_message` (never start a new browser subagent, or login state is lost).

### 2. Curate 100% Real Human Dancers (No Generic AI-Spam Hashtags)
Instruct the `browser` subagent to collect canonical URLs (`https://www.tiktok.com/@<creator>/video/<id>`) using **Mode A** or **Mode B** from [REQUIREMENTS.md](./references/REQUIREMENTS.md):
- **Mode A — User Feed + Real-Creator Snowball (Recommended):** Scroll the user's For You / Following feed; whenever a **real human Vietnamese dancer** passes, visit their profile to verify they are a real person (consistent face/room across posts), take `2–3` clean tripod clips, and expand via **Suggested Accounts**.
- **Mode B — Real Dance Practice Search:** Search real-world rehearsal keywords (`"tập nhảy"`, `"phòng tập nhảy"`, `"lớp nhảy"`, `"tập nhảy cover"`, `"dance practice vietnam"`). **Do NOT search generic spam tags like `#nhaytiktok`**, which are flooded with AI-generated dance videos.

**Pre-screen EVERY clip visually across 2+ timestamps before adding its URL:**
- **100% Real Human (Anti-AI Gate):** Reject any `"AI-generated"` label, plastic/glossy AI skin, melting fingers, bending wall edges, or spam handles with shifting faces across posts.
- **Strict 9:16 Solo Dancer:** One Vietnamese dancer, waist-up or full-body, head/arms/hands inside frame (skip duets, split-screens, or background crowds).
- **Still Tripod Background:** Camera is fixed on a stand/tripod (skip handheld walking, pans, or beat-zoom edits).
- **Clean Dance Physics:** Continuous single-shot dancing (skip jump-cut outfit swaps, speed-ramps, or reverse loops).
- **Zero Glow / Flicker / Heavy Text:** Even natural or steady indoor light (skip CapCut glitter/sparkles/halos, flashing RGB/ring lights, and floating Vietnamese lyrics/stickers covering the scene).
- **CAPTCHA Pause Rule:** If a CAPTCHA or security slider appears at any point while scrolling, the subagent must stop at once and report it so the user can solve it in the browser pane before resuming via `send_message`.

Stop scrolling once the target batch size is collected (e.g., `25–50` candidate URLs for a pilot batch, or `200+` for production).

### 3. Save URLs & Run Automated Download + Strict Quality Filter
1. Save the deduplicated canonical URLs (one per line, UTF-8) to `data/urls_tiktok.txt`.
2. Run [crawl_and_filter_tiktok.py](./scripts/crawl_and_filter_tiktok.py) to download the MP4s (`4s–60s`), deduplicate (SHA-256 + 3-frame perceptual dHash), split onto the `24 fps` Wan2.1 `4n + 1` grid (`49, 65, 81, 121` frames), enforce `bg_motion <= 0.25`, `fg_motion >= 1.0`, `flicker <= 12.0`, `overlay <= 0.18`, `sharpness >= 40.0`, and render 6-frame contact sheets:

```bash
# On Windows via WSL Ubuntu (~/.venvs/vieth3), or `python` on Linux H100:
~/.venvs/vieth3/bin/python .agents/skills/tiktok-dance-dataset-crawler/scripts/crawl_and_filter_tiktok.py \
  --urls-file data/urls_tiktok.txt \
  --output-dir data/tiktok_crawl_batch \
  --max-per-source 1
```

*(For Stage 1 `480p` pilot sourcing where `540×960` clips are sparse, you may pass `--min-width 480 --min-height 832`; keep `540×960` or `704×1248` for `720p` Stage 2.)*

### 4. Inspect Audit Summary & 6-Frame Contact Sheets
1. Read `data/tiktok_crawl_batch/audit_summary.json` to check pass rate and rejection reasons (`shaky_background`, `flicker_or_glow`, `watermark_or_hardsub`).
2. Inspect the generated `data/tiktok_crawl_batch/audit_sheets/*.jpg` via `read_file` against [REQUIREMENTS.md](./references/REQUIREMENTS.md), removing any subtle false-passes from `manifest_filtered.jsonl`.

### 5. Hand Off to VietH3 Recaptioning & Wan2.1 Caching
Feed approved `manifest_filtered.jsonl` into Vietnamese recaptioning and Wan2.1-VAE latent caching:
```bash
python -m vieth3.data.recaption_vi \
  --manifest data/tiktok_crawl_batch/manifest_filtered.jsonl \
  --output data/tiktok_work/manifest_captioned.jsonl

python -m vieth3.data.cache_latents \
  --manifest data/tiktok_work/manifest_captioned.jsonl \
  --output-dir data/cached_tiktok_480p \
  --model-config configs/model_vieth3_tiktok_9x16_dense.yaml \
  --vae-type wan21 \
  --vae-id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --text-encoder qwen25vl \
  --bucket-stage 480p \
  --device cuda --dtype bfloat16
```
