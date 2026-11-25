# Vietnamese 9:16 TikTok Dance Dataset — Exact Requirements Specification

Every clip admitted to **VietH3** training (`configs/model_vieth3_tiktok_9x16_dense.yaml`) MUST satisfy **all 5 gates** below. Rejecting 70–85% of raw scrapes is normal and protects the model from background drift, glowing artifacts, and multi-limb hallucinations.

---

## 1. Video Format, Aspect Ratio & Frame Grid

| Parameter | Stage 1 (`480p` Pilot) | Stage 2 (`720p` Production) | Automated Gate |
| :--- | :--- | :--- | :--- |
| **Orientation** | Strict `9:16` portrait (`height > width`) | Strict `9:16` portrait (`height > width`) | `require_portrait=True` (`0.50 <= W/H <= 0.62`) |
| **Minimum Resolution** | `>= 480 × 832` (prefer `540 × 960`) | `>= 704 × 1248` (`720 × 1280`) | `min_width`, `min_height` |
| **Frame Rate & Codec** | `24 fps`, H.264 `yuv420p`, `CRF <= 18` | `24 fps`, H.264 `yuv420p`, `CRF <= 18` | `scene_split.py` |
| **Clip Frame Counts** | Wan2.1 `4n + 1`: **`49, 65, 81, 121`** (`min >= 21`) | Wan2.1 `4n + 1`: **`49, 65, 81, 121`** (`min >= 21`) | `--grid wan21` |
| **Raw Source Duration** | `4s – 60s` per source video | `4s – 60s` per source video | `yt-dlp --match-filter` |

---

## 2. Static Camera & Still Background (Anti-Wobble Gate)

- **Requirement:** Camera must be **tripod-mounted / phone-stand locked** with a rigid, stationary background (bedroom wall, living room, dance studio, café, quiet street).
- **Threshold:** **`bg_motion_score <= 0.25 px/frame`** (measured over outer margins + 30th-percentile optical flow).
- **HARD REJECT:**
  - Handheld selfie walking or friend-held shaky filming
  - Auto-tracking / dynamic zoom / beat-sync camera shake edits
  - Slow background pan, gimbal orbit, or car/train window parallax

---

## 3. Dance Motion & Human Physics (Anti-Hallucination Gate)

- **Requirement:** **One solo Vietnamese dancer**, waist-up or full-body, facing camera, with head, arms, and hands cleanly inside frame for smooth physics learning.
- **Thresholds:**
  - **`fg_motion_score >= 1.0 px/frame`** (subject is actively dancing, not posing still)
  - **`motion_score <= 18.0 px/frame`** (reject chaotic blur/whip-cuts)
- **HARD REJECT:**
  - Multi-person / duet / split-screen collages or crowds walking directly behind the dancer
  - Jump-cut outfit swaps, speed-ramps (`0.5x → 3x`), or reverse boomerang loops
  - AI-generated / heavy beauty-warp body distortion clips

---

## 4. Anti-Glow, Lighting Stability & Clean Pixels

- **Requirement:** Steady natural daylight or even soft indoor lighting, crisp skin/clothing edges, zero overlays.
- **Thresholds:**
  - **`flicker_score <= 12.0`** (blocks frame-to-frame luminance jumps and bright `>245` sparkle flashes)
  - **`overlay_score <= 0.18`** (blocks bottom Vietnamese lyrics/hardsubs and corner stickers/logos)
  - **`laplacian_var >= 40.0`** (blocks out-of-focus, heavy motion-blur, or low-bitrate re-uploads)
- **HARD REJECT:**
  - CapCut glitter/sparkle/bokeh/halo filters (primary cause of "weird glowy moving things")
  - Flashing RGB LEDs, neon club strobes, or pulsing ring-light exposure
  - Floating Vietnamese captions, emoji stickers, or large TikTok end-card watermarks

---

## 5. Deduplication & Visual Contact-Sheet QA

1. **Deduplication (`ingest_tiktok.py`):** Exact `SHA-256` + 3-frame perceptual `dHash` removes identical copies and re-encoded TikTok reposts before splitting.
2. **Mandatory Visual Audit (`audit_sheets/*.jpg`):** Every clip passing automated filters generates a **6-frame contact sheet**. Visually inspect sheets before training to catch subtle issues heuristics miss (tiny background passersby, CapCut glow halos, cropped hands, or text over chest).

---

## 6. Anti-AI / Real-Human Creator Verification Gate (CRITICAL)

Generic tags like `#nhaytiktok` are heavily flooded with **AI-generated dance spam**. Training on synthetic AI clips causes severe model collapse (plastic skin, warped hands, glowing halos). **Never admit any video until it passes this Real-Human check:**

### Instant-Reject AI-Slop Red Flags
- TikTok `"AI-generated"` / `"Creator labeled as AI"` badge or `#aigenerated` / `#ai` tags.
- Glossy plastic skin, rubbery/melting fingers during fast waves, or background lines bending near hips/elbows.
- Repetitive 5–10 second loops with uncanny perfectly centered framing and zero room imperfections.
- Spammy numbered/random handles posting dozens of different faces/bodies in identical poses.

### Real-Human Green Flags
- Consistent real person & real imperfect Vietnamese room/studio across multiple posts on the creator's profile (plus normal talking, daily-life, or mirror vlogs).
- Natural skin texture, realistic hair flyaways, crisp fabric folds, and steady `15s–45s` single-shot takes.

---

## 7. Real-Human Discovery Strategy (No AI-Spam Tags)

1. **Primary — Real-Creator Snowball (Highest Yield):**
   - Start from verified real Vietnamese dancers discovered on the user's For You / Following feed.
   - Open a verified creator's profile, grab `2–3` clean tripod-locked dance videos, then expand to **Suggested / Similar Accounts** or **Following** to snowball 20–40 authentic human creators.
2. **Secondary — Real-World Dance Practice Queries (Avoid `#nhaytiktok`):**
   - Search real rehearsal/studio keywords that AI spam almost never uses:
     - `"tập nhảy"` · `"phòng tập nhảy"` · `"lớp nhảy"` · `"tập nhảy cover"` · `"dance practice vietnam"`
   - Pair with creator checks above before collecting any URL.
