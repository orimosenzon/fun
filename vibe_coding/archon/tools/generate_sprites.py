#!/usr/bin/env python3
"""Generate 16 piece sprites for Archon via Nano Banana (Gemini 2.5 Flash Image)."""

import os
import sys
from pathlib import Path

# Load .env from project root
ROOT = Path(__file__).parent.parent
env_file = ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

SPRITE_DIR = ROOT / "public" / "assets" / "sprites"
SPRITE_DIR.mkdir(parents=True, exist_ok=True)

BASE_STYLE = (
    "HD modern pixel art sprite, inspired by Archon: The Light and the Dark (1983) "
    "but rendered at high resolution. Clean transparent background. "
    "Single character centered, full-body, 3/4 front view facing camera. "
    "Bold silhouette readable at small size. Crisp pixel edges, "
    "limited but vibrant color palette, strong rim lighting. "
    "Square composition, character fills 80% of frame. "
    "No text, no logos, no border."
)

PIECES = {
    # ── LIGHT SIDE ──
    "knight":   "A heroic human knight in shining silver plate armor, golden trim, holding a longsword pointed up and a kite shield. Heroic stance.",
    "archer":   "A nimble elven archer in green and tan leather, hooded, drawing back a longbow with golden-feathered arrow nocked.",
    "valkyrie": "A winged female warrior in radiant golden armor, large feathered white wings spread, holding a spear, flowing red cape.",
    "golem":    "A massive stone earth golem, glowing yellow runes carved into pale granite body, broad shoulders, blocky heroic stance, friendly side.",
    "unicorn":  "A majestic pure white unicorn rearing on hind legs, long silver horn glowing, flowing mane, sparkling magical aura around hooves.",
    "djinni":   "A floating muscular blue-skinned genie, arms crossed, lower body trailing into wisps of blue smoke, golden bracelets, turban with jewel.",
    "phoenix":  "A magnificent fiery phoenix bird with wings outstretched, orange/red/gold flame feathers, embers trailing, brilliant glow.",
    "wizard":   "An old wise wizard with long flowing white beard, deep blue robes embroidered with gold stars, holding a glowing staff topped with a crystal, pointed hat.",

    # ── DARK SIDE ──
    "goblin":       "A snarling green-skinned goblin warrior, ragged leather armor, wielding a jagged rusty sword, pointed ears, yellow eyes, hunched menacing pose.",
    "manticore":    "A monstrous manticore with crimson lion body, leathery bat wings folded, segmented black scorpion tail raised high with dripping venom, snarling human-like face.",
    "harpy":        "A dark winged harpy creature, woman with feathered black wings and bird talons for feet, wild dark hair, mid-shriek with mouth open, sharp claws.",
    "troll":        "A huge hulking cave troll, mottled brown-grey leathery skin, single tusk, swinging a massive bone club, brutish stance, small malevolent eyes.",
    "basilisk":     "A large reptilian basilisk, long sinuous deep-green and yellow scaled body, glowing baleful yellow petrifying eyes, jaw open showing fangs.",
    "dragon":       "A classic fierce red dragon, scaled crimson body, leathery wings half-spread, breathing a curl of fire, sharp horns, perched on rocks.",
    "shapeshifter": "An ambiguous shadowy humanoid figure, deep purple-black smoky form, glowing white eyes, partially translucent edges, body shifting between forms.",
    "sorceress":    "A dark sorceress in long flowing black robes with deep purple accents, raven-black hair, glowing purple magic swirling around her raised hand, malevolent beauty.",
}

def generate(name: str, prompt: str, out: Path) -> bool:
    full = f"{prompt} {BASE_STYLE}"
    print(f"  → {name} ...", flush=True)
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=full,
            config={"response_modalities": ["IMAGE", "TEXT"]},
        )
        for part in resp.candidates[0].content.parts:
            if getattr(part, "inline_data", None) is not None:
                out.write_bytes(part.inline_data.data)
                print(f"    ✓ saved {out.name} ({len(part.inline_data.data)//1024} KB)")
                return True
        print(f"    ✗ no image in response")
        return False
    except Exception as e:
        print(f"    ✗ error: {e}")
        return False

def main():
    total = len(PIECES)
    done = 0
    skipped = 0
    failed = []
    print(f"Generating {total} sprites → {SPRITE_DIR}")
    print("=" * 60)
    for name, prompt in PIECES.items():
        out = SPRITE_DIR / f"{name}.png"
        if out.exists() and out.stat().st_size > 1024:
            print(f"  ⏭ {name} — already exists ({out.stat().st_size//1024} KB)")
            skipped += 1
            continue
        if generate(name, prompt, out):
            done += 1
        else:
            failed.append(name)
    print("=" * 60)
    print(f"done: {done}, skipped: {skipped}, failed: {len(failed)}")
    if failed:
        print(f"failed pieces: {', '.join(failed)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
