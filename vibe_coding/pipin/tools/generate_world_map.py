#!/usr/bin/env python3
"""ייצור מפת עולם סכמטית-אמנותית של ארץ התיכונה עם Nano Banana"""

from google import genai
import os

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MAP_PROMPT = """
Create a detailed hand-drawn fantasy map in the style of Tolkien's Middle-earth maps,
with parchment/aged paper background, ink illustration style, and elegant calligraphy labels.

The map should show these locations connected by paths/roads, arranged geographically:

FAR WEST (coast):
- Grey Havens (harbor with ships, on western coast)

THE SHIRE (northwest, green rolling hills):
- Hobbiton (hobbit holes in hills)
- Bucklebury (river crossing with ferry)
- Old Forest (dark dense trees)
- Tom Bombadil's House (cottage with flowers)

ERIADOR (north-center):
- Bree (crossroads town with inn)
- Weathertop (ruined tower on hilltop)
- Trollshaws (rocky forest)
- Rivendell (elven valley with waterfalls)

MISTY MOUNTAINS (center, tall mountain range running north-south):
- Moria West Gate (doors in cliff)
- Mines of Moria (underground, shown as dotted)
- Bridge of Khazad-dum (eastern exit)
- Dimrill Dale (mountain valley with lake)
- Lothlorien (golden forest, south of mountains)

ROHAN (south-center, vast grasslands):
- Isengard (dark tower in circle wall)
- Fangorn Forest (ancient forest)
- Helm's Deep (fortress in mountain)
- Edoras (golden hall on hill)

GONDOR (south, white cities):
- Minas Tirith (white tiered city, prominent)
- Osgiliath (ruined city on river)
- Pelargir (river port)
- Dol Amroth (coastal castle, southwest)
- Ithilien (green wooded area)

MORDOR (southeast, dark volcanic wasteland):
- Dead Marshes (swamp, north of Mordor)
- Black Gate (massive iron gates)
- Minas Morgul (glowing green tower)
- Cirith Ungol (mountain pass with tower)
- Mordor Plateau (volcanic wasteland)
- Mount Doom (active volcano, central)
- Barad-dur (dark tower with eye, east)

Style requirements:
- Parchment/aged paper texture background
- Hand-drawn ink illustration style like classic Tolkien maps
- Mountains shown as small peaked illustrations
- Forests shown as clusters of tiny trees
- Rivers drawn as flowing lines
- Paths/roads shown as dotted or dashed lines connecting locations
- Each location labeled in elegant fantasy script (English)
- Compass rose in one corner
- Decorative border
- Title "Middle-earth" in ornate lettering at top
- The map should look like it was drawn by a cartographer in Middle-earth
- Wide landscape format
- No modern elements, pure fantasy cartography
"""

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "..", "images", "world_map.webp")

    print("מייצר מפת עולם של ארץ התיכונה...")

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=MAP_PROMPT,
            config={
                "response_modalities": ["IMAGE", "TEXT"],
            }
        )

        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data is not None:
                image_data = part.inline_data.data
                with open(output_path, 'wb') as f:
                    f.write(image_data)
                print(f"✓ מפת העולם נשמרה: {output_path}")
                return

        print("✗ לא התקבלה תמונה מהמודל")

    except Exception as e:
        print(f"✗ שגיאה: {e}")


if __name__ == "__main__":
    main()
