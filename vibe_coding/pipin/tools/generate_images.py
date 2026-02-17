#!/usr/bin/env python3
"""סקריפט לייצור תמונות מקומות ודמויות עם Nano Banana (Gemini Image Generation)"""

from google import genai
import time
import os

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# ═══════════════════════════════════════════
#   תמונות מקומות
# ═══════════════════════════════════════════

BASE_STYLE = """
Fantasy illustration in the style of Alan Lee and John Howe,
watercolor with ink details, muted earth tones,
Tolkien's Middle-earth aesthetic, atmospheric lighting,
wide landscape composition, no text, no characters in frame.
"""

LOCATION_PROMPTS = {
    "grey_havens": "Elven harbor at sunset, tall white ships with silver sails, long stone quays reaching into a calm sea, towers of white stone, gulls flying overhead, melancholic golden light",
    "hobbiton": "A cozy hobbit village with round green doors set into grassy hillsides, flower gardens, a winding path, smoke rising from chimneys, warm afternoon light",
    "bucklebury": "Brandywine river crossing with a wooden ferry, rolling green hills of the Shire on both banks, willows along the riverbank, peaceful morning mist",
    "old_forest": "A dark ancient forest with gnarled twisted trees, thick roots crossing the path, dim green light filtering through dense canopy, oppressive and watchful atmosphere",
    "tom_bombadil": "A cheerful cottage in a forest clearing by a stream, yellow flowers everywhere, a bright blue door, warm firelight from within, contrast of cozy warmth against dark forest behind",
    "bree": "A medieval crossroads town at dusk, the Prancing Pony inn with warm light from windows, a low wooden palisade wall, muddy cobblestone streets, rain-slicked roofs",
    "weathertop": "A ruined watchtower on a lone hilltop, broken stone walls, ancient steps leading up, dark storm clouds gathering, windswept grasslands stretching in all directions",
    "trollshaws": "A rocky pine forest with mossy boulders and a stone bridge over a rushing stream, three large stone shapes (petrified trolls) partially hidden by vegetation, dappled woodland light",
    "rivendell": "An elven valley sanctuary, graceful buildings of carved stone among waterfalls, autumn trees with golden leaves, arched bridges over crystal streams, mountains rising behind, ethereal morning light",
    "moria_west": "Massive ancient dwarf doors carved into a sheer cliff face, reflecting pool before them, moon shining overhead illuminating faint silvery runes on the stone, dark and still",
    "moria": "A vast underground hall with towering stone pillars in rows disappearing into darkness, dwarf architecture carved with geometric patterns, a faint orange glow from deep below, dust motes in shaft of light from above",
    "moria_east": "The eastern exit of a mountain, a narrow stone bridge over a bottomless chasm, light streaming in from an opening ahead, ancient carved pillars, sense of escape and danger",
    "dimrill_dale": "A mountain valley with a still mirror-like lake (Mirrormere), snow-capped peaks reflected perfectly in the water, morning mist, ancient dwarf pillars along the shore",
    "lothlorien": "An enchanted golden forest, massive mallorn trees with silver bark and golden leaves, soft ethereal light, elevated wooden platforms (flets) in the canopy, a dreamlike timeless quality",
    "fangorn": "An ancient primeval forest, incredibly old gnarled trees covered in moss and lichen, thick tangled undergrowth, green filtered light, a sense that the trees are watching, ancient and deep",
    "isengard": "A dark tower (Orthanc) of black glossy stone rising from a circular walled compound, iron machinery and pits scarring the ground, smoke and industrial fires, ominous dark clouds above",
    "edoras": "A golden-roofed mead hall (Meduseld) atop a hill, wooden palisade walls, horse-lord banners fluttering in strong wind, vast grasslands of Rohan stretching to the horizon, dramatic sky",
    "helms_deep": "A massive fortress built into a mountain cliff, the Deeping Wall stretching across a valley, a horn-shaped mountain behind, torches along the battlements, dramatic stormy sky",
    "minas_tirith": "A magnificent white stone city of seven tiers built into a mountainside, the Tower of Ecthelion gleaming at the summit, the great gates below, Pelennor Fields spreading before it, epic scale",
    "osgiliath": "A ruined city straddling a wide river, broken stone bridges, crumbling domed buildings, overgrown with weeds, a somber war-torn atmosphere, grey overcast sky",
    "pelargir": "A river port city with stone quays, many-masted ships, warehouses along the waterfront, the Anduin river flowing wide and grey, southern architecture with domed buildings",
    "dol_amroth": "A white coastal castle on cliffs above the sea, tall towers with blue and silver banners of a swan-ship, waves crashing below, seabirds, bright maritime light",
    "ithilien": "A green wooded land with hidden pools and waterfalls, lush vegetation, wildflowers among ruins of ancient Gondorian gardens, a secret paradise, warm dappled sunlight",
    "dead_marshes": "A vast desolate swamp at twilight, stagnant pools reflecting a pale sickly light, ghostly faces barely visible beneath the water surface, wisps of pale flame, thick fog, deeply unsettling",
    "black_gate": "Enormous iron gates set between two mountain spurs (Morannon), blackened wasteland before them, orc fortifications on either side, ash and dust in the air, utterly forbidding, dark red sky",
    "minas_morgul": "A corrupted tower city glowing with a sickly pale green phosphorescent light, twisted architecture, a bridge over a poisoned stream, dead flowers, the road winding up to a dark pass behind",
    "cirith_ungol": "A narrow mountain pass with a sinister stone tower, steep stairs carved into rock, cobwebs in dark crevices, a sense of being watched, pale moonlight, threatening shadows",
    "mordor_plateau": "A barren volcanic plateau of cracked black rock, rivers of lava in the distance, ash falling like snow, Mount Doom smoldering on the horizon, Barad-dur's dark tower far away, red-tinged hellscape",
    "mount_doom": "A massive active volcano belching fire and smoke, rivers of molten lava flowing down its slopes, a narrow path winding up to a dark entrance, red glowing sky, apocalyptic atmosphere",
    "barad_dur": "An impossibly tall dark fortress of iron and black stone, a great flaming eye at its summit, lightning striking around it, surrounded by ash plains and orc armies, ultimate evil incarnate",
}

# ═══════════════════════════════════════════
#   תמונות דמויות (NPCs)
# ═══════════════════════════════════════════

NPC_BASE_STYLE = """
Fantasy character portrait in the style of Alan Lee and John Howe,
watercolor with ink details, muted earth tones,
Tolkien's Middle-earth aesthetic, soft atmospheric lighting,
bust/shoulder portrait composition, painterly quality, no text.
"""

NPC_PROMPTS = {
    "gandalf": "An old wizard with a long grey beard and bushy eyebrows, wearing a pointed grey hat and grey robes, wise kind eyes with a hint of mischief, leaning on a wooden staff",
    "gaffer_gamgee": "An old hobbit gardener, short and stout, sun-weathered face, white wispy hair, wearing simple brown clothes and a worn hat, holding garden shears",
    "tom_bombadil": "A merry fellow with a bright blue coat and yellow boots, round rosy face, brown curly hair under a hat with a blue feather, laughing joyfully",
    "butterbur": "A round jolly innkeeper with ruddy cheeks, balding head, wearing a stained apron, friendly but forgetful expression, wiping his hands on a cloth",
    "strider": "A rugged ranger with dark shoulder-length hair, weathered face with grey eyes, stubble beard, wearing a worn green-brown cloak with a silver star brooch, alert and noble bearing beneath rough exterior",
    "elrond": "A noble half-elf lord with long dark hair, ageless face that is both young and ancient, wearing rich robes of blue and silver, a circlet on his brow, wise compassionate eyes",
    "galadriel": "An ethereal elf queen with long golden-silver hair, radiantly beautiful ageless face, wearing flowing white robes, a subtle inner glow, eyes that see through time, both gentle and terrifying",
    "treebeard": "An ancient tree-creature (Ent), face formed from bark with deep eyes like dark pools, mossy beard of hanging lichen, gnarled branch-like features, impossibly old and wise",
    "saruman": "A tall imposing wizard with long white hair and beard, wearing white robes with subtle rainbow shimmer, piercing dark eyes, an air of corrupted grandeur, holding a black staff",
    "theoden": "An aging king of Rohan with long grey-blond hair, golden crown, fur-trimmed royal robes, weary but dignified face, once-strong warrior now burdened by age and sorrow",
    "eowyn": "A young woman of Rohan with long golden hair, fair stern face, wearing a white dress with golden horse-lord embroidery, determined eyes hiding sadness, shield-maiden bearing",
    "denethor": "A proud aging lord in rich but dark robes, sharp intelligent eyes gone slightly mad with grief, grey hair, a silver circlet, gaunt aristocratic face, holding a seeing stone",
    "faramir": "A young nobleman with gentle but brave features, short brown hair, wearing leather and mail of Gondor, kind eyes, ranger's green cloak, resembles his father but softer",
    "gollum": "A wretched emaciated creature with huge pale eyes, thin wispy hair, grey skin, crouching, long thin fingers, pitiful yet menacing, talking to himself",
}


def generate_image(prompt, output_path):
    """ייצור תמונה בודדת עם Gemini"""
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=prompt,
            config={
                "response_modalities": ["IMAGE", "TEXT"],
            }
        )

        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data is not None:
                image_data = part.inline_data.data
                with open(output_path, 'wb') as f:
                    f.write(image_data)
                return True
        return False
    except Exception as e:
        print(f"  ✗ שגיאה: {e}")
        return False


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    locations_dir = os.path.join(script_dir, "..", "images", "locations")
    npcs_dir = os.path.join(script_dir, "..", "images", "npcs")
    os.makedirs(locations_dir, exist_ok=True)
    os.makedirs(npcs_dir, exist_ok=True)

    print("=" * 50)
    print("  מייצר תמונות מקומות")
    print("=" * 50)

    for location_id, specific_prompt in LOCATION_PROMPTS.items():
        output_path = os.path.join(locations_dir, f"{location_id}.webp")
        if os.path.exists(output_path):
            print(f"  ⏭ כבר קיים: {location_id}.webp")
            continue

        print(f"  מייצר: {location_id}...")
        full_prompt = f"{BASE_STYLE}\n\nScene: {specific_prompt}"

        if generate_image(full_prompt, output_path):
            print(f"  ✓ נשמר: {location_id}.webp")
        else:
            print(f"  ✗ נכשל: {location_id}")

        time.sleep(2)

    print("\n" + "=" * 50)
    print("  מייצר פורטרטים של דמויות")
    print("=" * 50)

    for npc_id, specific_prompt in NPC_PROMPTS.items():
        output_path = os.path.join(npcs_dir, f"{npc_id}.webp")
        if os.path.exists(output_path):
            print(f"  ⏭ כבר קיים: {npc_id}.webp")
            continue

        print(f"  מייצר: {npc_id}...")
        full_prompt = f"{NPC_BASE_STYLE}\n\nCharacter: {specific_prompt}"

        if generate_image(full_prompt, output_path):
            print(f"  ✓ נשמר: {npc_id}.webp")
        else:
            print(f"  ✗ נכשל: {npc_id}")

        time.sleep(2)

    print("\n✨ סיום! כל התמונות נוצרו.")


if __name__ == "__main__":
    main()
