## נקודת הכניסה: סביבה (שמיים, שמש, ערפל), העולם, השחקן והממשק.
##
## פרמטרים משורת הפקודה (אחרי --), לבדיקות אוטומטיות:
##   --shot=PATH        לצלם מסך אחרי שהעולם נטען ולצאת
##   --at=LAT,LON       להתחיל במיקום הזה
##   --alt=METERS       גובה מעל הקרקע (0 = מצב הליכה)
##   --heading=DEG      כיוון מבט (0 = צפון)
##   --pitch=DEG        הטיית מבט
extends Node3D

# מרכז המושבה, ליד המועצה
const START_LATLON := Vector2(32.4735, 34.9765)

var world: World
var player: Player
var hud: Hud
var sun: DirectionalLight3D
var args := {}
var shot_frames := -1


func _ready() -> void:
	for a in OS.get_cmdline_user_args():
		var kv := a.trim_prefix("--").split("=", true, 1)
		args[kv[0]] = kv[1] if kv.size() > 1 else ""

	_make_environment()
	world = World.new()
	add_child(world)
	world.load_all()

	player = Player.new()
	player.world = world
	add_child(player)

	var ll := START_LATLON
	if args.has("at"):
		var p: PackedStringArray = args.at.split(",")
		ll = Vector2(float(p[0]), float(p[1]))
	var alt := float(args.get("alt", "180"))
	teleport(ll.x, ll.y, alt, float(args.get("heading", "0")), float(args.get("pitch", "-25" if alt > 0 else "-3")))

	hud = Hud.new()
	hud.main = self
	add_child(hud)

	if args.has("shot"):
		shot_frames = 90


func teleport(lat: float, lon: float, alt: float, heading := -1.0, pitch := -10.0) -> void:
	var p := world.local_from_latlon(lat, lon)
	player.position = p + Vector3.UP * maxf(alt, 0.0)
	player.set_flying(alt > 0.0)
	if heading >= 0.0:
		player.look_heading(heading, pitch)


func _make_environment() -> void:
	var sky_mat := ProceduralSkyMaterial.new()
	sky_mat.sky_top_color = Color(0.32, 0.52, 0.82)
	sky_mat.sky_horizon_color = Color(0.72, 0.80, 0.88)
	sky_mat.ground_horizon_color = Color(0.72, 0.80, 0.88)
	sky_mat.ground_bottom_color = Color(0.35, 0.33, 0.30)
	sky_mat.sun_angle_max = 20.0
	var sky := Sky.new()
	sky.sky_material = sky_mat
	var env := Environment.new()
	env.background_mode = Environment.BG_SKY
	env.sky = sky
	env.ambient_light_source = Environment.AMBIENT_SOURCE_SKY
	env.ambient_light_energy = 0.6
	env.tonemap_mode = Environment.TONE_MAPPER_AGX
	env.tonemap_exposure = 0.85
	env.ssao_enabled = true
	env.ssao_radius = 1.5
	env.ssao_intensity = 1.6
	env.fog_enabled = true
	env.fog_mode = Environment.FOG_MODE_DEPTH
	env.fog_light_color = Color(0.76, 0.82, 0.88)
	env.fog_depth_begin = 1200.0
	env.fog_depth_end = 9000.0
	env.fog_density = 0.6
	env.fog_sky_affect = 0.0
	var we := WorldEnvironment.new()
	we.environment = env
	add_child(we)

	# שמש של אחר צהריים בסתיו, מדרום-מערב
	sun = DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-38, 215, 0)
	sun.light_color = Color(1.0, 0.95, 0.86)
	sun.light_energy = 1.25
	sun.shadow_enabled = true
	sun.directional_shadow_max_distance = 450.0
	sun.directional_shadow_mode = DirectionalLight3D.SHADOW_PARALLEL_4_SPLITS
	add_child(sun)


func _process(_delta: float) -> void:
	if shot_frames > 0:
		shot_frames -= 1
		if shot_frames == 0:
			var img := get_viewport().get_texture().get_image()
			img.save_png(args.shot)
			print("shot saved: ", args.shot, "  fps=", Engine.get_frames_per_second())
			get_tree().quit()
