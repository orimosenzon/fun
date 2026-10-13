## טוען את העולם שבנה pipeline/build_world.py מתוך game/world/.
## התיקייה מסומנת ב-.gdignore כדי שהעורך לא ינסה לייבא 50MB של נכסים;
## הכל נטען בזמן ריצה עם GLTFDocument.
class_name World
extends Node3D

const TREE_CHUNK := 400.0
const TREE_RANGE := 900.0

var meta: Dictionary
var heights: PackedFloat32Array
var nx: int
var ny: int
var grid: float
var west: float
var south: float
var places: Array = []
var pois: Array = []
var street: Street


func world_path(name: String) -> String:
	return ProjectSettings.globalize_path("res://world/" + name)


func load_all() -> void:
	meta = JSON.parse_string(FileAccess.get_file_as_string(world_path("meta.json")))
	nx = int(meta.nx)
	ny = int(meta.ny)
	grid = float(meta.grid)
	west = float(meta.west)
	south = float(meta.south)
	heights = FileAccess.get_file_as_bytes(world_path("heights.bin")).to_float32_array()
	places = JSON.parse_string(FileAccess.get_file_as_string(world_path("places.json")))

	var ground_img := Image.load_from_file(world_path("ground.png"))
	ground_img.generate_mipmaps()
	var terrain_mat := ShaderMaterial.new()
	terrain_mat.shader = load("res://shaders/terrain.gdshader")
	terrain_mat.set_shader_parameter("ground", ImageTexture.create_from_image(ground_img))
	terrain_mat.set_shader_parameter("bounds", Vector4(meta.west, meta.south, meta.east, meta.north))
	_add_glb("terrain.glb", terrain_mat, false)

	var bmat := ShaderMaterial.new()
	bmat.shader = load("res://shaders/building.gdshader")
	var buildings := _add_glb("buildings.glb", bmat, true)
	# התנגשות עם בניינים, בשביל מצב הליכה
	for mi in buildings:
		var body := StaticBody3D.new()
		var shape := CollisionShape3D.new()
		shape.shape = mi.mesh.create_trimesh_shape()
		body.add_child(shape)
		mi.add_child(body)

	var rmat := ShaderMaterial.new()
	rmat.shader = load("res://shaders/road.gdshader")
	_add_glb("roads.glb", rmat, false)

	if FileAccess.file_exists(world_path("walls.glb")):
		var wmat := ShaderMaterial.new()
		wmat.shader = load("res://shaders/wall.gdshader")
		for mi in _add_glb("walls.glb", wmat, true):
			mi.visibility_range_end = 1500.0
			var body := StaticBody3D.new()
			var shape := CollisionShape3D.new()
			shape.shape = mi.mesh.create_trimesh_shape()
			body.add_child(shape)
			mi.add_child(body)

	_add_trees()

	street = Street.new()
	add_child(street)
	street.build(self)
	pois = street.pois


func _add_glb(name: String, mat: Material, shadows: bool) -> Array[MeshInstance3D]:
	var doc := GLTFDocument.new()
	var state := GLTFState.new()
	var err := doc.append_from_file(world_path(name), state)
	if err != OK:
		push_error("טעינת %s נכשלה: %s" % [name, err])
		return []
	var scene := doc.generate_scene(state)
	add_child(scene)
	var out: Array[MeshInstance3D] = []
	for n in scene.find_children("*", "MeshInstance3D", true, false):
		var mi := n as MeshInstance3D
		mi.material_override = mat
		mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_ON if shadows \
			else GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		out.append(mi)
	return out


## גובה הקרקע (y) בנקודה x,z של גודו, אינטרפולציה בילינארית כמו בצד הפייתון.
func ground_height(x: float, z: float) -> float:
	var fx := clampf((x - west) / grid, 0.0, nx - 1.001)
	var fy := clampf((-z - south) / grid, 0.0, ny - 1.001)
	var ix := int(fx)
	var iy := int(fy)
	var tx := fx - ix
	var ty := fy - iy
	var i := iy * nx + ix
	return lerpf(lerpf(heights[i], heights[i + 1], tx), lerpf(heights[i + nx], heights[i + nx + 1], tx), ty)


func latlon(pos: Vector3) -> Vector2:
	return Vector2(meta.lat0 + (-pos.z) / meta.m_per_deg_lat, meta.lon0 + pos.x / meta.m_per_deg_lon)


func local_from_latlon(lat: float, lon: float) -> Vector3:
	var x: float = (lon - meta.lon0) * meta.m_per_deg_lon
	var z: float = -(lat - meta.lat0) * meta.m_per_deg_lat
	return Vector3(x, ground_height(x, z), z)


# ---------------------------------------------------------------- עצים

func _tree_mesh() -> ArrayMesh:
	# עץ ביחידות: גובה 1, קוטר צמרת 1. כל מופע מותח אותו ל-(c, h, c).
	var st := SurfaceTool.new()
	var mesh := ArrayMesh.new()
	var trunk := CylinderMesh.new()
	trunk.top_radius = 0.035
	trunk.bottom_radius = 0.05
	trunk.height = 0.5
	trunk.radial_segments = 6
	trunk.rings = 1
	st.create_from(trunk, 0)
	var arr := st.commit_to_arrays()
	for i in arr[Mesh.ARRAY_VERTEX].size():
		arr[Mesh.ARRAY_VERTEX][i].y += 0.25
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arr)
	var crown := SphereMesh.new()
	crown.radius = 0.5
	crown.height = 0.62
	crown.radial_segments = 9
	crown.rings = 5
	st = SurfaceTool.new()
	st.create_from(crown, 0)
	arr = st.commit_to_arrays()
	var noise := FastNoiseLite.new()
	noise.frequency = 3.0
	for i in arr[Mesh.ARRAY_VERTEX].size():
		var v: Vector3 = arr[Mesh.ARRAY_VERTEX][i]
		v *= 1.0 + 0.18 * noise.get_noise_3dv(v * 2.0)
		v.y += 0.64
		arr[Mesh.ARRAY_VERTEX][i] = v
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arr)
	var bark := StandardMaterial3D.new()
	bark.albedo_color = Color(0.36, 0.28, 0.22)
	bark.roughness = 1.0
	var leaves := StandardMaterial3D.new()
	leaves.vertex_color_use_as_albedo = true
	leaves.albedo_color = Color.WHITE
	leaves.roughness = 0.95
	mesh.surface_set_material(0, bark)
	mesh.surface_set_material(1, leaves)
	return mesh


func _add_trees() -> void:
	var rows: Array = JSON.parse_string(FileAccess.get_file_as_string(world_path("trees.json")))
	var chunks := {}
	for r in rows:
		var key := Vector2i(floori(r[0] / TREE_CHUNK), floori(r[1] / TREE_CHUNK))
		if not chunks.has(key):
			chunks[key] = []
		chunks[key].append(r)
	var mesh := _tree_mesh()
	var rng := RandomNumberGenerator.new()
	rng.seed = 3
	for key in chunks:
		var list: Array = chunks[key]
		var mm := MultiMesh.new()
		mm.transform_format = MultiMesh.TRANSFORM_3D
		mm.use_colors = true
		mm.mesh = mesh
		mm.instance_count = list.size()
		var center := Vector3((key.x + 0.5) * TREE_CHUNK, 0, -(key.y + 0.5) * TREE_CHUNK)
		for i in list.size():
			var r: Array = list[i]
			var h: float = clampf(r[2], 2.0, 30.0)
			var c: float = clampf(r[3], 1.5, 25.0)
			var x: float = r[0]
			var z: float = -r[1]
			var basis := Basis(Vector3.UP, rng.randf() * TAU).scaled(Vector3(c, h, c))
			var pos := Vector3(x, ground_height(x, z) - 0.2, z) - center
			mm.set_instance_transform(i, Transform3D(basis, pos))
			var kind := int(r[4])
			var g := rng.randf_range(0.8, 1.15)
			var col := Color(0.12, 0.24, 0.07) * g
			if kind == 2:
				col = Color(0.06, 0.17, 0.05) * g  # הדר: ירוק כהה ומבריק
			elif rng.randf() < 0.2:
				col = Color(0.20, 0.27, 0.12) * g  # אקליפטוס/איקליפטוס בהיר
			col.a = 1.0
			mm.set_instance_color(i, col)
		var mmi := MultiMeshInstance3D.new()
		mmi.multimesh = mm
		mmi.position = center
		mmi.visibility_range_end = TREE_RANGE
		mmi.visibility_range_end_margin = 100.0
		mmi.visibility_range_fade_mode = GeometryInstance3D.VISIBILITY_RANGE_FADE_SELF
		add_child(mmi)
