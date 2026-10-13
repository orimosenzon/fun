## ריהוט רחוב, כבלי חשמל ונקודות מידע (פרדספדיה ו-Wikimedia Commons).
class_name Street
extends Node3D

const CHUNK := 500.0
const RANGE := 700.0
const POI_LABEL_RANGE := 260.0

## furniture.json: [x, y, כיוון במעלות (0 = צפון, עם כיוון השעון), סוג]
enum { BUS_STOP, SIGNAL, LAMP, POLE, TOWER, BENCH }

var world: World
var pois: Array = []


func build(w: World) -> void:
	world = w
	_furniture()
	_power()
	_pois()


func _load(name: String) -> Variant:
	var p := world.world_path(name)
	if not FileAccess.file_exists(p):
		return []
	return JSON.parse_string(FileAccess.get_file_as_string(p))


# ---------------------------------------------------------------- מודלים

static func _mat(c: Color, rough := 0.8, metal := 0.0, emit := Color.BLACK) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = c
	m.roughness = rough
	m.metallic = metal
	if emit != Color.BLACK:
		m.emission_enabled = true
		m.emission = emit
		m.emission_energy_multiplier = 2.0
	return m


## מודל מורכב מקופסאות וגלילים. parts: [[mesh, transform, material], ...]
static func _compose(parts: Array) -> ArrayMesh:
	var out := ArrayMesh.new()
	var by_mat := {}
	for p in parts:
		var key: Material = p[2]
		if not by_mat.has(key):
			var st := SurfaceTool.new()
			st.begin(Mesh.PRIMITIVE_TRIANGLES)
			by_mat[key] = st
		(by_mat[key] as SurfaceTool).append_from(p[0], 0, p[1])
	for m in by_mat:
		var st: SurfaceTool = by_mat[m]
		st.commit(out)
		out.surface_set_material(out.get_surface_count() - 1, m)
	return out


static func _box(sx: float, sy: float, sz: float) -> BoxMesh:
	var b := BoxMesh.new()
	b.size = Vector3(sx, sy, sz)
	return b


static func _cyl(r_top: float, r_bot: float, h: float, seg := 8) -> CylinderMesh:
	var c := CylinderMesh.new()
	c.top_radius = r_top
	c.bottom_radius = r_bot
	c.height = h
	c.radial_segments = seg
	c.rings = 1
	return c


static func _at(x: float, y: float, z: float, rot_x := 0.0) -> Transform3D:
	return Transform3D(Basis(Vector3.RIGHT, rot_x), Vector3(x, y, z))


## כל המודלים פונים ל-Z שלילי (צפון) בכיוון 0.
func _models() -> Dictionary:
	var metal := _mat(Color(0.45, 0.47, 0.5), 0.5, 0.6)
	var dark := _mat(Color(0.12, 0.12, 0.13), 0.6, 0.3)
	var concrete := _mat(Color(0.62, 0.6, 0.57), 0.95)
	var glass := _mat(Color(0.3, 0.38, 0.42), 0.1, 0.2)
	var roof := _mat(Color(0.75, 0.75, 0.73), 0.6, 0.4)
	var wood := _mat(Color(0.45, 0.32, 0.2), 0.9)
	var lamp := _mat(Color(1, 0.95, 0.8), 0.3, 0.0, Color(1, 0.9, 0.7))
	var red := _mat(Color(0.3, 0.02, 0.02), 0.3, 0.0, Color(0.9, 0.1, 0.05))
	var green := _mat(Color(0.02, 0.2, 0.05), 0.3, 0.0, Color(0.1, 0.9, 0.3))
	var m := {}
	# תחנת אוטובוס: סככה באורך 4 מ' לאורך הכביש (ציר Z), גב, גג וספסל
	m[BUS_STOP] = _compose([
		[_box(0.08, 2.4, 0.08), _at(-0.7, 1.2, -1.9), metal], [_box(0.08, 2.4, 0.08), _at(-0.7, 1.2, 1.9), metal],
		[_box(0.04, 1.9, 3.8), _at(-0.72, 1.25, 0), glass],
		[_box(1.7, 0.1, 4.2), _at(-0.1, 2.45, 0), roof],
		[_box(0.45, 0.06, 2.6), _at(-0.4, 0.48, 0), wood], [_box(0.05, 0.45, 2.6), _at(-0.6, 0.7, 0), wood],
		[_box(0.06, 0.48, 0.06), _at(-0.4, 0.24, -1.2), metal], [_box(0.06, 0.48, 0.06), _at(-0.4, 0.24, 1.2), metal],
		[_box(0.05, 0.6, 0.9), _at(0.75, 2.0, 1.6), _mat(Color(0.95, 0.85, 0.1), 0.5)],
	])
	# רמזור: עמוד וראש עם שלושה אורות (אדום דולק), פונה לאורך הכביש
	m[SIGNAL] = _compose([
		[_cyl(0.07, 0.09, 3.6), _at(0, 1.8, 0), dark],
		[_box(0.32, 0.95, 0.25), _at(0, 3.4, 0), dark],
		[_cyl(0.1, 0.1, 0.05, 10), _at(0, 3.7, -0.14, PI / 2), red],
		[_cyl(0.1, 0.1, 0.05, 10), _at(0, 3.1, -0.14, PI / 2), green],
	])
	# פנס רחוב: עמוד 7 מ' וזרוע שפונה אל הכביש (Z שלילי)
	m[LAMP] = _compose([
		[_cyl(0.06, 0.1, 7.0), _at(0, 3.5, 0), metal],
		[_box(0.06, 0.06, 1.6), _at(0, 6.9, -0.75), metal],
		[_box(0.28, 0.1, 0.55), _at(0, 6.85, -1.55), metal],
		[_box(0.22, 0.03, 0.45), _at(0, 6.79, -1.55), lamp],
	])
	# עמוד חשמל מבטון עם זרוע
	m[POLE] = _compose([
		[_cyl(0.11, 0.17, 10.0), _at(0, 5.0, 0), concrete],
		[_box(1.8, 0.1, 0.1), _at(0, 9.3, 0), concrete],
	])
	# עמוד מתח גבוה: ארבע רגליים משופעות וזרועות
	var legs := []
	for sx in [-1, 1]:
		for sz in [-1, 1]:
			var bottom := Vector3(sx * 2.5, 0, sz * 2.5)
			var topp := Vector3(sx * 0.6, 24, sz * 0.6)
			var mid := (bottom + topp) / 2
			var dir := (topp - bottom).normalized()
			var basis := Basis(Vector3.UP.cross(dir).normalized(), Vector3.UP.angle_to(dir)) if dir != Vector3.UP else Basis()
			legs.append([_box(0.18, (topp - bottom).length(), 0.18), Transform3D(basis, mid), metal])
	for h in [14.0, 19.0, 23.5]:
		legs.append([_box(9.0 if h < 20 else 6.0, 0.25, 0.3), _at(0, h, 0), metal])
		for k in range(4):
			legs.append([_box(0.1, 0.1, 5.0 * (1.0 - h / 30.0)), Transform3D(Basis(Vector3.UP, k * PI / 2), Vector3(0, h - 3, 0)), metal])
	m[TOWER] = _compose(legs)
	m[BENCH] = _compose([
		[_box(1.8, 0.06, 0.45), _at(0, 0.45, 0), wood], [_box(1.8, 0.4, 0.05), _at(0, 0.7, 0.22), wood],
		[_box(0.06, 0.45, 0.4), _at(-0.8, 0.22, 0), dark], [_box(0.06, 0.45, 0.4), _at(0.8, 0.22, 0), dark],
	])
	return m


# ---------------------------------------------------------------- ריהוט

func _furniture() -> void:
	var rows: Array = _load("furniture.json")
	if rows.is_empty():
		return
	var models := _models()
	var chunks := {}
	for r in rows:
		var key := Vector3i(floori(r[0] / CHUNK), floori(r[1] / CHUNK), int(r[3]))
		if not chunks.has(key):
			chunks[key] = []
		chunks[key].append(r)
	for key in chunks:
		var list: Array = chunks[key]
		var kind: int = key.z
		var mm := MultiMesh.new()
		mm.transform_format = MultiMesh.TRANSFORM_3D
		mm.mesh = models[kind]
		mm.instance_count = list.size()
		var center := Vector3((key.x + 0.5) * CHUNK, 0, -(key.y + 0.5) * CHUNK)
		for i in list.size():
			var r: Array = list[i]
			var x: float = r[0]
			var z: float = -r[1]
			# כיוון: 0 = צפון (-Z), עם כיוון השעון; סיבוב חיובי סביב Y הוא נגד כיוון השעון
			var b := Basis(Vector3.UP, -deg_to_rad(r[2]))
			mm.set_instance_transform(i, Transform3D(b, Vector3(x, world.ground_height(x, z), z) - center))
		var mmi := MultiMeshInstance3D.new()
		mmi.multimesh = mm
		mmi.position = center
		mmi.visibility_range_end = 2500.0 if kind == TOWER else RANGE
		mmi.visibility_range_end_margin = 80.0
		mmi.visibility_range_fade_mode = GeometryInstance3D.VISIBILITY_RANGE_FADE_SELF
		add_child(mmi)


# ---------------------------------------------------------------- כבלי חשמל

func _power() -> void:
	var lines: Array = _load("power.json")
	if lines.is_empty():
		return
	var verts := PackedVector3Array()
	for line in lines:
		for i in range(line.size() - 1):
			var a: Array = line[i]
			var b: Array = line[i + 1]
			var pa := Vector3(a[0], 0, -a[1])
			var pb := Vector3(b[0], 0, -b[1])
			pa.y = world.ground_height(pa.x, pa.z) + float(a[2])
			pb.y = world.ground_height(pb.x, pb.z) + float(b[2])
			var span := pa.distance_to(pb)
			if span < 1.0 or span > 800.0:
				continue
			var side := (pb - pa).cross(Vector3.UP).normalized()
			var spread := 3.5 if float(a[2]) > 15.0 else 0.8
			var sag := span * 0.025
			for off in [-spread, 0.0, spread]:
				var prev: Vector3 = pa + side * off
				for s in range(1, 13):
					var t := s / 12.0
					var p: Vector3 = pa.lerp(pb, t) + side * off
					p.y -= sag * 4.0 * t * (1.0 - t)
					verts.append(prev)
					verts.append(p)
					prev = p
	var arr := []
	arr.resize(Mesh.ARRAY_MAX)
	arr[Mesh.ARRAY_VERTEX] = verts
	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_LINES, arr)
	var m := StandardMaterial3D.new()
	m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	m.albedo_color = Color(0.12, 0.12, 0.12)
	mesh.surface_set_material(0, m)
	var mi := MeshInstance3D.new()
	mi.mesh = mesh
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	add_child(mi)


# ---------------------------------------------------------------- נקודות מידע

const GROUP_COLORS := {
	"אוכל ושתייה": Color(0.95, 0.55, 0.15), "תרבות וקהילה": Color(0.7, 0.35, 0.9),
	"טבע וגנים": Color(0.25, 0.75, 0.3), "היסטוריה והנצחה": Color(0.75, 0.6, 0.35),
	"תמונה": Color(0.2, 0.55, 0.95),
}


func _pois() -> void:
	pois = _load("pois.json")
	var pin_mesh := _compose([
		[_cyl(0.05, 0.05, 4.0), _at(0, 2.0, 0), _mat(Color(0.9, 0.9, 0.9), 0.5)],
	])
	var head := SphereMesh.new()
	head.radius = 0.55
	head.height = 1.1
	head.radial_segments = 12
	head.rings = 6
	var head_mat := StandardMaterial3D.new()
	head_mat.vertex_color_use_as_albedo = true
	head_mat.emission_enabled = true
	head_mat.emission = Color(0.15, 0.15, 0.15)
	head.material = head_mat
	var mm_pin := MultiMesh.new()
	mm_pin.transform_format = MultiMesh.TRANSFORM_3D
	mm_pin.mesh = pin_mesh
	mm_pin.instance_count = pois.size()
	var mm_head := MultiMesh.new()
	mm_head.transform_format = MultiMesh.TRANSFORM_3D
	mm_head.use_colors = true
	mm_head.mesh = head
	mm_head.instance_count = pois.size()
	for i in pois.size():
		var p: Dictionary = pois[i]
		var x: float = p.x
		var z: float = -p.y
		var g := world.ground_height(x, z)
		var is_photo: bool = p.src == "commons"
		var h := 2.5 if is_photo else 4.0
		var pos := Vector3(x, g, z)
		p["pos"] = pos + Vector3.UP * (h + 0.6)
		var s := 0.6 if is_photo else 1.0
		mm_pin.set_instance_transform(i, Transform3D(Basis().scaled(Vector3(1, h / 4.0, 1)), pos))
		mm_head.set_instance_transform(i, Transform3D(Basis().scaled(Vector3.ONE * s), pos + Vector3.UP * (h + 0.4)))
		mm_head.set_instance_color(i, GROUP_COLORS.get(p.group, Color(0.95, 0.3, 0.3)))
		if not is_photo:
			var l := Label3D.new()
			l.text = p.name
			l.billboard = BaseMaterial3D.BILLBOARD_ENABLED
			l.font_size = 64
			l.pixel_size = 0.012
			l.outline_size = 16
			l.modulate = Color.WHITE
			l.no_depth_test = false
			l.position = pos + Vector3.UP * (h + 1.6)
			l.visibility_range_end = POI_LABEL_RANGE
			l.visibility_range_end_margin = 30.0
			l.visibility_range_fade_mode = GeometryInstance3D.VISIBILITY_RANGE_FADE_SELF
			add_child(l)
	for mm in [mm_pin, mm_head]:
		var mmi := MultiMeshInstance3D.new()
		mmi.multimesh = mm
		mmi.visibility_range_end = 900.0
		mmi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		# ה-AABB של המולטימש מחושב ממיקומי המופעים; בלי זה הוא היה נחתך
		add_child(mmi)


## נקודת המידע הקרובה ביותר לנקודה, בטווח max_dist, או מילון ריק.
## מקום מפרדספדיה מקבל עדיפות על תמונה באותו מרחק.
func nearest_poi(at: Vector3, max_dist: float) -> Dictionary:
	var best := {}
	var bd := max_dist * max_dist
	for p in pois:
		var d: float = (p.pos as Vector3).distance_squared_to(at) * (0.4 if p.src == "pp" else 1.0)
		if d < bd:
			bd = d
			best = p
	return best
