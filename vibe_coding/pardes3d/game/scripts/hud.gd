## ממשק: מיקום ומצב, מפה קטנה, רשימת מקומות לקפיצה, ופתיחת סטריט ויו בדפדפן.
class_name Hud
extends CanvasLayer

const MAP_SIZE := 220.0

var main: Node
var info: Label
var help: Label
var minimap: TextureRect
var marker: Control
var places_panel: PanelContainer
var places_list: ItemList
var place_rows: Array = []
var prompt: Label
var poi_panel: PanelContainer
var poi_title: Label
var poi_img: TextureRect
var poi_text: Label
var poi_credit: Label
var poi_current: Dictionary = {}
var poi_near: Dictionary = {}
var credits: PanelContainer
var http: HTTPRequest
var img_cache := {}


func _ready() -> void:
	var root := Control.new()
	root.set_anchors_preset(Control.PRESET_FULL_RECT)
	root.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(root)

	info = _label(root, 18)
	info.position = Vector2(16, 12)

	help = _label(root, 15)
	help.text = "\n".join([
		"לחיצה בעכבר: שליטה במבט   Esc: שחרור העכבר",
		"WASD / חיצים: תנועה   Shift: מהר   E/Q: למעלה/למטה (רחפן)",
		"F: רחפן / הליכה   גלגלת: מהירות הרחפן   רווח: קפיצה",
		"P: רשימת מקומות   G: פתח את המקום בגוגל סטריט ויו   M: מפה   H: הסתר עזרה",
		"I: פרטים על הנקודה הקרובה (סיכות צבעוניות: פרדספדיה, כחולות: תמונות)   C: מקורות וקרדיט",
	])
	help.set_anchors_preset(Control.PRESET_BOTTOM_LEFT)
	help.position = Vector2(16, get_viewport().get_visible_rect().size.y - 110)

	# מפה קטנה מטקסטורת הקרקע
	minimap = TextureRect.new()
	var img := Image.load_from_file(main.world.world_path("ground.png"))
	img.resize(512, 512)
	minimap.texture = ImageTexture.create_from_image(img)
	minimap.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	minimap.size = Vector2(MAP_SIZE, MAP_SIZE)
	minimap.anchor_left = 1.0
	minimap.anchor_right = 1.0
	minimap.position = Vector2(get_viewport().get_visible_rect().size.x - MAP_SIZE - 16, 16)
	minimap.modulate = Color(1, 1, 1, 0.92)
	root.add_child(minimap)
	marker = Control.new()
	marker.draw.connect(_draw_marker)
	minimap.add_child(marker)

	_make_places(root)
	_make_poi_panel(root)
	_make_credits(root)
	prompt = _label(root, 20)
	http = HTTPRequest.new()
	add_child(http)
	http.request_completed.connect(_on_image)
	get_viewport().size_changed.connect(_layout)
	_layout()


func _label(parent: Control, size: int) -> Label:
	var l := Label.new()
	l.add_theme_font_size_override("font_size", size)
	l.add_theme_color_override("font_color", Color.WHITE)
	l.add_theme_color_override("font_outline_color", Color(0, 0, 0, 0.85))
	l.add_theme_constant_override("outline_size", 5)
	parent.add_child(l)
	return l


func _layout() -> void:
	var s := get_viewport().get_visible_rect().size
	minimap.position = Vector2(s.x - MAP_SIZE - 16, 16)
	help.position = Vector2(16, s.y - 136)
	prompt.position = Vector2(s.x / 2 - 200, s.y - 90)
	poi_panel.position = Vector2(16, 60)
	poi_panel.size = Vector2(minf(560, s.x * 0.45), s.y - 220)
	credits.position = Vector2(s.x / 2 - 330, 80)
	credits.size = Vector2(660, 0)
	places_panel.position = Vector2(s.x - 360, MAP_SIZE + 32)
	places_panel.size = Vector2(344, s.y - MAP_SIZE - 48)


func _make_places(root: Control) -> void:
	places_panel = PanelContainer.new()
	places_panel.visible = false
	root.add_child(places_panel)
	var box := VBoxContainer.new()
	places_panel.add_child(box)
	var title := Label.new()
	title.text = "קפיצה למקום (לחיצה כפולה)"
	box.add_child(title)
	places_list = ItemList.new()
	places_list.size_flags_vertical = Control.SIZE_EXPAND_FILL
	box.add_child(places_list)
	place_rows = main.world.places.duplicate()
	for p in main.world.pois:
		if p.src == "pp":
			place_rows.append({"name": p.name, "kind": "פרדספדיה", "x": p.x, "y": p.y})
	place_rows.sort_custom(func(a, b): return a.name < b.name)
	for p in place_rows:
		places_list.add_item(p.name + ("  ·  " + p.kind if p.kind != "" else ""))
	places_list.item_activated.connect(_on_place)


func _make_poi_panel(root: Control) -> void:
	poi_panel = PanelContainer.new()
	poi_panel.visible = false
	root.add_child(poi_panel)
	var scroll := ScrollContainer.new()
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	poi_panel.add_child(scroll)
	var box := VBoxContainer.new()
	box.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.add_child(box)
	poi_title = Label.new()
	poi_title.add_theme_font_size_override("font_size", 22)
	poi_title.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	box.add_child(poi_title)
	poi_img = TextureRect.new()
	poi_img.expand_mode = TextureRect.EXPAND_FIT_WIDTH_PROPORTIONAL
	poi_img.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	poi_img.custom_minimum_size = Vector2(0, 280)
	box.add_child(poi_img)
	poi_text = Label.new()
	poi_text.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	box.add_child(poi_text)
	poi_credit = Label.new()
	poi_credit.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	poi_credit.add_theme_font_size_override("font_size", 13)
	poi_credit.add_theme_color_override("font_color", Color(0.75, 0.75, 0.75))
	box.add_child(poi_credit)
	var row := HBoxContainer.new()
	box.add_child(row)
	var open := Button.new()
	open.text = "פתח בדפדפן"
	open.pressed.connect(func(): if poi_current.get("url", "") != "": OS.shell_open(poi_current.url))
	row.add_child(open)
	var close := Button.new()
	close.text = "סגור (I)"
	close.pressed.connect(func(): poi_panel.visible = false)
	row.add_child(close)


func _make_credits(root: Control) -> void:
	credits = PanelContainer.new()
	credits.visible = false
	root.add_child(credits)
	var l := Label.new()
	l.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	var lines: Array = ["מקורות הנתונים (C לסגירה):", ""]
	lines.append_array(main.world.meta.get("attribution", []))
	l.text = "\n".join(lines)
	credits.add_child(l)


func _show_poi(p: Dictionary) -> void:
	poi_current = p
	poi_title.text = p.name + ("  ·  " + p.group if p.group != "" else "")
	var t: String = p.get("text", "")
	if p.get("address", "") != "":
		t = p.address + "\n\n" + t
	if p.get("approx", false):
		t += "\n\n(המיקום משוער, לפי הרחוב)"
	poi_text.text = t
	poi_credit.text = p.get("credit", "")
	poi_img.texture = null
	poi_img.visible = p.get("img", "") != ""
	if poi_img.visible:
		if img_cache.has(p.img):
			poi_img.texture = img_cache[p.img]
		else:
			http.cancel_request()
			http.request(p.img, ["User-Agent: pardes3d/0.1 (github.com/orimosenzon/fun)"])
	poi_panel.visible = true
	Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
	main.player.mouse_captured = false


func _on_image(result: int, code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
	if result != HTTPRequest.RESULT_SUCCESS or code != 200:
		poi_text.text += "\n\n(התמונה לא נטענה: %d)" % code
		return
	var img := Image.new()
	var url: String = poi_current.get("img", "")
	var err := img.load_jpg_from_buffer(body)
	if err != OK:
		err = img.load_png_from_buffer(body)
	if err != OK:
		err = img.load_webp_from_buffer(body)
	if err != OK:
		return
	var tex := ImageTexture.create_from_image(img)
	img_cache[url] = tex
	poi_img.texture = tex


func _on_place(i: int) -> void:
	var p: Dictionary = place_rows[i]
	var w: World = main.world
	var ll := w.latlon(Vector3(p.x, 0, -p.y))
	# נוחתים 40 מטר דרומית למקום ומסתכלים צפונה, ברחפן נמוך
	var back := Vector3(p.x, 0, -p.y + 60)
	var ll_back := w.latlon(back)
	main.teleport(ll_back.x, ll_back.y, 35.0, 0.0, -20.0)
	places_panel.visible = false


func _unhandled_key_input(event: InputEvent) -> void:
	if not (event is InputEventKey and event.pressed and not event.echo):
		return
	var pl: Player = main.player
	match event.keycode:
		KEY_F:
			pl.set_flying(not pl.flying)
		KEY_H:
			help.visible = not help.visible
		KEY_M:
			minimap.visible = not minimap.visible
		KEY_P:
			places_panel.visible = not places_panel.visible
			if places_panel.visible:
				Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
				pl.mouse_captured = false
				places_list.grab_focus()
		KEY_G:
			_open_street_view()
		KEY_I:
			if poi_panel.visible:
				poi_panel.visible = false
			elif not poi_near.is_empty():
				_show_poi(poi_near)
		KEY_C:
			credits.visible = not credits.visible


func _open_street_view() -> void:
	# סטריט ויו משמש כאן רק כרפרנס חזותי לעין: פותחים את גוגל מפות בדפדפן.
	# לא מורידים ממנו תמונות (ראו README, תנאי השימוש של גוגל).
	var pl: Player = main.player
	var ll: Vector2 = main.world.latlon(pl.global_position)
	var url := "https://www.google.com/maps/@?api=1&map_action=pano&viewpoint=%.6f,%.6f&heading=%d&pitch=%d&fov=90" % [
		ll.x, ll.y, int(pl.heading_deg()), int(rad_to_deg(pl.pitch))]
	OS.shell_open(url)


func _process(_delta: float) -> void:
	var pl: Player = main.player
	var w: World = main.world
	var ll := w.latlon(pl.global_position)
	var agl := pl.global_position.y - w.ground_height(pl.global_position.x, pl.global_position.z)
	info.text = "%s   %.5f, %.5f   כיוון %d°   גובה %d מ' מעל הקרקע (%d מ' מעל הים)   %d FPS" % [
		"רחפן" if pl.flying else "הליכה", ll.x, ll.y, int(pl.heading_deg()), int(agl),
		int(pl.global_position.y), Engine.get_frames_per_second()]
	marker.queue_redraw()
	poi_near = w.street.nearest_poi(pl.camera.global_position, 30.0) if w.street else {}
	prompt.text = "" if poi_near.is_empty() or poi_panel.visible else "I: " + str(poi_near.name)


func _draw_marker() -> void:
	var w: World = main.world
	var pl: Player = main.player
	var m: Dictionary = w.meta
	var u: float = (pl.global_position.x - m.west) / (m.east - m.west)
	var v: float = (m.north + pl.global_position.z) / (m.north - m.south)
	var c := Vector2(u, v) * MAP_SIZE
	var h := deg_to_rad(pl.heading_deg())
	var fwd := Vector2(sin(h), -cos(h))
	var side := Vector2(-fwd.y, fwd.x)
	marker.draw_colored_polygon(PackedVector2Array([c + fwd * 9, c - fwd * 5 + side * 5, c - fwd * 5 - side * 5]), Color(1, 0.2, 0.2))
	marker.draw_arc(c, 10, 0, TAU, 20, Color(1, 1, 1, 0.8), 1.5)
