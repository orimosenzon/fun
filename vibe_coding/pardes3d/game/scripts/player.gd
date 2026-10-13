## שני מצבי תנועה: רחפן (תעופה חופשית) והולך רגל (גובה עיניים, כבידה, התנגשות בבניינים).
class_name Player
extends CharacterBody3D

const EYE := 1.65
const WALK_SPEED := 4.0
const RUN_SPEED := 11.0
const GRAVITY := 18.0
const JUMP := 5.5

var world: World
var camera: Camera3D
var flying := true
var fly_speed := 60.0
var yaw := 0.0
var pitch := -0.35
var mouse_captured := false


func _ready() -> void:
	var shape := CollisionShape3D.new()
	var capsule := CapsuleShape3D.new()
	capsule.radius = 0.35
	capsule.height = 1.8
	shape.shape = capsule
	shape.position.y = 0.9
	add_child(shape)
	camera = Camera3D.new()
	camera.position.y = EYE
	camera.near = 0.1
	camera.far = 12000.0
	camera.fov = 70.0
	add_child(camera)
	set_flying(true)


func set_flying(on: bool) -> void:
	flying = on
	# ברחפן לא מתנגשים בכלום
	collision_mask = 0 if on else 1
	velocity = Vector3.ZERO
	if not on:
		_snap_to_ground()


func _snap_to_ground() -> void:
	position.y = maxf(position.y, world.ground_height(position.x, position.z))


func heading_deg() -> float:
	# 0 = צפון, 90 = מזרח
	return fposmod(rad_to_deg(-yaw), 360.0)


func look_heading(deg: float, pitch_deg: float = -10.0) -> void:
	yaw = -deg_to_rad(deg)
	pitch = deg_to_rad(pitch_deg)


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton and event.pressed:
		if event.button_index == MOUSE_BUTTON_LEFT and not mouse_captured:
			Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
			mouse_captured = true
		elif event.button_index == MOUSE_BUTTON_WHEEL_UP and flying:
			fly_speed = minf(fly_speed * 1.25, 1500.0)
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN and flying:
			fly_speed = maxf(fly_speed / 1.25, 3.0)
	elif event is InputEventMouseMotion and mouse_captured:
		yaw -= event.relative.x * 0.0025
		pitch = clampf(pitch - event.relative.y * 0.0025, -1.5, 1.5)
	elif event.is_action_pressed("ui_cancel") and mouse_captured:
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
		mouse_captured = false
		get_viewport().set_input_as_handled()


func _physics_process(delta: float) -> void:
	rotation = Vector3(0, yaw, 0)
	camera.rotation = Vector3(pitch, 0, 0)
	var input := Vector3.ZERO
	if Input.is_key_pressed(KEY_W) or Input.is_key_pressed(KEY_UP): input.z -= 1
	if Input.is_key_pressed(KEY_S) or Input.is_key_pressed(KEY_DOWN): input.z += 1
	if Input.is_key_pressed(KEY_A) or Input.is_key_pressed(KEY_LEFT): input.x -= 1
	if Input.is_key_pressed(KEY_D) or Input.is_key_pressed(KEY_RIGHT): input.x += 1
	var fast := Input.is_key_pressed(KEY_SHIFT)
	if flying:
		if Input.is_key_pressed(KEY_E) or Input.is_key_pressed(KEY_SPACE): input.y += 1
		if Input.is_key_pressed(KEY_Q) or Input.is_key_pressed(KEY_CTRL): input.y -= 1
		var dir := camera.global_basis * Vector3(input.x, 0, input.z) + Vector3.UP * input.y
		position += dir.normalized() * fly_speed * (4.0 if fast else 1.0) * delta if dir.length() > 0 else Vector3.ZERO
		position.y = maxf(position.y, world.ground_height(position.x, position.z) + 0.5)
		return
	var dir := (global_basis * Vector3(input.x, 0, input.z))
	dir.y = 0
	dir = dir.normalized()
	var speed := RUN_SPEED if fast else WALK_SPEED
	velocity.x = dir.x * speed
	velocity.z = dir.z * speed
	var ground := world.ground_height(position.x, position.z)
	var on_ground := position.y <= ground + 0.05
	if on_ground:
		velocity.y = JUMP if Input.is_key_pressed(KEY_SPACE) else 0.0
	else:
		velocity.y -= GRAVITY * delta
	move_and_slide()
	# הקרקע לא בפיזיקה: היא גריד גבהים, אז מצמידים אליה ידנית
	ground = world.ground_height(position.x, position.z)
	if position.y < ground:
		position.y = ground
		velocity.y = 0.0
