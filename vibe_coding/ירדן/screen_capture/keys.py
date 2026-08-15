from pynput import keyboard


def key_to_str(key):
    """Serialize a pynput key object to a string suitable for JSON storage."""
    return str(key)


def str_to_key(key_str):
    """Deserialize a string produced by key_to_str back into a pynput key object."""
    if key_str.startswith("Key."):
        name = key_str.split(".", 1)[1]
        return getattr(keyboard.Key, name)
    if key_str.startswith("'") and key_str.endswith("'"):
        return keyboard.KeyCode.from_char(key_str[1:-1])
    if key_str.startswith("<") and key_str.endswith(">"):
        vk = int(key_str[1:-1])
        return keyboard.KeyCode.from_vk(vk)
    raise ValueError(f"Cannot parse key: {key_str!r}")


def key_display_name(key_str):
    """Human-readable label for a stored key string, for menus/logs."""
    if key_str is None:
        return "not set"
    if key_str.startswith("Key."):
        return key_str.split(".", 1)[1]
    if key_str.startswith("'") and key_str.endswith("'"):
        return key_str[1:-1].upper()
    return key_str
