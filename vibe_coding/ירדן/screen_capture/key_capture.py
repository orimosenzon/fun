from pynput import keyboard


def capture_next_key(on_captured):
    """Start a listener that fires on_captured(key) for the very next key press, then stops itself."""

    def on_press(key):
        listener.stop()
        on_captured(key)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener
