from pynput import keyboard

# This will store the state of each key
keys = {}

def on_press(key):
    try:
        keys[key.char] = True
        print(f'Key {key.char} pressed')
    except AttributeError:
        keys[str(key)] = True
        print(f'Special key {key} pressed')

def on_release(key):
    try:
        keys[key.char] = False
        print(f'Key {key.char} released')
    except AttributeError:
        keys[str(key)] = False
        print(f'Special key {key} released')

def make_keyboard_listener(init_keys):
    for key in init_keys:
        keys[key] = False
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return keys