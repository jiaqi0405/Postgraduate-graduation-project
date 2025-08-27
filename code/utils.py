import math

def find_key(map_data, value, default_value=None):
    new_map = {value:key for key, value in map_data.items()}
    return new_map.get(value, default_value)

def parse_size(text, default_value=0):
    try:
        n = float(text)
    except Exception:
        n = default_value
    return n

def parse_a_b(text, default_value=0):
    a_text, b_text = text.split('x', 1)
    try:
        a = float(a_text)
    except Exception:
        a = default_value
    try:
        b = float(b_text)
    except Exception:
        b = default_value
    return (a, b)

def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
