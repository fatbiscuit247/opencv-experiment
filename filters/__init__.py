from .beauty import apply as beauty
from .hsv import warm, cool, vivid, fade

def original(frame):
    return frame.copy()