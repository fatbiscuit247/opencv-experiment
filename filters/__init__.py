from .beauty import apply as beauty
from .hsv import warm, cool, vivid, fade
from filters.overlays import apply_cat_ears

def original(frame):
    return frame.copy()