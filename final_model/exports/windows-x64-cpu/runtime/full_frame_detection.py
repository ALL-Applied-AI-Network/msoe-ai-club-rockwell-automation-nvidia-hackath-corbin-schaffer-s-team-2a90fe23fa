"""Full-frame slot assignment shared with the GB10 inference bundle."""
import math


def selected(slot, scope):
    return slot["tray"] == "G" or scope == "both"


def assign_to_slot(box_xywh, slots, scope="G"):
    x, y, width, height = box_xywh
    if width <= 0 or height <= 0:
        return None
    candidates = [slot for slot in slots if selected(slot, scope)]
    if not candidates:
        return None
    center_x, center_y = x + width / 2, y + height / 2
    anchor = lambda slot: slot.get("inspection_anchor_native", slot["center_native"])
    best = min(candidates, key=lambda slot: (anchor(slot)[0] - center_x) ** 2 + (anchor(slot)[1] - center_y) ** 2)
    best_x, best_y = anchor(best)
    distances = [
        math.hypot(best_x - anchor(slot)[0], best_y - anchor(slot)[1])
        for slot in candidates
        if slot["slot_id"] != best["slot_id"]
    ]
    radius = min(distances) / 2 if distances else 0
    if math.hypot(center_x - best_x, center_y - best_y) > radius:
        return None
    return best["slot_id"]
