"""G-slot temporal fusion for full-frame video detectors."""
from collections import defaultdict
import math


class VideoInspector:
    def __init__(self, slots, confirmation_frames=3, absence_frames=3, threshold=0.1):
        if confirmation_frames < 1 or absence_frames < 1:
            raise ValueError("Frame thresholds must be positive")
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Confidence threshold must be in [0, 1]")
        self.slots = [slot for slot in slots if slot["tray"] == "G"]
        if {slot["slot_id"] for slot in self.slots} != {f"G{index:02d}" for index in range(1, 31)}:
            raise ValueError("Calibration must contain exactly G01-G30")
        self.confirm = confirmation_frames
        self.absence = absence_frames
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.state = {
            slot["slot_id"]: {
                "seen": 0,
                "missed": 0,
                "occupied": False,
                "defect_streak": {},
                "defects": set(),
            }
            for slot in self.slots
        }

    def update(self, detections, frame_index, timestamp):
        present = set()
        defects = defaultdict(set)
        for detection in detections:
            if detection.get("score", 0) < self.threshold:
                continue
            slot_id = detection.get("slot_id")
            category = detection.get("category_id")
            if slot_id not in self.state or category not in (1, 2, 3):
                continue
            present.add(slot_id)
            if category != 1:
                defects[slot_id].add(category)
        for slot_id, state in self.state.items():
            if slot_id in present:
                state["seen"] += 1
                state["missed"] = 0
                if state["seen"] >= self.confirm:
                    state["occupied"] = True
            else:
                state["seen"] = 0
                state["missed"] += 1
                if state["missed"] >= self.absence:
                    state["occupied"] = False
            for category in (2, 3):
                streak = state["defect_streak"].get(category, 0)
                state["defect_streak"][category] = streak + 1 if category in defects[slot_id] else 0
                if state["defect_streak"][category] >= self.confirm:
                    state["defects"].add(category)
                if not state["occupied"] or (
                    slot_id in present and category not in defects[slot_id]
                ):
                    state["defects"].discard(category)
        occupied = [slot["slot_id"] for slot in self.slots if self.state[slot["slot_id"]]["occupied"]]
        unknown = [
            slot_id
            for slot_id, state in self.state.items()
            if state["seen"] < self.confirm and state["missed"] < self.absence
        ]
        empty = [
            slot_id
            for slot_id, state in self.state.items()
            if not state["occupied"] and state["missed"] >= self.absence
        ]
        last = max(
            [slot["position"] for slot in self.slots if slot["slot_id"] in occupied],
            default=0,
        )
        gaps = [
            slot["slot_id"]
            for slot in self.slots
            if slot["position"] < last and slot["slot_id"] in empty
        ]
        return {
            "frame_index": frame_index,
            "timestamp_seconds": timestamp,
            "order_status": "complete" if len(occupied) == 30 and not unknown else "incomplete",
            "occupied_count": len(occupied),
            "empty_slots": empty,
            "unknown_slots": unknown,
            "gap_slots": gaps,
            "defects": [
                {
                    "slot_id": slot_id,
                    "category_id": category,
                    "defect": "tilted_cap" if category == 2 else "missing_cap",
                }
                for slot_id, state in self.state.items()
                for category in sorted(state["defects"])
            ],
        }
