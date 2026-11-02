"""G-slot temporal fusion for full-frame video detectors. No training or ground-truth access."""
from collections import defaultdict
import math

class VideoInspector:
    def __init__(self, slots, confirmation_frames=3, absence_frames=3, threshold=0.1):
        if confirmation_frames<1 or absence_frames<1:raise ValueError('Frame thresholds must be positive')
        if not math.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('Confidence threshold must be in [0,1]')
        self.slots=[s for s in slots if s['tray']=='G']
        if {s['slot_id'] for s in self.slots}!={f'G{i:02d}' for i in range(1,31)}:
            raise ValueError('Calibration must contain exactly G01-G30')
        self.confirm=confirmation_frames;self.absence=absence_frames;self.threshold=threshold
        self.reset()
    def reset(self):
        self.state={s['slot_id']:{'seen':0,'missed':0,'occupied':False,'defect_streak':{},'defects':set()} for s in self.slots}
    def update(self, detections, frame_index, timestamp):
        present=set();defects=defaultdict(set)
        for d in detections:
            if d.get('score',0)<self.threshold:continue
            sid=d.get('slot_id');category=d.get('category_id')
            if sid not in self.state or category not in (1,2,3):continue
            present.add(sid)
            if category!=1:defects[sid].add(category)
        for sid,state in self.state.items():
            if sid in present:
                state['seen']+=1;state['missed']=0
                if state['seen']>=self.confirm:state['occupied']=True
            else:
                state['seen']=0;state['missed']+=1
                if state['missed']>=self.absence:state['occupied']=False
            for category in (2,3):
                streak=state['defect_streak'].get(category,0)
                state['defect_streak'][category]=streak+1 if category in defects[sid] else 0
                if state['defect_streak'][category]>=self.confirm:state['defects'].add(category)
                # No stale defect carried into another tray/order.
                if not state['occupied'] or (sid in present and category not in defects[sid]):state['defects'].discard(category)
        occupied=[s['slot_id'] for s in self.slots if self.state[s['slot_id']]['occupied']]
        unknown=[sid for sid,s in self.state.items() if s['seen']<self.confirm and s['missed']<self.absence]
        empty=[sid for sid,s in self.state.items() if not s['occupied'] and s['missed']>=self.absence]
        last=max([s['position'] for s in self.slots if s['slot_id'] in occupied],default=0)
        gaps=[s['slot_id'] for s in self.slots if s['position']<last and s['slot_id'] in empty]
        return {'frame_index':frame_index,'timestamp_seconds':timestamp,
                'order_status':'complete' if len(occupied)==30 and not unknown else 'incomplete',
                'occupied_count':len(occupied),'empty_slots':empty,'unknown_slots':unknown,'gap_slots':gaps,
                'defects':[{'slot_id':sid,'category_id':cat,'defect':'tilted_cap' if cat==2 else 'missing_cap'}
                           for sid,s in self.state.items() for cat in sorted(s['defects'])]}
