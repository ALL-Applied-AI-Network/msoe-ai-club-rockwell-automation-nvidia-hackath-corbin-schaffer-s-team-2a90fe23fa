"""Full-frame COCO boxes and fixed-camera slot assignment; G tray is the default scope."""
import math

CATEGORIES = [{'id':1,'name':'vial'}, {'id':2,'name':'tilted_cap'}, {'id':3,'name':'missing_cap'}]

def good_tray_order_status(occupied_slots):
    """Order fullness is independent of cap quality and fill-sequence gaps."""
    expected = [f"G{i:02d}" for i in range(1, 31)]
    occupied = set(occupied_slots) & set(expected)
    empty = [name for name in expected if name not in occupied]
    return {"order_status": "incomplete" if empty else "complete",
            "order_complete": not empty, "occupied_count": len(occupied),
            "required_count": 30, "empty_slots": empty}


def selected(slot, scope):
    return slot['tray']=='G' or scope=='both'

def clipped_box(bounds, width, height):
    if not bounds or len(bounds)!=4 or not all(math.isfinite(x) for x in bounds):
        raise ValueError('Invalid inspection geometry')
    x0,y0,x1,y1=bounds
    if x1<=x0 or y1<=y0: raise ValueError('Empty/reversed inspection geometry')
    x0,y0,x1,y1=max(0,x0),max(0,y0),min(width,x1),min(height,y1)
    if x1<=x0 or y1<=y0:return None
    return [x0,y0,x1-x0,y1-y0]

def frame_annotations(metadata, scope='G'):
    if scope not in ('G','both'):raise ValueError('Scope must be G or both')
    width,height=metadata['native_size'];annotations=[];slots=[]
    for slot in metadata['slots']:
        if not selected(slot,scope):continue
        item=dict(slot)
        item['center_full_frame']=slot['center_native']
        item['bbox_full_frame']=clipped_box(slot['inspection_bounds_native'],width,height) if slot['present'] else None
        slots.append(item)
        if not slot['present']:continue
        box=item['bbox_full_frame']
        if box is None:continue
        if slot.get('label_source')!='relative_cap_tilt_v1':
            raise ValueError('Detection labels require geometric cap policy; replay old sidecars first')
        present=slot.get('cap_present')
        tilt=slot.get('cap_tilt_degrees')
        if present is None or (present and tilt is None):raise ValueError('Missing cap state')
        if present and not math.isfinite(tilt):raise ValueError('Non-finite cap tilt')
        tilted = present and (tilt >= 1.0 or math.isclose(tilt,1.0,abs_tol=1e-9))
        defect = 3 if not present else (2 if tilted else None)
        # Occupancy learns to locate the top inspection region of every present vial.
        # Defect boxes encompass the mouth and cap; they are not pixel masks.
        for category in ([1,defect] if defect else [1]):
            annotations.append({'category_id':category,'bbox':box,'area':box[2]*box[3],
                'iscrowd':0,'slot_id':slot['slot_id'],'tray':slot['tray'],
                'source_id':slot['source_id'],'cap_tilt_degrees':tilt,
                'label_source':slot['label_source']})
    return {'coordinate_system':'native_full_frame_xy_pixels','image_size':[width,height],
            'scope':scope,'categories':CATEGORIES,'annotations':annotations,'slots':slots,
            'good_tray_order':good_tray_order_status(s['slot_id'] for s in slots if s['present']),
            'gap_errors':[s['slot_id'] for s in slots if s['gap_error']]}

def assign_to_slot(box_xywh, slots, scope='G'):
    """Assign predicted box centre to the nearest calibrated slot *within half pitch*.

    slots must come from fixed-camera calibration/sidecar in the same pixel frame.
    Restricting distance prevents remote S-tray predictions being forced onto G slots.
    """
    x,y,w,h=box_xywh
    if w<=0 or h<=0:return None
    candidates=[s for s in slots if selected(s,scope)]
    cx,cy=x+w/2,y+h/2
    if not candidates:return None
    def anchor(s):return s.get('inspection_anchor_native',s['center_native'])
    best=min(candidates,key=lambda s:(anchor(s)[0]-cx)**2+(anchor(s)[1]-cy)**2)
    bx,by=anchor(best)
    distances=[math.hypot(bx-anchor(s)[0],by-anchor(s)[1]) for s in candidates if s['slot_id']!=best['slot_id']]
    radius=min(distances)/2 if distances else 0
    if math.hypot(cx-bx,cy-by)>radius:return None
    return best['slot_id']
