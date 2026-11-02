"""Geometric cap inspection policy; yaw is not a seating defect."""
import math

POLICY = 'relative_cap_tilt_v1'
def label_cap(cap_present, vial_axis, cap_axis, threshold_degrees=1.0):
    if not cap_present:
        return {'label': 'NO_PASS', 'cap_tilt_degrees': None, 'label_reason': 'missing_cap'}
    def unit(axis):
        norm = math.sqrt(sum(float(v)**2 for v in axis))
        if norm == 0: raise ValueError('Cannot classify a zero-length geometry axis')
        return [float(v)/norm for v in axis]
    a,b=unit(vial_axis),unit(cap_axis)
    cosine=max(-1.0,min(1.0,sum(x*y for x,y in zip(a,b))))
    tilt=math.degrees(math.acos(cosine))
    good=tilt < threshold_degrees and not math.isclose(tilt, threshold_degrees, abs_tol=1e-9)
    return {'label':'PASS' if good else 'NO_PASS', 'cap_tilt_degrees':tilt,
            'label_reason':'cap_tilt_below_threshold' if good else 'cap_tilt_at_or_above_threshold'}
