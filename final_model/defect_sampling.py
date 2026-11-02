"""Sample a persistent cap condition once per vial, never independently per video frame."""
import math,random

def sample_defects(seed, probability, count=30):
    if not math.isfinite(probability) or not 0<=probability<=1:
        raise ValueError('Defect probability must be in [0,1]')
    rng=random.Random(seed)
    plan={}
    for index in range(count):
        if rng.random()<probability:
            kind=rng.choice(['tilt','missing'])
            plan[index]={'kind':kind,'tilt_degrees':rng.uniform(2,25) if kind=='tilt' else None}
    return plan
