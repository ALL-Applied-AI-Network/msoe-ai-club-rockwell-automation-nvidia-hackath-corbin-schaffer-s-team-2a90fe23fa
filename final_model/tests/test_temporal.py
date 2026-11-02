import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from temporal import VideoInspector
SLOTS=[{'slot_id':f'G{i:02d}','tray':'G','position':i} for i in range(1,31)]
def detections(count):return [{'slot_id':f'G{i:02d}','score':1,'category_id':1} for i in range(1,count+1)]
class TemporalTests(unittest.TestCase):
    def test_completion_and_missing_evidence(self):
        tracker=VideoInspector(SLOTS)
        self.assertEqual(tracker.update(detections(30),0,0)['order_status'],'incomplete')
        tracker.update(detections(30),1,.2)
        self.assertEqual(tracker.update(detections(30),2,.4)['order_status'],'complete')
        # One absent observation becomes unknown, never a confident complete.
        self.assertEqual(tracker.update(detections(29),3,.6)['order_status'],'incomplete')
        tracker.update(detections(29),4,.8)
        result=tracker.update(detections(29),5,1)
        self.assertEqual(result['empty_slots'],['G30']);self.assertEqual(result['gap_slots'],[])
    def test_gap_and_defect_independent_of_fullness(self):
        tracker=VideoInspector(SLOTS)
        items=[d for d in detections(30) if d['slot_id']!='G12']
        items.append({'slot_id':'G08','score':1,'category_id':2})
        for i in range(3):result=tracker.update(items,i,i*.2)
        self.assertEqual(result['gap_slots'],['G12']);self.assertEqual(result['defects'][0]['slot_id'],'G08')
        tracker.reset();self.assertEqual(tracker.update([],0,0)['occupied_count'],0)
    def test_low_confidence_and_s_tray_ignored(self):
        tracker=VideoInspector(SLOTS)
        for i in range(3):result=tracker.update([{'slot_id':'G01','score':.01,'category_id':1},{'slot_id':'S01','score':1,'category_id':1}],i,i*.2)
        self.assertEqual(result['occupied_count'],0)
    def test_trailing_empty_slots_are_not_missing(self):
        tracker=VideoInspector(SLOTS)
        for i in range(5):result=tracker.update(detections(12),i,i*.2)
        self.assertEqual(result['missing_slots'],[])
        self.assertEqual(result['order_status'],'incomplete')

    def test_gap_is_a_postprocessing_defect(self):
        tracker=VideoInspector(SLOTS)
        items=[d for d in detections(20) if d['slot_id']!='G07']
        for i in range(3):result=tracker.update(items,i,i*.2)
        self.assertEqual(result['missing_slots'],['G07'])
        self.assertEqual(result['missing_vial_defects'][0]['reason'],'fill_order_gap')
        self.assertIsNone(result['missing_vial_defects'][0]['category_id'])
        self.assertIn(result['missing_vial_defects'][0],result['defects'])

    def test_brief_absence_is_unknown_then_sustained_absence_is_missing(self):
        tracker=VideoInspector(SLOTS)
        for i in range(3):tracker.update(detections(30),i,i*.2)
        for i in range(3,5):
            result=tracker.update(detections(29),i,i*.2)
            self.assertIn('G30',result['unknown_slots'])
            self.assertEqual(result['missing_slots'],[])
        result=tracker.update(detections(29),5,1.)
        self.assertEqual(result['missing_slots'],['G30'])
        self.assertEqual(result['missing_vial_defects'][0]['reason'],'previously_occupied')
        for i in range(6,9):result=tracker.update(detections(30),i,i*.2)
        self.assertEqual(result['missing_slots'],[])
        self.assertEqual(result['order_status'],'complete')
        tracker.reset()
        for i in range(3):result=tracker.update([],i,i*.2)
        self.assertEqual(result['missing_slots'],[])

    def test_presence_and_cap_cutoffs_are_independent(self):
        tracker=VideoInspector(SLOTS,threshold=.9,presence_threshold=.25)
        items=[dict(d,score=.5) for d in detections(30)]
        items.append({'slot_id':'G01','category_id':2,'score':.8})
        for i in range(3):result=tracker.update(items,i,i*.2)
        self.assertEqual(result['order_status'],'complete')
        self.assertEqual(result['cap_defects'],[])
        items[-1]['score']=.95
        for i in range(3,6):result=tracker.update(items,i,i*.2)
        self.assertEqual(result['cap_defects'][0]['slot_id'],'G01')

    def test_invalid_presence_threshold_rejected(self):
        for value in (-.1,1.1,float('nan')):
            with self.assertRaises(ValueError):VideoInspector(SLOTS,presence_threshold=value)

if __name__=='__main__':unittest.main()
