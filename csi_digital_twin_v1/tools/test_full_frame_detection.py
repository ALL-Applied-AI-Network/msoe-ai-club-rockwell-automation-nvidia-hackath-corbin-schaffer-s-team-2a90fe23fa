import unittest
from full_frame_detection import clipped_box,frame_annotations,assign_to_slot,good_tray_order_status

class DetectionTests(unittest.TestCase):
    def slot(self, ident='G01', cap=True,tilt=0,present=True):
        return {'slot_id':ident,'tray':ident[0],'position':int(ident[1:]),'center_native':[20,20],
                'inspection_bounds_native':[10,10,30,30] if present else None,'present':present,
                'cap_present':cap,'cap_tilt_degrees':tilt,'label_source':'relative_cap_tilt_v1',
                'source_id':'source','gap_error':not present}
    def test_labels_and_scope(self):
        slots=[self.slot(),self.slot('G02',tilt=.99),self.slot('G03',tilt=1),
               self.slot('G04',cap=False,tilt=None),self.slot('G05',present=False),self.slot('S01',tilt=50)]
        r=frame_annotations({'native_size':[640,480],'slots':slots})
        defects=[(a['slot_id'],a['category_id']) for a in r['annotations'] if a['category_id']!=1]
        self.assertEqual(defects,[('G03',2),('G04',3)])
        self.assertEqual(r['gap_errors'],['G05'])
        self.assertTrue(all(s['tray']=='G' for s in r['slots']))
    def test_full_order_ignores_duplicates_and_s_tray(self):
        occupied=[f"G{i:02d}" for i in range(1,31)]
        result=good_tray_order_status(occupied+["G01", "S01", "S02"])
        self.assertTrue(result['order_complete'])
        self.assertEqual(result['occupied_count'],30)
        self.assertEqual(result['empty_slots'],[])
    def test_partial_prefix_is_incomplete(self):
        result=good_tray_order_status([f"G{i:02d}" for i in range(1,11)])
        self.assertFalse(result['order_complete'])
        self.assertEqual(result['empty_slots'],[f"G{i:02d}" for i in range(11,31)])
    def test_middle_gap_and_empty_tray(self):
        result=good_tray_order_status([f"G{i:02d}" for i in range(1,31) if i!=12])
        self.assertEqual(result['empty_slots'],['G12'])
        self.assertEqual(result['order_status'],'incomplete')
        self.assertEqual(good_tray_order_status([])['occupied_count'],0)
        self.assertEqual(len(good_tray_order_status(['S01'])['empty_slots']),30)
    def test_clipping(self):
        self.assertEqual(clipped_box([-5,-5,10,10],640,480),[0,0,10,10])
        self.assertIsNone(clipped_box([700,20,720,40],640,480))
        with self.assertRaises(ValueError):clipped_box([20,20,10,10],640,480)
    def test_mapping(self):
        a=self.slot();b=self.slot('G02');b['center_native']=[20,40]
        self.assertEqual(assign_to_slot([15,15,10,10],[a,b]),'G01')
        self.assertIsNone(assign_to_slot([480,300,20,20],[a,b]))
if __name__=='__main__':unittest.main()
