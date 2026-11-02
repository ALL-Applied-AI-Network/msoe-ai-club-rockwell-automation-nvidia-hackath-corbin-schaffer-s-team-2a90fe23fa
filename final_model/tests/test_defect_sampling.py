import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from defect_sampling import sample_defects
class SamplingTests(unittest.TestCase):
    def test_extremes_and_reproducibility(self):
        self.assertEqual(sample_defects(1,0),{})
        self.assertEqual(len(sample_defects(1,1)),30)
        self.assertEqual(sample_defects(18,.5),sample_defects(18,.5))
    def test_probability_and_validation(self):
        n=sum(len(sample_defects(i,.5)) for i in range(1000))
        low=sum(len(sample_defects(i,.1)) for i in range(1000))
        self.assertTrue(.48<n/30000<.52);self.assertTrue(.08<low/30000<.12)
        for rate in [-1,1.1,float('nan')]:
            with self.assertRaises(ValueError):sample_defects(1,rate)
if __name__=='__main__':unittest.main()
