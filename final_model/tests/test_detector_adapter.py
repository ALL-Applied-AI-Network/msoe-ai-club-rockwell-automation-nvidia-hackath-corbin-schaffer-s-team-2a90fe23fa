import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from detector_adapter import RTDETRVideoAdapter,VideoInputContract
class Detector(torch.nn.Module):
    def forward(self,x):
        # Known probabilities/categories and relative cx/cy/w/h boxes.
        return {'pred_logits':x.new_tensor([[[ -10.,10.,-10.,-10.],[-10.,-10.,10.,-10.]]]),
                'pred_boxes':x.new_tensor([[[.5,.5,.2,.4],[.25,.25,.1,.1]]])}
class AdapterTests(unittest.TestCase):
    def test_box_coordinates_and_labels(self):
        adapter=RTDETRVideoAdapter(Detector(),num_select=2)
        out=adapter(torch.zeros(1,3,480,640))
        bylabel={int(row[5]):row for row in out}
        torch.testing.assert_close(bylabel[1][:4],torch.tensor([256.,144.,384.,336.]))
        torch.testing.assert_close(bylabel[2][:4],torch.tensor([128.,96.,192.,144.]))
        self.assertTrue(all(float(row[4])>.99 for row in out))
    def test_traced_contract_rejects_wrong_size(self):
        x=torch.zeros(1,3,480,640);adapter=RTDETRVideoAdapter(Detector(),num_select=2)
        scripted=torch.jit.script(VideoInputContract(torch.jit.trace(adapter,x,check_trace=False)))
        torch.testing.assert_close(scripted(x),adapter(x))
        with self.assertRaises((RuntimeError,torch.jit.Error)):scripted(torch.zeros(1,3,640,640))
if __name__=='__main__':unittest.main()
