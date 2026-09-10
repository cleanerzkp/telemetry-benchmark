import unittest
import numpy as np
from src.metrics import intervals,aggregate,event_metrics
from src.models import rolling_score,DenoisingAE
import torch

class IncidentMetrics(unittest.TestCase):
    def metric(self,y,p,gap=0): return event_metrics(y,p,gap)[0]
    def test_empty_positive_vectors(self):
        m=self.metric([0,0],[0,0]);self.assertEqual(m['f1'],0);self.assertEqual(m['false_alarms'],0)
    def test_invalid_values(self):
        with self.assertRaises(ValueError): intervals([0,2])
        with self.assertRaises(ValueError): event_metrics([0],[0,1])
    def test_half_open_boundaries(self):
        self.assertEqual(intervals([1,1,0,1]),[(0,2),(3,4)])
        self.assertEqual(self.metric([1,0],[0,1])['matched'],0)
    def test_merge_deduplicates(self):
        m=self.metric([1,1,1],[1,0,1],1)
        self.assertEqual(m['alarm_episodes'],1);self.assertEqual(m['precision'],1)
    def test_unmerged_duplicate_penalty(self):
        m=self.metric([1,1,1],[1,0,1]);self.assertEqual(m['precision'],.5)
        self.assertEqual(m['duplicate_alarms'],1);self.assertEqual(m['false_alarms'],0)
    def test_merge_gap_does_not_create_fake_hit(self):
        m=self.metric([0,1,0],[1,0,1],1)
        self.assertEqual(m['matched'],0);self.assertEqual(m['false_alarms'],1)
    def test_long_alarm_does_not_claim_two_incidents(self):
        m=self.metric([1,0,1],[1,1,1]);self.assertEqual(m['matched'],1);self.assertEqual(m['recall'],.5)
    def test_false_alarm_rate(self):
        y=np.zeros(1440,int);p=y.copy();p[10]=1
        self.assertEqual(self.metric(y,p)['false_alarms_per_day'],1)
    def test_delay_uses_actual_exceedance(self):
        m=self.metric([0,0,1,1,1],[1,0,0,0,1],3)
        self.assertEqual(m['median_delay_minutes'],2)
    def test_exact_gap(self):
        self.assertEqual(len(aggregate([1,0,0,1],2)),1)
        self.assertEqual(len(aggregate([1,0,0,1],1)),2)

class Models(unittest.TestCase):
    def test_rolling_is_causal(self):
        x=np.arange(20.)[:,None];changed=x.copy();changed[15:]=999
        np.testing.assert_allclose(rolling_score(x,np.ones(1),3)[:12],rolling_score(changed,np.ones(1),3)[:12])
        self.assertEqual(rolling_score(x,np.ones(1),3)[0],2.)
    def test_zero_flat_signal(self):
        np.testing.assert_array_equal(rolling_score(np.ones((10,2)),np.ones(2),3),np.zeros(7))
    def test_network_shape_and_gradient(self):
        model=DenoisingAE();x=torch.randn(8,38)
        loss=((model(x)-x)**2).mean();loss.backward()
        self.assertEqual(model(x).shape,x.shape)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

if __name__=='__main__':unittest.main()
