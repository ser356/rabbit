"""Pruebas pequeñas: no usan el vídeo ni el checkpoint del usuario."""
import sys, unittest
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from anchor_boxes import load_anchors
from tracker_v22 import AnchoredTracker,relative_motion
from temporal_v22 import window_features,TemporalNet

class TestV22(unittest.TestCase):
    def test_temporal_shape(self):
        import torch
        features=np.arange(48,dtype=np.float32).reshape(12,4)
        times=np.arange(12)*.5
        x=window_features(features,times,0,3.5,8)
        self.assertEqual(x.shape,(8,4))
        network=TemporalNet(4,2)
        self.assertEqual(tuple(network(torch.as_tensor(x)[None]).shape),(1,2))
    def test_tracker_reseeds(self):
        rng=np.random.default_rng(42)
        box=np.array([55,45,100,90])
        texture=rng.integers(0,255,(45,45,3),dtype=np.uint8)
        anchors={'reference_width':160,'reference_height':120,
                 'anchors':[{'time':0.0,'bbox':box.tolist()},
                            {'time':3.0,'bbox':(box+np.array([12,0,12,0])).tolist()}]}
        tracker=AnchoredTracker(anchors)
        sources=[]
        for t in np.arange(0,3.5,.5):
            frame=np.zeros((120,160,3),dtype=np.uint8)
            x=55+round(t*4)
            frame[45:90,x:x+45]=texture
            tr=tracker.update(frame,float(t))
            sources.append(tr.source)
        self.assertEqual(sources[0],'anchor_keyframe')
        self.assertIn('lk_optical_flow',sources)
        self.assertEqual(sources[-1],'anchor_keyframe')
    def test_motion_no_previous(self):
        pic=np.zeros((80,90,3),dtype=np.uint8)
        r=relative_motion(None,pic,(10,10,40,40),None)
        self.assertEqual(r['activity_quality'],'no_previous')

if __name__=='__main__':unittest.main()
