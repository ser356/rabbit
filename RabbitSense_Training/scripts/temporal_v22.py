"""Aprendizaje temporal supervisado a partir de clips etiquetados (no heurísticas)."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn

class TemporalNet(nn.Module):
    def __init__(self,feat_dim,n_classes,hidden=64):
        super().__init__()
        self.embed=nn.Sequential(nn.LayerNorm(feat_dim),nn.Linear(feat_dim,hidden),nn.GELU())
        self.rnn=nn.GRU(hidden,hidden,batch_first=True,bidirectional=True)
        self.head=nn.Sequential(nn.LayerNorm(hidden*2),nn.Dropout(.35),nn.Linear(hidden*2,n_classes))
    def forward(self,x):
        y,_=self.rnn(self.embed(x))
        return self.head(y.mean(1))

def window_features(features,times,start,end,num_frames=8):
    """Interpela una serie real hacia una ventana temporal fija.

    Únicamente válido para intervalos contenidos en el vídeo.
    """
    feats=np.asarray(features,dtype=np.float32)
    stamps=np.asarray(times,dtype=np.float64)
    if len(stamps)!=len(feats) or len(stamps)<2 or not np.all(np.diff(stamps)>0):
        raise ValueError('Serie temporal irregular o insuficiente')
    if end<=start or start<stamps[0]-1e-5 or start>stamps[-1]:
        raise ValueError('Ventana fuera de rango')
    targets=np.linspace(start,min(end,stamps[-1]),num_frames,dtype=np.float64)
    return np.stack([np.interp(targets,stamps,feats[:,i]) for i in range(feats.shape[1])],axis=1).astype('float32')

def load_action_model(path,device):
    ckpt=torch.load(path,map_location='cpu',weights_only=True)
    net=TemporalNet(int(ckpt['feat_dim']),len(ckpt['classes']),int(ckpt['hidden']))
    net.load_state_dict(ckpt['model_state'])
    return net.to(device).eval(),ckpt
