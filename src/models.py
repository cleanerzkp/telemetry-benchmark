"""Three fixed detectors. No hyperparameter search or test-based selection."""
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import IsolationForest

class DenoisingAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(38,32),nn.ReLU(),nn.Linear(32,8),nn.ReLU(),
                                   nn.Linear(8,32),nn.ReLU(),nn.Linear(32,38))
    def forward(self,x): return self.network(x)

def fit_models(x,cfg):
    torch.set_num_threads(1); torch.manual_seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    center=x.mean(0); scale=np.maximum(x.std(0),cfg['std_floor'])
    z=((x-center)/scale).astype('float32')
    # 38 independent univariate forests. No joint cross-channel feature learning.
    forests=[IsolationForest(random_state=cfg['seed'],n_jobs=1).fit(z[:,j:j+1]) for j in range(38)]
    ae=DenoisingAE(); optimizer=torch.optim.Adam(ae.parameters())
    data=torch.from_numpy(z); losses=[]
    for _ in range(cfg['epochs']):
        order=torch.randperm(len(data)); total=0.
        for ids in order.split(cfg['batch_size']):
            target=data[ids]; corrupted=target.clone()
            corrupted[torch.rand_like(corrupted)<cfg['mask_probability']]=0.
            optimizer.zero_grad(); loss=nn.functional.mse_loss(ae(corrupted),target)
            loss.backward(); optimizer.step(); total+=loss.item()*len(ids)
        losses.append(total/len(data))
    return dict(center=center,scale=scale,forests=forests,
                ae_state={k:v.detach().cpu() for k,v in ae.state_dict().items()},
                loss=losses,plot_channel=int(np.argmax(x.var(0))))

def rolling_score(x,scale,window):
    if len(x)<=window: raise ValueError('Sequence shorter than context')
    sums=np.vstack([np.zeros((1,x.shape[1])),np.cumsum(x,axis=0)])
    # Mean of previous W samples; x[t] and future x are excluded.
    previous=(sums[window:len(x)]-sums[:len(x)-window])/window
    return np.max(np.abs(x[window:]-previous)/scale,axis=1)

def score_models(model,x,cfg):
    w=cfg['rolling_window']; z=((x-model['center'])/model['scale']).astype('float32')
    forest_scores=np.column_stack([-f.score_samples(z[w:,j:j+1]) for j,f in enumerate(model['forests'])])
    ae=DenoisingAE(); ae.load_state_dict(model['ae_state']); ae.eval()
    with torch.no_grad():
        target=torch.from_numpy(z[w:])
        reconstruction=ae(target).numpy()
    contributions=(reconstruction-z[w:])**2
    return {'rolling':rolling_score(x,model['scale'],w),
            'iforest':forest_scores.max(axis=1),
            'autoencoder':contributions.mean(axis=1)},contributions
