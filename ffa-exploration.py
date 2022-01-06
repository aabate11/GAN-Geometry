from core import get_full_hessian, hessian_compute, save_imgrid, show_imgrid
from core.GAN_utils import DCGAN_wrapper, loadDCGAN, BigGAN_wrapper, loadBigGAN, upconvGAN
import torch
import matplotlib.pyplot as plt

from neuralmodels_torch import VoxelEncoder, load_images
from neuralmodels_torch import preprocess, load_mapping_weights, Constants
from neuralmodels_torch import ReprDistResize

from core.hessian_axis_visualize import vis_eigen_action, vis_eigen_explore, vis_distance_curve

#### LOAD IN LFFA-VOXEL DISTANCE MODEL:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

roi = "lffa-voxels"
#model_path = "C:\\Users\\aabat\\Documents\\kanlab\\pytorch_models\\lffa_voxels".format(roi)
#weights_dir = 'C:\\Users\\aabat\\Documents\\nm_weights\\'

model_path = "/home/ratan/Documents/pytorch_models/lffa_voxels"
weights_dir = "/home/ratan/Documents/mapping_weights"

mapping_weights = load_mapping_weights(roi, weights_dir)

model = VoxelEncoder(model_path + ".py",
                    model_path + ".pth",
                    mapping_weights,
                    device = device, batch_size=5)

ImDist = ReprDistResize(model, batch_size=4)
print("model is loaded!")


#### LOAD IN GAN
BGAN = loadBigGAN()  # Default to be "biggan-deep-256"

BGAN.cuda().eval()
BGAN.requires_grad_(False)
G = BigGAN_wrapper(BGAN)

#%%
feat = G.sample_vector(device="cuda", class_id=321).detach().clone()
eva_FI, evc_FI, H_FI = hessian_compute(G, feat, ImDist, hessian_method="ForwardIter", cutoff=10)

#%%
# eva_BI, evc_BI, H_BI = hessian_compute(G, feat, ImDist, hessian_method="BackwardIter", cutoff=30)
# eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP")
#%%
mtg, codes_all, = vis_eigen_explore(feat.cpu().numpy(), evc_FI, eva_FI, G, ImDist=None, eiglist=[1,2,3,4,7], transpose=False,
      maxdist=0.6, scaling=None, rown=7, sphere=False, distrown=15,
      save=False, namestr="ffa2_demo")
#%%
vis_distance_curve(feat.cpu().numpy(), evc_FI, eva_FI, G, ImDist, eiglist=[1,2,3,4,7],
                   maxdist=0.6, rown=7, sphere=False, distrown=15, namestr="ffa_demo")
