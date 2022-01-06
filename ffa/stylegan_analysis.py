import sys
import torch
import pickle

from neuralmodels_torch import VoxelEncoder, load_images
from neuralmodels_torch import preprocess, load_mapping_weights, Constants
from neuralmodels_torch import ReprDistResize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#### PREPARE GAN###########

dnn_path = "/home/ratan/Documents/packages/stylegan2-ada-pytorch/"
torch_utils_path = "/home/ratan/Documents/packages/stylegan2-ada-pytorch/torch_utils/"

sys.path.append(dnn_path)
#sys.path.append(torch_utils_path)

model_path  = "/home/ratan/.cache/dnnlib/downloads/2366a0cffcb890fdb0ee0a193f4e0440_https___nvlabs-fi-cdn.nvidia.com_stylegan2-ada-pytorch_pretrained_ffhq.pkl"

with open(model_path, 'rb') as f:
    SGAN = pickle.load(f)['G_ema'].cuda()

SGAN.to(device).eval()
SGAN.requires_grad_(False)

#### LOAD IN LFFA-VOXEL DISTANCE MODEL:

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
print("neural model is loaded!")


######### COMPUTE EIGENFEATURES ################
geom_path = "/home/ratan/Documents/packages/GAN-Geometry"
sys.path.append(geom_path)
from core import get_full_hessian, hessian_compute, save_imgrid, show_imgrid
from core.GAN_utils import StyleGAN2CUSTOM_wrapper
from core.hessian_axis_visualize import vis_eigen_action, vis_eigen_explore, vis_distance_curve


G = StyleGAN2CUSTOM_wrapper(SGAN)

#%%
feat = G.sample_vector(device=device).detach().clone()
eva_FI, evc_FI, H_FI = hessian_compute(G, feat, ImDist, hessian_method="ForwardIter", cutoff=10)

#%%
# eva_BI, evc_BI, H_BI = hessian_compute(G, feat, ImDist, hessian_method="BackwardIter", cutoff=30)
# eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP")
#%%
mtg, codes_all, = vis_eigen_explore(feat.cpu().numpy(), evc_FI, eva_FI, G, ImDist=None, eiglist=[1,2,3,4,7], transpose=False,
      maxdist=0.6, scaling=None, rown=7, sphere=False, distrown=15,
      save=True, namestr="ffa-stylegan-demo")
#%%
vis_distance_curve(feat.cpu().numpy(), evc_FI, eva_FI, G, ImDist, eiglist=[1,2,3,4,7],
                   maxdist=0.6, rown=7, sphere=False, distrown=15, namestr="ffa-stylegan-demo")

print("completed")
