import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import numpy as np

#link api functions to your own codebase

def create_nerfs(args):
    """Instantiate NeRF's MLP model.
    """
    model = NeRF()
    #model = nn.DataParallel(model).to(device)
    model = model.to(device)

    model_fine = NeRF()
    #model_fine = nn.DataParallel(model_fine).to(device)
    model_fine = model_fine.to(device)

    print('Loading NeRF from', args.weight)
    ckpt = torch.load(args.weight)
    # Load model
    model.load_state_dict(ckpt['model_state_dict'])
    # if model_fine is not None:
    model_fine.load_state_dict(ckpt['model_fine_state_dict'])

    model_cfg = {
        'network_fn': model,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'N_importance': args.N_importance
    }
    return model_cfg

def render_image(camK, W=None, H=None, c2w=None, chunk=1024*32, **kwargs):
    # 1. create rays
    if kwargs['scale'] != 0:
        camK = camK/kwargs['scale']
    camK[2, 2] = 1.0

    rays_o, rays_d = camera_rays(camK, W, H, c2w)
    #sh = [W,H]
    sh = rays_d.shape  # [..., 3]
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    # 2. calculate bounding-box of each ray
    rays = torch.cat([rays_o, rays_d], dim=-1)
    pose = kwargs['pose']
    bbox = kwargs['bbox']
    bounds = ray_box_intersection(rays, bbox)

    # render rays.
    rays = torch.cat([rays, bounds], dim=-1)
    all_ret = batchify(render_rays, chunk, rays, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    rgb = float2uint8(all_ret['rgb'].numpy())
    alpha = float2uint8(all_ret['alpha'].cpu().numpy())
    alpha = np.dstack((alpha, alpha, alpha))
    depth = all_ret['depth'].numpy()

    depth = 1./depth
    depth_min = depth.min()
    depth_max = depth.max()
    depth = (depth - depth_min)/(depth_max-depth_min)
    depth = (depth*255).astype(np.uint8)
    depth = np.dstack((depth, depth, depth))

    image = np.hstack((rgb, alpha, depth))
    return image


def render_rays(ray_batch,
                model,
                model_fine,
                pose,
                batch_size,
                N_samples=64,
                N_importance=64,
                **kwargs):

    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]

    near = 1.0
    far = 4.0

    # render pure scene.
    t_vals = torch.linspace(0., 1., steps=N_samples)
    t_vals = t_vals.expand([N_rays, N_samples])

    z_vals = near.unsqueeze(-1)*(1.-t_vals) + far.unsqueeze(-1)*t_vals

    # rotate rays according to object pose. this can also be compensated
    rays_o = rays_o - pose[:3, 3]
    rays_o = torch.matmul(pose[:3, :3].T, rays_o.T).T/pose[3, 3]
    rays_d = torch.matmul(pose[:3, :3].T, rays_d.T).T/pose[3, 3]

    pts = rays_o[..., None, :] + rays_d[..., None, :]*z_vals[..., :, None]
    viewdirs = F.normalize(rays_d, dim=-1)

    ret = batchify(model, batch_size, pts, viewdirs)
    weight = sigma2weights(ret['sigma'], z_vals, rays_d)

    z_vals_mid = .5*(z_vals[..., 1:]+z_vals[..., :-1])
    z_samples = importance_sampling(
        z_vals_mid, weight[..., 1:-1], N_importance, uniform=True)
    z_samples = z_samples.detach()
    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

    pts = rays_o[..., None, :] + rays_d[..., None, :]*z_vals[..., :, None]
    ret = batchify(model_fine, batch_size, pts, viewdirs)

    weight = sigma2weights(ret['sigma'], z_vals, rays_d)

    rgb = torch.sum(torch.sigmoid(ret['rgb'])*weight[..., None], dim=-2)
    depth = torch.sum(z_vals*weight, dim=-1)
    alpha = torch.sum(weight)

    return {'rgb': rgb, 'depth': depth, 'alpha': alpha}


sys.argv = ['-f']
# get default settings.
args = get_configs()
args.n_gpus = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor')
model_cfg = create_nerfs(args)

def render(data):
    # object bounding-box, used for calculate z_min and z_max
    bbox = torch.zeros((6, 3))
    expand_factor = 1.2
    for i in range(3):
        bbox[i, 0] = (data['bboxObj']['max']['x'] +
                      data['bboxObj']['min']['x'])/2.0
        bbox[i, 1] = (data['bboxObj']['max']['y'] +
                      data['bboxObj']['min']['y'])/2.0
        bbox[i, 2] = (data['bboxObj']['max']['z'] +
                      data['bboxObj']['min']['z'])/2.0

    bbox[3, 0] = (data['bboxObj']['max']['x'] -
                  data['bboxObj']['min']['x'])*expand_factor
    bbox[4, 1] = (data['bboxObj']['max']['y'] -
                  data['bboxObj']['min']['y'])*expand_factor
    bbox[5, 2] = (data['bboxObj']['max']['z'] -
                  data['bboxObj']['min']['z'])*expand_factor

    bbox[0] -= bbox[5]/2.0
    bbox[1] -= bbox[3]/2.0
    bbox[2] -= bbox[4]/2.0

    pose = torch.zeros((4, 4))
    pose[0, 3] = data['transObj']['x']
    pose[1, 3] = data['transObj']['y']
    pose[2, 3] = data['transObj']['z']

    quat = torch.zeros((1, 4))
    quat[0, 0] = data['rotObj']['_w']
    quat[0, 1] = data['rotObj']['_x']
    quat[0, 2] = data['rotObj']['_y']
    quat[0, 3] = data['rotObj']['_z']
    pose[:3, :3] = quat2mat(quat)[0]
    pose[3, 3] = data['scaleObj']['x']

    model_cfg.update(bbox=bbox)
    model_cfg.update(pose=pose)
    model_cfg.update(scale=data['renderScale'])

    c2w = torch.eye(4)
    c2w[0, 3] = data['transCamera']['x']
    c2w[1, 3] = data['transCamera']['x']
    quat = torch.zeros((1, 4))
    quat[0, 0] = data['rotCamera']['_w']
    quat[0, 1] = data['rotCamera']['_x']
    quat[0, 2] = data['rotCamera']['_y']
    quat[0, 3] = data['rotCamera']['_z']
    c2w[:3, :3] = quat2mat(quat)[0]

    camK = torch.Tensor(camK).to(device)

    with torch.no_grad():
        image = render_image(camK, None, None, c2w, args.chunk, **model_cfg)
        return image
