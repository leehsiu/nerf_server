import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
import numpy as np
import imageio
import json
import random
import time
import matplotlib.pyplot as plt
import scipy.io

from msranerf.models.nerf_helper import *
from msranerf.models.nerf import get_embedder
from msranerf.models.nerf import NeRF

from msranerf.utils.config import get_configs
from msranerf.utils.geometry import *

def create_nerfs(args, modelW):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, 0)
    input_ch_views = 0
    embeddirs_fn, input_ch_views = get_embedder(
        args.multires_views, 0)
    output_ch = 5
    skips = [4]

    def network_query_fn(inputs, viewdirs, network_fn):
        return run_network(inputs, viewdirs, network_fn,
                           embed_fn,
                           embeddirs_fn,
                           args.netchunk*args.n_gpus)

    model_cfgs = []
    for w in modelW:
        model = NeRF(D=args.netdepth, W=args.netwidth,
                     input_ch=input_ch, output_ch=output_ch, skips=skips,
                     input_ch_views=input_ch_views, use_viewdirs=True)
        #model = nn.DataParallel(model).to(device)
        model = model.to(device)

        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=True)
        #model_fine = nn.DataParallel(model_fine).to(device)
        model_fine = model_fine.to(device)

        print('Loading NeRF from', w)
        ckpt = torch.load(w)
        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        # if model_fine is not None:
        model_fine.load_state_dict(ckpt['network_fine_state_dict'])

        model_cfg = {
            'network_query_fn': network_query_fn,
            'network_fn': model,
            'network_fine': model_fine,
        }
        model_cfgs.append(model_cfg)
    return model_cfgs


sys.argv = ['-f']
args = get_configs()
args.n_gpus = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_weights = []
model_weights.append('/home/v-xiuli1/weights/nerf/fortress.tar')
model_weights.append('/home/v-xiuli1/weights/nerf/hotdog.tar')
torch.set_default_tensor_type('torch.cuda.FloatTensor')
model_cfgs = create_nerfs(args,model_weights)

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(pts, dirs, model, embedder_pts, embedder_dirs, netchunk=1024*64):

    inputs_flat = torch.reshape(pts, [-1, pts.shape[-1]])
    embedded = embedder_pts(inputs_flat)
    if dirs is not None:
        input_dirs = dirs[:, None].expand(pts.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embedder_dirs(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)
    outputs_flat = batchify(model, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(
        pts.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(camK, rays_flat, render_cfgs, chunk=1024*32):
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(camK, rays_flat[i:i+chunk], render_cfgs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render_image(camK, W=None, H=None, c2w=None, render_cfgs=[], chunk=1024*32):
    # render image.
    # Args:
    #   camK:3x3,intrinsic.
    #   W:int,Width of image in pixels.
    #   H:int,Height of image in pixels.
    #   c2w:3x4,extrinsic.
    #   render_cfgs:configurations of each object: weight,bbox,object-view,ndc.
    #   chunk: int. Maximum number of rays to process simultaneously. Used to
    #     control maximum memory usage. Does not affect final results.
    # Returns:
    #   rgb_map: [batch_size, 3]. Predicted RGB values for rays.
    #   depth_map: [batch_size]. Disparity map. Inverse of depth.
    #   acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
    # 1. create rays
    rays_o, rays_d = camera_rays(camK, W, H, c2w)
    #sh = [W,H]
    sh = rays_d.shape  # [..., 3]
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    # 2. calculate bounding-box of each ray
    rays = torch.cat([rays_o, rays_d], dim=-1)
    bounds = []
    for cfg in render_cfgs:
        #pose = [Nx7] (xyz,s,wijk)
        pose = cfg['pose']
        bbox = cfg['bbox']
        ndc = cfg['ndc']
        if ndc:
            bound = torch.zeros((rays.shape[0], 2))
            bound[:, 0] = 1.0
            bound[:, 1] = -1.0
        else:
            bound = line_box_intersection(rays, bbox)
        bounds.append(bound)
    bounds = torch.cat(bounds, dim=-1)

    # render rays.
    rays = torch.cat([rays, bounds], dim=-1)
    all_ret = batchify_rays(camK, rays, render_cfgs, chunk)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map']
    k_extract += ['acc_map', 'depth_scene']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def render_path(camK, c2w, render_cfgs, chunk, render_factor=0):
    # downsample for render efficency.
    if render_factor != 0:
        camK = camK/render_factor
    camK[2, 2] = 1.0
    rgb,acc,depth, _ = render_image(
        camK, c2w=c2w[:3, :4], render_cfgs=render_cfgs, chunk=chunk)
    rgb = rgb.cpu().numpy()
    acc = acc.cpu().numpy()
    depth = depth.cpu().numpy()

    rgb = to8b(rgb)
    acc = to8b(acc)

    depth = 1./depth
    depth_min = depth.min()
    depth_max = depth.max()
    depth = (depth - depth_min)/(depth_max-depth_min)
    depth = (depth*255).astype(np.uint8)
    depth = np.dstack((depth,depth,depth))

    image_all = np.hstack((rgb,acc,depth))

    return image_all

def create_nerfs(args, modelW):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, 0)
    input_ch_views = 0
    embeddirs_fn, input_ch_views = get_embedder(
        args.multires_views, 0)
    output_ch = 5
    skips = [4]

    def network_query_fn(inputs, viewdirs, network_fn):
        return run_network(inputs, viewdirs, network_fn,
                           embed_fn,
                           embeddirs_fn,
                           args.netchunk*args.n_gpus)

    model_cfgs = []
    for w in modelW:
        model = NeRF(D=args.netdepth, W=args.netwidth,
                     input_ch=input_ch, output_ch=output_ch, skips=skips,
                     input_ch_views=input_ch_views, use_viewdirs=True)
        #model = nn.DataParallel(model).to(device)
        model = model.to(device)

        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=True)
        #model_fine = nn.DataParallel(model_fine).to(device)
        model_fine = model_fine.to(device)

        print('Loading NeRF from', w)
        ckpt = torch.load(w)
        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        # if model_fine is not None:
        model_fine.load_state_dict(ckpt['network_fine_state_dict'])

        model_cfg = {
            'network_query_fn': network_query_fn,
            'network_fn': model,
            'network_fine': model_fine,
        }
        model_cfgs.append(model_cfg)
    return model_cfgs


def sigma2alpha(sigma, dists):
    return 1. - torch.exp(-F.relu(sigma)*dists)


def sigma2weights(sigma, z_vals, rays_d):
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(
        dists[..., :1].shape)], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    alpha = sigma2alpha(sigma, dists)
    weights = alpha * \
        torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    return weights


def blendray(weights, rgb):
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
    acc_map = torch.sum(weights, -1)
    return rgb_map, acc_map


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
    """
    def raw2alpha(raw, dists, act_fn=F.relu): return 1. - \
        torch.exp(-act_fn(raw)*dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(
        dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * \
        torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]

    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map),
                            depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(camK,
                ray_batch,
                render_cfgs,
                N_samples=64,
                N_importance=64):
    # volumetric rendering.
    # Args.
    #   camK
    #   ray_batch
    #   render_cfgs.

    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    bounds = ray_batch[..., 6:]

    obj_near = bounds[..., 2]
    obj_far = bounds[..., 3]

    scene_id = torch.where(obj_far < 0)[0]
    obj_id = torch.where(obj_far > 0)[0]

    # render pure scene.
    t_vals = torch.linspace(0., 1., steps=N_samples)
    t_vals = t_vals.expand([N_rays, N_samples])

    network_query_fn = render_cfgs[0]['network_query_fn']
    network_fn_scene = render_cfgs[0]['network_fn']
    network_fine_scene = render_cfgs[0]['network_fine']

    rays_o_NDC, rays_d_NDC, t_offset = rays_to_NDC(rays_o, rays_d, camK, 1.0)
    pts = rays_o_NDC[..., None, :] + rays_d_NDC[..., None, :] * \
        t_vals[..., :, None]
    raw_scene = network_query_fn(
        pts, viewdirs, network_fn_scene)
    raw_scene_weight = sigma2weights(
        raw_scene[..., 3], t_vals, rays_d_NDC)
    t_vals_mid = .5 * (t_vals[..., 1:] + t_vals[..., :-1])
    t_samples = sample_pdf(
        t_vals_mid, raw_scene_weight[..., 1:-1], N_importance, det=True, pytest=False)
    t_samples = t_samples.detach()
    t_vals_scene, _ = torch.sort(
        torch.cat([t_vals, t_samples], -1), -1)
    pts = rays_o_NDC[..., None, :] + rays_d_NDC[..., None, :] * \
        t_vals_scene[..., :, None]
    raw_scene = network_query_fn(pts, viewdirs, network_fine_scene)
    raw_scene = torch.cat(
        [torch.zeros_like(raw_scene[..., :3]), raw_scene], dim=-1)
    raw_scene[..., 0] = 1.0

    if len(obj_id) > 0:
        pose = render_cfgs[1]['pose']
        t_vals_obj = obj_near[obj_id].unsqueeze(-1)*(
            1.-t_vals[obj_id, ...]) + obj_far[obj_id].unsqueeze(-1)*t_vals[obj_id, ...]
        rays_o_obj = rays_o[obj_id, :] - pose[:3, 3]
        rays_o_obj = mat3_dot_vec3(pose[:3, :3].t(), rays_o_obj)/pose[3, 3]
        rays_d_obj = mat3_dot_vec3(
            pose[:3, :3].t(), rays_d[obj_id, :])/pose[3, 3]

        pts = rays_o_obj[..., None, :] + \
            rays_d_obj[..., None, :]*t_vals_obj[..., :, None]
        viewdirs_obj = rays_d_obj / \
            torch.norm(rays_d_obj, dim=-1, keepdim=True)
        raw_obj = network_query_fn(
            pts, viewdirs_obj, render_cfgs[1]['network_fn'])
        raw_obj_weight = sigma2weights(
            raw_obj[..., 3], t_vals_obj, rays_d_obj)
        t_vals_mid = .5*(t_vals_obj[..., 1:]+t_vals_obj[..., :-1])
        t_samples = sample_pdf(
            t_vals_mid, raw_obj_weight[..., 1:-1], 128, det=True, pytest=False)
        t_samples = t_samples.detach()
        t_vals_obj, _ = torch.sort(
            torch.cat([t_vals_obj, t_samples], -1), -1)
        pts = rays_o_obj[..., None, :] + \
            rays_d_obj[..., None, :]*t_vals_obj[..., :, None]
        raw_obj = network_query_fn(
            pts, viewdirs_obj, render_cfgs[1]['network_fine'])
        raw_obj = torch.cat(
            [torch.zeros_like(raw_obj[..., :3]), raw_obj], dim=-1)
        raw_obj[..., 1] = 1.0

        t_vals_obj_NDC = 1 + 1.0 / \
            (-1.0+(t_vals_obj-t_offset[obj_id, ..., None])
             * rays_d[obj_id, ..., 2].unsqueeze(-1))

        #t_vals_merge = t_vals_obj
        #raw_merge = raw_obj

        t_vals_merge = t_vals_scene[obj_id, ...]
        t_vals_merge, t_vals_merge_id = torch.sort(
            torch.cat([t_vals_merge, t_vals_obj_NDC], -1), -1)
        raw_merge = torch.cat([raw_scene[obj_id, ...], raw_obj], dim=1)
        raw_merge = torch.split(raw_merge, 1, dim=-1)
        raw_merge = [torch.gather(elem.squeeze(-1), 1, t_vals_merge_id)
                     for elem in raw_merge]
        raw_merge = torch.stack(raw_merge, dim=-1)

        weights_merge = sigma2weights(
            raw_merge[..., -1], t_vals_merge, rays_d_NDC[obj_id, ...])
        rgb_merge, acc_merge = blendray(
            weights_merge, torch.sigmoid(raw_merge[..., :6]))

    weights_scene = sigma2weights(raw_scene[..., -1], t_vals_scene, rays_d_NDC)
    depth_scene = torch.sum(weights_scene*t_vals_scene, dim=-1)
    depth_scene = depth_scene / \
        (depth_scene-1+1e-7) * (1/rays_d[..., 2]) + t_offset
    rgb_scene, acc_scene = blendray(
        weights_scene, torch.sigmoid(raw_scene[..., :6]))

    rgb_full = rgb_scene.clone()
    rgb_obj = torch.zeros_like(rgb_scene)

    if len(obj_id) > 0:
        rgb_full[obj_id, ...] = rgb_full[obj_id, ...] * \
            (1-acc_merge[..., None])+rgb_merge*acc_merge[..., None]
        rgb_obj[obj_id, ...] = rgb_merge

    # ret = {'rgb_map0': rgb_scene[..., 3:],
    #        'rgb_map1': rgb_obj[..., 3:], 'rgb_map': rgb_full[..., 3:]}
    ret = {'rgb_map':rgb_full[...,3:]}
    ret['acc_map'] = rgb_full[...,:3]
    ret['depth_scene'] = depth_scene

    return ret


def render(data):


    dd = np.load('/home/v-xiuli1/databag/nerf/llff_camera.npz')
    render_poses = dd['camView']
    camK = dd['camK']


    # init bbox.
    # update object configs.
    bbox = torch.zeros((6, 3))
    pose = torch.eye(4)
    model_cfgs[0].update(bbox=bbox)
    model_cfgs[0].update(pose=pose)
    model_cfgs[0].update(ndc=True)

    bbox = torch.zeros((6, 3))
    expand_factor = 1.2
    for i in range(3):
        bbox[i, 0] = (data['bboxObj']['max']['x'] + data['bboxObj']['min']['x'])/2.0
        bbox[i, 1] = (data['bboxObj']['max']['y'] + data['bboxObj']['min']['y'])/2.0
        bbox[i, 2] = (data['bboxObj']['max']['z'] + data['bboxObj']['min']['z'])/2.0

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

    model_cfgs[1].update(bbox=bbox)
    model_cfgs[1].update(pose=pose)
    model_cfgs[1].update(ndc=False)

    c2w = torch.eye(4)
    c2w[0,3] = data['transCamera']['x']
    c2w[1,3] = data['transCamera']['x']
    quat = torch.zeros((1, 4))
    quat[0, 0] = data['rotCamera']['_w']
    quat[0, 1] = data['rotCamera']['_x']
    quat[0, 2] = data['rotCamera']['_y']
    quat[0, 3] = data['rotCamera']['_z']
    c2w[:3, :3] = quat2mat(quat)[0]

    camK = torch.Tensor(camK).to(device)

    with torch.no_grad():
        image = render_path(camK,c2w,model_cfgs,args.chunk,1./data['renderScale'])
        return image
