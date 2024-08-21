import argparse
import math

import numpy as np

import mcubes

import torch

import trimesh

import models_coarse_cond, models_ae

from pathlib import Path

N = 8192

if __name__ == "__main__":
    parser = argparse.ArgumentParser('', add_help=False)
    # parser.add_argument('--ae', type=str, required=True) # 'kl_d512_m512_l16'
    # parser.add_argument('--ae-pth', type=str, required=True) # 'output/ae/kl_d512_m512_l16/checkpoint-199.pth'
    # parser.add_argument('--dm', type=str, required=True) # 'kl_d512_m512_l16_edm'
    # parser.add_argument('--dm-pth', type=str, required=True) # 'output/uncond_dm/kl_d512_m512_l16_edm/checkpoint-999.pth'
    
    parser.add_argument('--ae', type=str, default= 'kl_d512_m512_l8')
    parser.add_argument('--ae-pth', type=str, default='pretrained/ae/kl_d512_m512_l8/checkpoint-199.pth')

    parser.add_argument('--ae-cond', default='ae_d512_m512', type=str, metavar='MODEL',
                        help='Name of autoencoder')

    parser.add_argument('--ae-cond-pth', default='pretrained/ae/ae_d512_m512/checkpoint-199.pth', help='Autoencoder checkpoint')

    parser.add_argument('--dm', type=str, default='kl_d512_m512_l8_d24_edm')
    # parser.add_argument('--dm-pth', type=str, default='pretrained/class_cond_dm/kl_d512_m512_l8_d24_edm/checkpoint-499.pth')
    parser.add_argument('--dm-pth', type=str, default='output_sn/dm/kl_d512_m512_l8_d24_edm/checkpoint-4999.pth')
    parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
    

    args = parser.parse_args()
    print(args)

    device = args.device

    ae = models_ae.__dict__[args.ae](N=N)
    ae.eval()
    ae.load_state_dict(torch.load(args.ae_pth)['model'])
    ae.to(device)

    cond_ae = models_ae.__dict__[args.ae_cond](N=N)
    cond_ae.eval()
    cond_ae.load_state_dict(torch.load(args.ae_cond_pth)['model'])
    cond_ae.to(device)

    model = models_coarse_cond.__dict__[args.dm]()
    model.eval()

    model.load_state_dict(torch.load(args.dm_pth)['model'])
    model.to(device)

    density = 128
    gap = 2. / density
    x = np.linspace(-1, 1, density+1)
    y = np.linspace(-1, 1, density+1)
    z = np.linspace(-1, 1, density+1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].to(device, non_blocking=True)

    with torch.no_grad():
        # surface = np.load('/data/ljf/sdfdata/xzkb/xzkb.npz')['vol_points']
        # surface = torch.Tensor(surface)
        # surface = surface.unsqueeze(0)

        coarse_surface = np.load('/data/ljf/sdfdata/wushi/wushi_32.npz')['vol_points']
        coarse_surface = np.load('/data/ljf/shapenet_test/1006be65e7bc937e9141f9b58470d646/1006be65e7bc937e9141f9b58470d646_points.npz')['coarse_surface_points']
        coarse_surface = torch.Tensor(coarse_surface)
        coarse_surface = coarse_surface.unsqueeze(0)

        # print(surface.dtype)
        # print(surface.shape)
 
        # ind = np.random.default_rng().choice(
        #     surface[0].numpy().shape[0], N, replace=False
        # )
        # # print(ind.dtype)
        # # print(ind.shape)
        # ind = torch.Tensor(ind).long()
        # # print(ind.dtype)
        # # print(ind.shape)
        # # ind = ind.to(device)
        # surface2048 = surface[0][ind][None].float()
        # # print(surface2048.dtype)
        # surface2048 = surface2048.to(device, non_blocking=True)

        ind_coarse = np.random.default_rng().choice(
            coarse_surface[0].numpy().shape[0], N, replace=False
        )
        ind_coarse = torch.Tensor(ind_coarse).long()
        surface2048_coarse = coarse_surface[0][ind_coarse][None].float()
        surface2048_coarse = surface2048_coarse.to(device, non_blocking=True)

        input_points = trimesh.PointCloud(surface2048_coarse[0].cpu().numpy())
        input_points.export('input.ply')

        # kl, latents = ae.encode(surface2048)
        cond_latents = cond_ae.encode(surface2048_coarse)
        # model(surface2048, grid)['logits']
        # print(latents.shape)
        # print(latents.dtype)
        # exit(0)
        
        # latents = model.denoise(
        #     latents, cond=cond_latents,
        #     # num_steps=200
        # ).float()

        latents = model.sample(
            cond=cond_latents
        ).float()

        # print(latents.dtype)
        output = ae.decode(latents, grid).squeeze(-1)
        volume = output.view(density+1, density+1, density+1).permute(1, 0, 2).cpu().numpy()

        verts, faces = mcubes.marching_cubes(volume, 0)
        verts *= gap
        verts -= 1.
        m = trimesh.Trimesh(verts, faces)

        m.export('output.obj')