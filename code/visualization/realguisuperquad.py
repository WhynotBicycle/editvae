"""
Superquadrics in Python
=======================
Script to create an Interactive Superquadric drawing
tool.

Based on: Pratik Mallya (December 2011)
Author: Shidi Li (Feb. 2021)
"""

## colors:
# 'Accent' or 'Blues' or 'BrBG' or 'BuGn' or 'BuPu' or 'CMRmap' or 'Dark2' or 'GnBu' or 'Greens' or 'Greys' or 'OrRd' or 'Oranges' or 'PRGn' or 'Paired' or 'Pastel1' or 'Pastel2' or 'PiYG' or 'PuBu' or 'PuBuGn' or 'PuOr' or 'PuRd' or 'Purples' or 'RdBu' or 'RdGy' or 'RdPu' or 'RdYlBu' or 'RdYlGn' or 'Reds' or 'Set1' or 'Set2' or 'Set3' or 'Spectral' or 'Vega10' or 'Vega20' or 'Vega20b' or 'Vega20c' or 'Wistia' or 'YlGn' or 'YlGnBu' or 'YlOrBr' or 'YlOrRd' or 'afmhot' or 'autumn' or 'binary' or 'black-white' or 'blue-red' or 'bone' or 'brg' or 'bwr' or 'cool' or 'coolwarm' or 'copper' or 'cubehelix' or 'file' or 'flag' or 'gist_earth' or 'gist_gray' or 'gist_heat' or 'gist_ncar' or 'gist_rainbow' or 'gist_stern' or 'gist_yarg' or 'gnuplot' or 'gnuplot2' or 'gray' or 'hot' or 'hsv' or 'inferno' or 'jet' or 'magma' or 'nipy_spectral' or 'ocean' or 'pink' or 'plasma' or 'prism' or 'rainbow' or 'seismic' or 'spectral' or 'spring' or 'summer' or 'terrain' or 'viridis' or 'winter'

## shape of points
# 2darrow’ or ‘2dcircle’ or ‘2dcross’ or ‘2ddash’ or ‘2ddiamond’ or ‘2dhooked_arrow’ or ‘2dsquare’ or ‘2dthick_arrow’ or ‘2dthick_cross’ or ‘2dtriangle’ or ‘2dvertex’ or ‘arrow’ or ‘axes’ or ‘cone’ or ‘cube’ or ‘cylinder’ or ‘point’ or ‘sphere’

import torch
import numpy as np
import sys
sys.path.append('code/primitive_fitting')
from primitives import deform, quaternions_to_rotation_matrices



def _visR(pt, theta=None, phi=None, gamma=None):
    if theta is None:
        theta = torch.tensor(1/2 * math.pi) # x
    if phi is None:
        phi = torch.tensor(-1/18 * math.pi) # y
    if gamma is None:
        gamma = torch.tensor(-1/4 * math.pi) # z

    Rx = torch.tensor([[1, 0, 0],
                       [0, torch.cos(theta), -torch.sin(theta)],
                       [0, torch.sin(theta), torch.cos(theta)]])
    Ry = torch.tensor([[torch.cos(phi), 0, torch.sin(phi)],
                       [0, 1, 0],
                       [-torch.sin(phi), 0, torch.cos(phi)]])
    Rz = torch.tensor([[torch.cos(gamma), -torch.sin(gamma), 0],
                       [torch.sin(gamma), torch.cos(gamma), 0],
                       [0, 0, 1]])

    pt = Rx.matmul(pt.T).T
    pt = Ry.matmul(pt.T).T
    pt = Rz.matmul(pt.T).T
    return pt

def fexp(x,p):
    """a different kind of exponentiation"""
    return (torch.sign(x)*(torch.abs(x)**p))

def tens_fld(trans, rotas, sz, sp, df, vis_r=None, origin_coord=False):
    """this module plots superquadratic surfaces with the given parameters"""
    phi, theta = torch.tensor(np.mgrid[0:np.pi:80j, 0:2*np.pi:80j], dtype = torch.float)
    phi = phi.reshape((-1, 1)).squeeze(1)
    theta = theta.reshape((-1, 1)).squeeze(1)
    x = sz[0]*(fexp(torch.sin(phi),sp[0])) *(fexp(torch.cos(theta),sp[1]))
    y = sz[1]*(fexp(torch.sin(phi),sp[0])) *(fexp(torch.sin(theta),sp[1]))
    z = sz[2]*(fexp(torch.cos(phi),sp[0]))
    samples = torch.cat([x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)], dim=1) # N * 3
    
    ## deformation
    sz = sz.unsqueeze(0); df = df.unsqueeze(0); trans = trans.unsqueeze(0)
    rotas = rotas.unsqueeze(0); sp = sp.unsqueeze(0) # B * X
    samples = deform(samples.unsqueeze(0).unsqueeze(0), sz.unsqueeze(0), df.unsqueeze(0)).squeeze(1) # B * N * 3
    
    if not origin_coord:
        R = quaternions_to_rotation_matrices(rotas).transpose(1, 2)
        samples = R.unsqueeze(1).matmul(samples.unsqueeze(-1)).squeeze(-1)
        samples = samples + trans.unsqueeze(1)
    samples = samples.squeeze(0) # N * 3
    
    ## rotation for transformation
    if vis_r is not None:
        samples = _visR(samples, vis_r[0], vis_r[1], vis_r[2])
    
    x, y, z = torch.split(samples, 1, dim=1)
    x = x.reshape((-1, 80)).numpy()
    y = y.reshape((-1, 80)).numpy()
    z = z.reshape((-1, 80)).numpy()
    return x , y , z 


from traits.api import HasTraits, Range, Instance, \
                    on_trait_change
from traitsui.api import View, Item, HGroup
from tvtk.pyface.scene_editor import SceneEditor
from mayavi.tools.mlab_scene_model import \
                    MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene


# q_size: [alpha_x, alpha_y, alpha_z]
# q_shape: [shape1, shape2]
# q_trans: [q_trans_x, q_trans_y, q_trans_z]
# q_rota: [q_rota_a, q_rota_b, q_rota_c, q_rota_d]
# q_deform: [q_deform1, q_deform2]
class MayaviVisualization(HasTraits):
#     alpha = Range(0.0, 4.0,  1.0/4)
#     beta  = Range(0.0, 4.0,  1.0/4)
    alpha = 1
    beta = 1
    scene = Instance(MlabSceneModel, ())

    def __init__(self, superquads=None, point_clouds=None, sq_colors = None, sq_opa=.8,
                 pc_colors=None, pc_opa=[1,1,1,1,1,1,1,1], vis_r=None,
                 origin_coord=False, vis_sq=True, pt_size=.025):
        # Do not forget to call the parent's __init__
        HasTraits.__init__(self)
#         fig = self.scene.mlab.figure(fgcolor=(0., 0., 0.), bgcolor=(1, 1, 1), size=(550,550))
#         fig.scene.disable_render = False
        
        if superquads is not None:
            for i in range(len(superquads)):
                parts = superquads[i]
                trans = parts[0].detach().cpu(); rotas = parts[1].detach().cpu(); sz = parts[2].detach().cpu()
                sp = parts[3].detach().cpu(); df = parts[4].detach().cpu()
                # superquadrics
                if vis_sq:
                    x, y, z, = tens_fld(trans, rotas, sz, sp, df, vis_r, origin_coord)
                    self.plot = self.scene.mlab.mesh(x, y, z, color=sq_colors[i], representation='surface', opacity=sq_opa)
            
        # draw point cloud
        # expect N*3
        if point_clouds is not None:
            for i in range(len(point_clouds)):
                pc = point_clouds[i].detach().cpu()
                
                if origin_coord:
                    R = quaternions_to_rotation_matrices(rotas.unsqueeze(0))
                    pc = pc - trans.unsqueeze(0)
                    pc = R.unsqueeze(1).matmul(pc.unsqueeze(-1)).squeeze(-1).squeeze(0)
                
                if vis_r is not None:
                    pc = _visR(pc, vis_r[0], vis_r[1], vis_r[2])
                x, y, z = pc.detach().cpu().split(1, dim=1)
                x = x.squeeze(1).numpy()
                y = y.squeeze(1).numpy()
                z = z.squeeze(1).numpy()
                s = np.ones(x.shape[0])
                self.points = self.scene.mlab.points3d(x, y, z, s, color=pc_colors[i], 
                                                       scale_factor=pt_size, opacity=pc_opa[i])
                self.points.actor.property.shading = True
                self.scene.movie_maker.record = True
#                 print(self.points.actor.property)
        

    @on_trait_change('beta,alpha')
    def update_plot(self):
        x, y, z, = tens_fld(1,1,1,self.beta, self.alpha)
        self.plot.mlab_source.set(x=x, y=y, z=z)
        
#     @self.scene.mlab.animate(delay = 100)
#     def updateAnimation():
#         t = 0.0
#         while True:
#             self.points.mlab_source.set(x = np.cos(t), y = np.sin(t), z = 0)
#             t += 0.1
#             yield


    # the layout of the dialog created
#     self.updateAnimation()
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                    height=550, width=550, show_label=False),
                HGroup(
                        '_', 'beta', 'alpha',
                    ),
                )

# visualization = Visualization()
# visualization.configure_traits()


