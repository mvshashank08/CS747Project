

from LinearModel import LinearModel
import torch.nn as nn
import numpy as np
import torch
import util
from Discriminator import ShapeDiscriminator, PoseDiscriminator, FullPoseDiscriminator
from SMPL import SMPL
from config import args
import config
import Resnet
import sys


class ThetaRegressor(LinearModel):
    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func, iterations):
        super(ThetaRegressor, self).__init__(fc_layers, use_dropout, drop_prob, use_ac_func)
        self.iterations = iterations
        batch_size = max(args.batch_size + args.batch_3d_size, args.eval_batch_size)
        mean_theta = np.tile(util.load_mean_theta(), batch_size).reshape((batch_size, -1))
        self.register_buffer('mean_theta', torch.from_numpy(mean_theta).float())
    '''
        param:
            inputs: is the output of encoder, which has 2048 features
        
        return:
            a list contains [ [theta1, theta1, ..., theta1], [theta2, theta2, ..., theta2], ... , ], shape is iterations X N X 85(or other theta count)
    '''
    def forward(self, inputs):
        thetas = []
        shape = inputs.shape
        theta = self.mean_theta[:shape[0], :]
        for _ in range(self.iterations):
            total_inputs = torch.cat([inputs, theta], 1)
            theta = theta + self.fc_blocks(total_inputs)
            thetas.append(theta)
        return 
    
class HMRNetBase(nn.Module):
    def __init__(self):
        super(HMRNetBase, self).__init__()
        assert args.crop_size == 224
        self.feature_count = 2048 
        self.enable_inter_supervions = False
        self.beta_count = 10
        self.smpl_model = args.smpl_model
        self.smpl_mean_theta_path = args.smpl_mean_theta_path
        self.total_theta_count = 85
        self.joint_count = 24

        print('start creating sub modules...')
        self._create_sub_modules()     
        
        
    def _create_sub_modules(self):
        '''
            SMPL can create a mesh from beta & theta
        '''
        self.smpl = SMPL(self.smpl_model, obj_saveable = True)
        print('creating resnet50')
        self.encoder = Resnet.load_Res50Model()
        '''
            regressor can predict betas(include beta and theta which needed by SMPL) from coder extracted from encoder in a iterative way
        '''
        fc_layers = [self.feature_count + self.total_theta_count, 1024, 1024, 85]
        use_dropout = [True, True, False]
        drop_prob = [0.5, 0.5, 0.5]
        use_ac_func = [True, True, False] #unfreeze the last layer
        iterations = 3
        self.regressor = ThetaRegressor(fc_layers, use_dropout, drop_prob, use_ac_func, iterations)
        self.iterations = iterations

        print('finished create the encoder modules...')

    def forward(self, inputs):
            feature = self.encoder(inputs)
            thetas = self.regressor(feature)
            detail_info = []
            for theta in thetas:
                detail_info.append(self._calc_detail_info(theta))
            return detail_info
        else:
            assert 0

    '''
        purpose:
            calc verts, joint2d, joint3d, Rotation matrix
        inputs:
            theta: N X (3 + 72 + 10)
        return:
            thetas, verts, j2d, j3d, Rs
    '''
    
    def _calc_detail_info(self, theta):
        cam = theta[:, 0:3].contiguous()
        pose = theta[:, 3:75].contiguous()
        shape = theta[:, 75:].contiguous()
        verts, j3d, Rs = self.smpl(beta = shape, theta = pose, get_skin = True)
        j2d = util.batch_orth_proj(j3d, cam)

        return (theta, verts, j2d, j3d, Rs)







