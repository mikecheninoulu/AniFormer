
from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange, repeat

class PoseFeature(nn.Module):
    def __init__(self, num_points = 6890):
        super(PoseFeature, self).__init__()

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.norm1 = torch.nn.LayerNorm(num_points)
        self.norm2 = torch.nn.LayerNorm(num_points)
        self.norm3 = torch.nn.LayerNorm(num_points)

    def forward(self, x):
        # print('check1')
        # print(x.shape)
        x = F.relu(self.norm1(self.conv1(x)))
        # print('check2')
        # print(x.shape)
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))

        return x


class SPAdaIN(nn.Module):
    def __init__(self,norm,input_nc,planes):
        super(SPAdaIN,self).__init__()
        self.conv_weight = nn.Conv1d(input_nc, planes, 1)
        self.conv_bias = nn.Conv1d(input_nc, planes, 1)
        self.norm = norm(planes)

    def forward(self,x,addition):

        x = self.norm(x)
        weight = self.conv_weight(addition)
        bias = self.conv_bias(addition)
        out =  weight * x + bias

        return out

class SPAdaINResBlock(nn.Module):
    def __init__(self,input_nc,planes,norm=nn.InstanceNorm1d,conv_kernel_size=1,padding=0):
        super(SPAdaINResBlock,self).__init__()
        self.spadain1 = SPAdaIN(norm=norm,input_nc=input_nc,planes=planes)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(planes, planes, kernel_size=conv_kernel_size, stride=1, padding=padding)
        self.spadain2 = SPAdaIN(norm=norm,input_nc=input_nc,planes=planes)
        self.conv2 = nn.Conv1d(planes,planes,kernel_size=conv_kernel_size, stride=1, padding=padding)
        self.spadain_res = SPAdaIN(norm=norm,input_nc=input_nc,planes=planes)
        self.conv_res=nn.Conv1d(planes,planes,kernel_size=conv_kernel_size, stride=1, padding=padding)

    def forward(self,x,addition):

        out = self.spadain1(x,addition)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.spadain2(out,addition)
        out = self.relu(out)
        out = self.conv2(out)

        residual = x
        residual = self.spadain_res(residual,addition)
        residual = self.relu(residual)
        residual = self.conv_res(residual)

        out = out + residual

        return  out

class AttentionBlock(nn.Module):
    def __init__(self,planes,norm=nn.InstanceNorm1d,conv_kernel_size=1,padding=0):
        super(AttentionBlock,self).__init__()

        self.query_conv = nn.Conv1d(planes, planes, kernel_size=conv_kernel_size, stride=1, padding=padding)
        self.key_conv = nn.Conv1d(planes, planes, kernel_size=conv_kernel_size, stride=1, padding=padding)
        self.value_conv = nn.Conv1d(planes, planes, kernel_size=conv_kernel_size, stride=1, padding=padding)

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, pose_f):
        m_batchsize, C, p_len = pose_f.size()
        #[8, 1024, 6890]
        proj_query = self.query_conv(pose_f).permute(0, 2, 1)
        proj_key = self.key_conv(pose_f)

        energy = torch.bmm(proj_query, proj_key)
        #print(energy.shape)
        attention = self.softmax(energy)

        proj_value = self.value_conv(pose_f)
        #print(attention.permute(0, 2, 1).shape)
        value_attention = torch.bmm(proj_value, attention.permute(0, 2, 1))

        out = pose_f + self.gamma*value_attention  # connection

        return  out

class TransformerDecoder(nn.Module):
    def __init__(self, num_points = 6890, bottleneck_size = 1024):
        self.bottleneck_size = bottleneck_size
        super(TransformerDecoder, self).__init__()

        self.att_block1 = AttentionBlock(planes=self.bottleneck_size)
        self.att_block2 = AttentionBlock(planes=self.bottleneck_size//2)
        self.att_block3 = AttentionBlock(planes=self.bottleneck_size//2)
        self.att_block4 = AttentionBlock(planes=self.bottleneck_size//4)


        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//2, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1)


        self.conv5 = torch.nn.Conv1d(self.bottleneck_size//4, 3, 1)

        self.spadain_block1 = SPAdaINResBlock(input_nc=3 ,planes=self.bottleneck_size)
        self.spadain_block2 = SPAdaINResBlock(input_nc=3 ,planes=self.bottleneck_size//2)
        self.spadain_block3 = SPAdaINResBlock(input_nc=3 ,planes=self.bottleneck_size//2)
        self.spadain_block4 = SPAdaINResBlock(input_nc=3 ,planes=self.bottleneck_size//4)


        self.norm1 = torch.nn.LayerNorm(num_points)
        self.norm2 = torch.nn.LayerNorm(num_points)
        self.norm3 = torch.nn.LayerNorm(num_points)
        self.th = nn.Tanh()

        nn.LayerNorm


    def forward(self, x1_f, addition):
        # [8,9, 1024, 6890]

        # [8,1, 1024, 6890]
        #x1_f
        #[8 * 9, 1024, 6890]
        #x2_f
        #[8, 1024, 6890]

        # x2_f = x2_f.repeat(x1_f.shape[0]//x2_f.shape[0], 1, 1)
        # print(x2_f.shape)
        #[8*9, 1024, 6890]
        # addition = addition.repeat(x1_f.shape[0]//addition.shape[0], 1, 1)
        # print(x2_f.shape)
        #[8*9, 3, 6890]
        #1024 -1024

        x1_f = self.conv1(x1_f)
        y = self.att_block1(x1_f)
        x1_f = self.spadain_block1(y,addition)

        #1024 -512
        x1_f = self.conv2(x1_f)
        y = self.att_block2(x1_f)
        x1_f = self.spadain_block2(y,addition)

        #512 -512
        x1_f = self.conv3(x1_f)
        y = self.att_block3(x1_f)
        x1_f = self.spadain_block3(y,addition)

        #512 -256
        x1_f = self.conv4(x1_f)
        y = self.att_block4(x1_f)
        x = self.spadain_block4(y,addition)

        #256-3
        #
        # x = self.nn.LayerNorm(x)
        # ##### x size [b, f, emb_dim], then take weighted mean on frame dimension, we only predict 3D pose of the center frame
        # x = self.weighted_mean(x)
        # x = x.view(b, 1, -1)
        #256-3
        x = 2*self.th(self.conv5(x))
        return x


class Transformer3D(nn.Module):
    def __init__(self, num_points = 6890, bottleneck_size = 1024, video_len = 1):
        super(Transformer3D, self).__init__()
        self.num_points = num_points
        # out_dim = num_joints * 3 *video_len
        self.bottleneck_size = bottleneck_size
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, video_len, bottleneck_size,num_points))
        self.encoder = PoseFeature(num_points = num_points)
        self.decoder = TransformerDecoder(num_points = num_points, bottleneck_size = self.bottleneck_size)

    def forward(self, source_sq, target_ms):
        #source_sq
        #[8, 3, 6890]
        #f_target = self.encoder(target_ms)
        # print(f_target.shape)
        #[8, 1024, 6890]

        b, f, c, p = source_sq.shape  ##### b is batch size, f is number of frames, p is number of vertices
        source_sq = rearrange(source_sq, 'b f c p  -> (b f) c p', )
        # print(source_sq.shape)
        f_sq = self.encoder(source_sq)
        # print(f_sq.shape)
        #[8 * 9, 1024, 6890]
        f_sq = f_sq.view(b, f, -1, p)
        # print(f_sq.shape)
        #[8 * 9, 1024, 6890]
        f_sq += self.Temporal_pos_embed
        # f_sq = self.pos_drop(f_sq)
        # out_sq = torch.zeros([b,f,c,p]).cuda()
        out_sq = torch.zeros([b,f,c,p], dtype=torch.float64).cuda()
        for frame in range(f):
            # print(f_sq[:,frame,:,:].shape)
            # print(out_sq.shape)
            out_sq[:,frame,:,:] = self.decoder(f_sq[:,frame,:,:], target_ms)

        # out =self.decoder(f_sq , f_target, target_ms)
        #[8 * 9, 1024, 6890]
        out = out_sq.view(b, f, -1, p)
        #[8, 9, 1024, 6890]

        return out.transpose(3,2)


