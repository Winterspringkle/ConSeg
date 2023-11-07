import sys
import math
import numpy as np
import einops
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.utils.checkpoint as checkpoint
from torch.nn import functional as F
class cross_module(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(cross_module, self).__init__()
        self.conv_pt=Conv_block(in_ch,in_ch,1)
        self.conv_ct = Conv_block(in_ch, in_ch, 1)
        self.conv_fusion = nn.Sequential(
            Conv_block(in_ch, out_ch, 3),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(),
            nn.Sigmoid())

        self.trans=SwinTrans_stage_block(embed_dim=in_ch,
                              num_layers=2,
                              num_heads=in_ch // 4,
                              window_size=[5, 5, 5],
                              cross_atten=False,
                              use_checkpoint=False)
    def forward(self, pt,ct):
        pt=self.conv_pt(pt)
        ct = self.conv_ct(ct)
        # fusion=torch.cat((pt,ct),dim=1)
        # print(fusion.shape)
        fusion=self.trans(pt,ct)
        fusion = self.conv_fusion(fusion)
        # print(fusion.shape)

        return fusion

class fusion_module(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(fusion_module, self).__init__()
        self.conv_pt = Conv_block(in_ch, in_ch, 1)
        self.conv_ct = Conv_block(in_ch, in_ch, 1)
        self.conv_fusion = nn.Sequential(
            Conv_block(2 * in_ch, out_ch, 3),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(),
            nn.Sigmoid())

    def forward(self, pt,ct):
        pt=self.conv_pt(pt)#B,C,L,H,W
        ct = self.conv_ct(ct)#B,C,L,H,W
        # print(pt.shape,ct.shape)
        pt_emb = torch.flatten(pt,2)#B,C,LHW
        ct_emb = torch.flatten(ct, 2)#B,C,LHW
        pet_emb_t = pt_emb.transpose(1, 2)#B,LHW,C
        ct_emb_t=ct_emb.transpose(1,2)#B,LHW,C
        # print(pt_emb.shape,ct_emb.shape)
        # print(ct_emb.transpose(1,2).shape)
        huge_matrix_pet = pet_emb_t.matmul(pt_emb)#B,LHW,C*B,C,LHW=B,LHW,LHW
        huge_matrix_ct=ct_emb_t.matmul(pt_emb)#B,LHW,C*B,C,LHW=B,LHW,LHW
        pet_cal_emb=torch.max(huge_matrix_pet,2)[0]#B,LHW
        ct_cal_emb = torch.max(huge_matrix_ct, 2)[0]#B,LHW
        #print(pet_cal_emb.shape, ct_cal_emb.shape)
        cal_map=(pet_cal_emb.reshape(pt.shape[0],7,7,7)+ct_cal_emb.reshape(pt.shape[0],7,7,7))/2
        # print(cal_map.shape)
        return pet_cal_emb,ct_cal_emb,cal_map
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False, rate=0.1):
        super(outconv, self).__init__()
        self.dropout = dropout
        if dropout:
            print('dropout', rate)
            self.dp = nn.Dropout3d(rate)
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        if self.dropout:
            x = self.dp(x)
        x = self.conv(x)
        return x

class process_module_bn(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(process_module_bn, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(1,1,1)),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU()
        )
        self.atten0 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(1, 1,1)),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(),
            nn.Sigmoid()
        )
        self.atten1 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(1,1, 1)),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(),
            nn.Sigmoid()
        )
        self.norm=nn.InstanceNorm3d(out_ch)

    def forward(self, x_0):
        x=self.conv(x_0)
        #BN
        #spatial+channel atten
        #x=x_w*bn(x)+x_b
        x_w = x*self.atten0(x)
        x_b = x*self.atten1(x)
        #x_1=self.norm(x_0)
        x=x_0*x_w+x_b
        return x

class ConSeg(nn.Module):

    def __init__(self, channel_num=16, use_checkpoint=False):
        super().__init__()

        self.Merging_encoder = Merging_encoder(channel_num=channel_num // 2,
                                               use_checkpoint=use_checkpoint)
        self.Diverging_decoder_pre = Diverging_decoder_pre(in_channels=channel_num,
                                                   channel_num=channel_num // 2,
                                                   use_checkpoint=use_checkpoint)
        self.Diverging_decoder = Diverging_decoder(in_channels=channel_num,
                                                   channel_num=channel_num // 2,
                                                   use_checkpoint=use_checkpoint)


    def forward(self, PET, CT):
        x_1, x_2, x_3, x_4, x_5,pet_emb,ct_emb,cal_map = self.Merging_encoder(PET, CT)
        a_map = self.Diverging_decoder_pre(x_1, x_2, x_3, x_4, x_5)
        Seg_PT_pred,cat_pred,x1,x2,x3,x4= self.Diverging_decoder(x_1, x_2, x_3, x_4, x_5,a_map)
        return Seg_PT_pred,a_map,cat_pred,x1,x2,x3,x4,pet_emb,ct_emb


# --------------------------------------------------------------------------------------
class Merging_encoder(nn.Module):

    def __init__(self,
                 channel_num: int,
                 use_checkpoint: bool = False):
        super().__init__()

        self.PET_encoder_1 = Residual_block(in_channels=1,
                                            out_channels=channel_num,
                                            conv_num=2,
                                            use_checkpoint=use_checkpoint)
        self.CT_encoder_1 = Residual_block(in_channels=1,
                                           out_channels=channel_num,
                                           conv_num=2,
                                           use_checkpoint=use_checkpoint)
        self.PET_encoder_2 = Self_Hybrid_block(in_channels=channel_num,
                                               embed_dim=channel_num * 2,
                                               conv_num=3,
                                               num_layers=2,
                                               num_heads=channel_num // 4,
                                               window_size=[5, 5, 5],
                                               use_checkpoint=use_checkpoint)
        self.CT_encoder_2 = Self_Hybrid_block(in_channels=channel_num,
                                              embed_dim=channel_num * 2,
                                              conv_num=3,
                                              num_layers=2,
                                              num_heads=channel_num // 4,
                                              window_size=[5, 5, 5],
                                              use_checkpoint=use_checkpoint)
        self.PET_encoder_3 = Cross_Hybrid_block(in_channels=channel_num * 2,
                                                embed_dim=channel_num * 4,
                                                conv_num=3,
                                                num_layers=2,
                                                num_heads=channel_num // 2,
                                                window_size=[5, 5, 5],
                                                use_checkpoint=use_checkpoint)
        self.CT_encoder_3 = Cross_Hybrid_block(in_channels=channel_num * 2,
                                               embed_dim=channel_num * 4,
                                               conv_num=3,
                                               num_layers=2,
                                               num_heads=channel_num // 2,
                                               window_size=[5, 5, 5],
                                               use_checkpoint=use_checkpoint)
        self.PET_encoder_4 = Cross_Hybrid_block(in_channels=channel_num * 4,
                                                embed_dim=channel_num * 8,
                                                conv_num=4,
                                                num_layers=2,
                                                num_heads=channel_num,
                                                window_size=[5, 5, 5],
                                                use_checkpoint=use_checkpoint)
        self.CT_encoder_4 = Cross_Hybrid_block(in_channels=channel_num * 4,
                                               embed_dim=channel_num * 8,
                                               conv_num=4,
                                               num_layers=2,
                                               num_heads=channel_num,
                                               window_size=[5, 5, 5],
                                               use_checkpoint=use_checkpoint)
        self.PET_encoder_5 = Cross_Hybrid_block(in_channels=channel_num * 8,
                                                embed_dim=channel_num * 16,
                                                conv_num=4,
                                                num_layers=2,
                                                num_heads=channel_num * 2,
                                                window_size=[5, 5, 5],
                                                use_checkpoint=use_checkpoint)
        self.CT_encoder_5 = Cross_Hybrid_block(in_channels=channel_num * 8,
                                               embed_dim=channel_num * 16,
                                               conv_num=4,
                                               num_layers=2,
                                               num_heads=channel_num * 2,
                                               window_size=[5, 5, 5],
                                               use_checkpoint=use_checkpoint)
        self.norm_PT_1 = process_module_bn(channel_num,channel_num)
        self.norm_CT_1 = process_module_bn(channel_num,channel_num)
        self.norm_PT_2 = process_module_bn(channel_num*2,channel_num*2)
        self.norm_CT_2 = process_module_bn(channel_num*2,channel_num*2)
        self.norm_PT_3 = process_module_bn(channel_num*4,channel_num*4)
        self.norm_CT_3 = process_module_bn(channel_num*4,channel_num*4)
        self.norm_PT_4 = process_module_bn(channel_num*8,channel_num*8)
        self.norm_CT_4 = process_module_bn(channel_num*8,channel_num*8)
        self.norm_PT_5 = process_module_bn(channel_num*16,channel_num*16)
        self.norm_CT_5 = process_module_bn(channel_num*16,channel_num*16)
        # self.cross1=cross_module(channel_num,channel_num)
        # self.cross2=cross_module(channel_num,channel_num)
        self.cross3 = cross_module(channel_num*2, channel_num*4)
        self.cross4 = cross_module(channel_num*4, channel_num*8)
        self.cross5 = cross_module(channel_num*8, channel_num*16)
        self.fusion = fusion_module(channel_num * 16, channel_num * 16)

        self.downsample_1 = nn.MaxPool3d(2, stride=2)
        self.downsample_2 = nn.MaxPool3d(2, stride=2)
        self.downsample_3 = nn.MaxPool3d(2, stride=2)
        self.downsample_4 = nn.MaxPool3d(2, stride=2)

    def forward(self, PET, CT):
        # full scale
        x_PET_1 = self.norm_PT_1(self.PET_encoder_1(PET))
        x_CT_1 = self.norm_CT_1(self.CT_encoder_1(CT))

        # downsample 1/2 scale
        x_PET = self.downsample_1(x_PET_1)
        x_CT = self.downsample_1(x_CT_1)


        x_PET_2 = self.norm_PT_2(self.PET_encoder_2(x_PET))
        x_CT_2 = self.norm_CT_2(self.CT_encoder_2(x_CT))

        # downsample 1/4 scale
        x_PET = self.downsample_2(x_PET_2)
        x_CT = self.downsample_2(x_CT_2)
        x_cross3 = self.cross3(x_PET, x_CT)

        x_PET_3 =self.PET_encoder_3(x_PET, x_cross3)
        x_CT_3=self.CT_encoder_3(x_CT, x_cross3)
        x_PET_3 = self.norm_PT_3(x_PET_3)
        x_CT_3 = self.norm_CT_3(x_CT_3)

        # downsample 1/8 scale
        x_PET = self.downsample_3(x_PET_3)
        x_CT = self.downsample_3(x_CT_3)
        x_cross4 = self.cross4(x_PET, x_CT)

        x_PET_4=self.PET_encoder_4(x_PET, x_cross4)
        x_CT_4=self.CT_encoder_4(x_CT, x_cross4)
        x_PET_4 = self.norm_PT_4(x_PET_4)
        x_CT_4 = self.norm_PT_4(x_CT_4)

        # downsample 1/16 scale
        x_PET = self.downsample_4(x_PET_4)
        x_CT = self.downsample_4(x_CT_4)
        x_cross5= self.cross5(x_PET, x_CT)

        x_PET_5 = self.norm_PT_5(self.PET_encoder_5(x_PET, x_cross5))
        x_CT_5 = self.norm_PT_5(self.CT_encoder_5(x_CT, x_cross5))
        # print(x_PET_5.shape,x_CT_5.shape)
        pet_emb, ct_emb,cal_map=self.fusion(x_PET_5,x_CT_5)
        # concatenate
        x_1 = torch.cat([x_PET_1, x_CT_1], dim=1)
        x_2 = torch.cat([x_PET_2, x_CT_2], dim=1)
        x_3 = torch.cat([x_PET_3, x_CT_3], dim=1)
        x_4 = torch.cat([x_PET_4, x_CT_4], dim=1)
        x_5 = torch.cat([x_PET_5, x_CT_5], dim=1)
        # print(cal_map.shape, x_5.shape)
        cal_map=F.sigmoid(cal_map.unsqueeze(1))
        fusion_feature=torch.cat([x_5*cal_map,x_5],dim=1)
        # print("fusion_feature",fusion_feature.shape)
        #B,L,H,W*B,C,L,H,W
        return x_1, x_2, x_3, x_4, x_5,pet_emb,ct_emb,cal_map

class Diverging_decoder_pre(nn.Module):

    def __init__(self,
                 in_channels: int,
                 channel_num: int,
                 use_checkpoint: bool = False):
        super().__init__()

        self.decoder_PT_5 = Residual_block(in_channels=in_channels * 16,
                                           out_channels=channel_num * 16,
                                           conv_num=4,
                                           use_checkpoint=use_checkpoint)
        self.decoder_PT_6 = Residual_block(in_channels=channel_num * 16 + in_channels * 8,
                                           out_channels=channel_num * 8,
                                           conv_num=4,
                                           use_checkpoint=use_checkpoint)
        self.decoder_PT_7 = Residual_block(in_channels=channel_num * 8 + in_channels * 4,
                                           out_channels=channel_num * 4,
                                           conv_num=3,
                                           use_checkpoint=use_checkpoint)
        self.decoder_PT_8 = Residual_block(in_channels=channel_num * 4 + in_channels * 2,
                                           out_channels=channel_num * 2,
                                           conv_num=3,
                                           use_checkpoint=use_checkpoint)
        self.decoder_PT_9 = Residual_block(in_channels=channel_num * 2 + in_channels,
                                           out_channels=channel_num,
                                           conv_num=2,
                                           use_checkpoint=use_checkpoint)


        self.upsample_6 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_7 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_8 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_9 = nn.Upsample(scale_factor=2, mode='nearest')

        self.atten_gate_6 = Region_Atten_block(in_channels * 8, channel_num * 16, channel_num * 8)
        self.atten_gate_7 = Region_Atten_block(in_channels * 4, channel_num * 8, channel_num * 4)
        self.atten_gate_8 = Region_Atten_block(in_channels * 2, channel_num * 4, channel_num * 2)
        self.atten_gate_9 = Region_Atten_block(in_channels, channel_num * 2, channel_num)

        self.Conv_PT = nn.Conv3d(channel_num, 1, kernel_size=1, stride=1)
        self.Conv_MLN = nn.Conv3d(channel_num, 1, kernel_size=1, stride=1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x_1, x_2, x_3, x_4, x_5):
        # 1/16 scale
        #B,L,H,W         B,C,L,H,W
        # x_5=x_5.unsqueeze(1)
        # x_5=x_5.repeat([1,256,1,1,1])
        x_PT_5 = self.decoder_PT_5(x_5)

        x_PT_5 = self.upsample_6(x_PT_5)
        x_PT_5 = torch.cat([x_PT_5, x_4], dim=1)
        x_PT_6 = self.decoder_PT_6(x_PT_5)

        # upsample 1/4 scale
        x_PT_6 = self.upsample_7(x_PT_6)
        x_PT_6 = torch.cat([x_PT_6, x_3], dim=1)
        x_PT_7 = self.decoder_PT_7(x_PT_6)

        x_PT_7 = self.upsample_8(x_PT_7)
        x_PT_7 = torch.cat([x_PT_7, x_2], dim=1)
        x_PT_8 = self.decoder_PT_8(x_PT_7)

        x_PT_8 = self.upsample_9(x_PT_8)
        x_PT_8 = torch.cat([x_PT_8, x_1], dim=1)
        x_PT_9 = self.decoder_PT_9(x_PT_8)

        # Segmentation output
        Seg_PT_pred = F.sigmoid(self.Conv_PT(x_PT_9))

        return Seg_PT_pred

class Diverging_decoder(nn.Module):

    def __init__(self,
                 in_channels: int,
                 channel_num: int,
                 use_checkpoint: bool = False):
        super().__init__()

        self.decoder_PT_5 = Residual_block(in_channels=in_channels * 16,
                                           out_channels=channel_num * 16,
                                           conv_num=4,
                                           use_checkpoint=use_checkpoint)
        self.decoder_PT_6 = Residual_block(in_channels=channel_num * 16 + in_channels * 8,
                                           out_channels=channel_num * 8,
                                           conv_num=4,
                                           use_checkpoint=use_checkpoint)
        self.decoder_PT_7 = Residual_block(in_channels=channel_num * 8 + in_channels * 4,
                                           out_channels=channel_num * 4,
                                           conv_num=3,
                                           use_checkpoint=use_checkpoint)
        self.decoder_PT_8 = Residual_block(in_channels=channel_num * 4 + in_channels * 2,
                                           out_channels=channel_num * 2,
                                           conv_num=3,
                                           use_checkpoint=use_checkpoint)
        self.decoder_PT_9 = Residual_block(in_channels=channel_num * 2 + in_channels,
                                           out_channels=channel_num,
                                           conv_num=2,
                                           use_checkpoint=use_checkpoint)


        self.upsample_6 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_7 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_8 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_9 = nn.Upsample(scale_factor=2, mode='nearest')

        self.atten_gate_6 = Region_Atten_block(in_channels * 8, channel_num * 16, channel_num * 8)
        self.atten_gate_7 = Region_Atten_block(in_channels * 4, channel_num * 8, channel_num * 4)
        self.atten_gate_8 = Region_Atten_block(in_channels * 2, channel_num * 4, channel_num * 2)
        self.atten_gate_9 = Region_Atten_block(in_channels, channel_num * 2, channel_num)

        self.dsoutc5 = outconv(256, 1)
        self.dsoutc4 = outconv(128, 1)
        self.dsoutc3 = outconv(64, 1)
        self.dsoutc2 = outconv(32, 1)


        self.Conv_PT = nn.Conv3d(channel_num, 1, kernel_size=1, stride=1)
        self.Conv_MLN = nn.Conv3d(channel_num, 1, kernel_size=1, stride=1)
        self.Sigmoid = nn.Sigmoid()
        self.out=outconv(4, 1)

    def forward(self, x_1, x_2, x_3, x_4, x_5,x_origin):
        # 1/16 scale
        x_PT_5 = self.decoder_PT_5(x_5)*F.interpolate(x_origin, x_5.shape[2:], mode='nearest')

        x_PT_5 = self.upsample_6(x_PT_5)
        x_PT_5 = torch.cat([x_PT_5, x_4], dim=1)
        x_PT_6 = self.decoder_PT_6(x_PT_5)*F.interpolate(x_origin, x_4.shape[2:], mode='nearest')

        # upsample 1/4 scale
        x_PT_6 = self.upsample_7(x_PT_6)
        x_PT_6 = torch.cat([x_PT_6, x_3], dim=1)
        x_PT_7 = self.decoder_PT_7(x_PT_6)*F.interpolate(x_origin, x_3.shape[2:], mode='nearest')

        x_PT_7 = self.upsample_8(x_PT_7)
        x_PT_7 = torch.cat([x_PT_7, x_2], dim=1)
        x_PT_8 = self.decoder_PT_8(x_PT_7)*F.interpolate(x_origin, x_2.shape[2:], mode='nearest')

        x_PT_8 = self.upsample_9(x_PT_8)
        x_PT_8 = torch.cat([x_PT_8, x_1], dim=1)
        x_PT_9 = self.decoder_PT_9(x_PT_8)*F.interpolate(x_origin, x_1.shape[2:], mode='nearest')

        # Segmentation output
        Seg_PT_pred = self.Conv_PT(x_PT_9)


        x11 = F.interpolate(self.dsoutc5(x_PT_5), x_origin.shape[2:], mode='nearest')
        x22 = F.interpolate(self.dsoutc4(x_PT_6), x_origin.shape[2:], mode='nearest')
        x33 = F.interpolate(self.dsoutc3(x_PT_7), x_origin.shape[2:], mode='nearest')
        x44 = F.interpolate(self.dsoutc2(x_PT_8), x_origin.shape[2:], mode='nearest')
        x = torch.cat([x11, x22, x33, x44],dim=1)
        cat_pred=self.out(x)

        return Seg_PT_pred,cat_pred,x11,x22,x33,x44


# --------------------------------------------------------------------------------------

class Cross_Hybrid_block(nn.Module):

    def __init__(self,
                 in_channels: int,
                 embed_dim: int,
                 conv_num: int,
                 num_layers: int,
                 num_heads: int,
                 window_size: list,
                 use_checkpoint: bool = False):
        super().__init__()

        self.Residual_block = Atten_residual_block(in_channels=in_channels,
                                             out_channels=embed_dim,
                                             conv_num=conv_num,
                                             use_checkpoint=use_checkpoint)

        self.Proj_1 = nn.Conv3d(in_channels, embed_dim, kernel_size=1, stride=1)
        self.Proj_2 = nn.Conv3d(in_channels, embed_dim, kernel_size=1, stride=1)
        self.SwinTrans_block = SwinTrans_stage_block(embed_dim=embed_dim,
                                                     num_layers=num_layers,
                                                     num_heads=num_heads,
                                                     window_size=window_size,
                                                     cross_atten=True,
                                                     use_checkpoint=use_checkpoint)

    def forward(self, x_1, atten):
        x_conv = self.Residual_block(x_1,atten)
        # x_1 = self.Proj_1(x_1)
        # x_2 = self.Proj_2(x_2)
        # x_trans = self.SwinTrans_block(x_1, x_2)

        # x_out = x_conv + x_trans
        return x_conv


class Self_Hybrid_block(nn.Module):

    def __init__(self,
                 in_channels: int,
                 embed_dim: int,
                 conv_num: int,
                 num_layers: int,
                 num_heads: int,
                 window_size: list,
                 use_checkpoint: bool = False):
        super().__init__()

        self.Residual_block = Residual_block(in_channels=in_channels,
                                             out_channels=embed_dim,
                                             conv_num=conv_num,
                                             use_checkpoint=use_checkpoint)

        self.Proj = nn.Conv3d(in_channels, embed_dim, kernel_size=1, stride=1)
        self.SwinTrans_block = SwinTrans_stage_block(embed_dim=embed_dim,
                                                     num_layers=num_layers,
                                                     num_heads=num_heads,
                                                     window_size=window_size,
                                                     cross_atten=False,
                                                     use_checkpoint=use_checkpoint)

    def forward(self, x_in):
        x_conv = self.Residual_block(x_in)

        # x = self.Proj(x_in)
        # x_trans = self.SwinTrans_block(x)
        #
        # x_out = x_conv + x_trans
        return x_conv


class Region_Atten_block(nn.Module):

    def __init__(self, channel_x, channel_g, channel_num):
        super().__init__()

        self.Conv_g_1 = nn.Conv3d(channel_g, channel_num, kernel_size=1, stride=1, padding='same')
        self.BN_g_1 = nn.BatchNorm3d(channel_num)

        self.Conv_g_2 = nn.Conv3d(channel_g, channel_num, kernel_size=1, stride=1, padding='same')
        self.BN_g_2 = nn.BatchNorm3d(channel_num)

        self.Conv_x_1 = nn.Conv3d(channel_x, channel_num, kernel_size=1, stride=1, padding='same')
        self.BN_x_1 = nn.BatchNorm3d(channel_num)

        self.Conv_x_2 = nn.Conv3d(channel_x, channel_num, kernel_size=1, stride=1, padding='same')
        self.BN_x_2 = nn.BatchNorm3d(channel_num)

        self.Conv_relu = nn.Conv3d(channel_num * 2, 3, kernel_size=1, stride=1, padding='same')
        self.BN_relu = nn.BatchNorm3d(3)

        self.ReLU = nn.ReLU()
        self.Softmax = nn.Softmax(dim=1)

        self.AvgPool = nn.AvgPool3d(2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x_in, g_in_1, g_in_2):
        g = self.Conv_g_1(g_in_1)
        g_int_1 = self.BN_g_1(g)

        g = self.Conv_g_2(g_in_2)
        g_int_2 = self.BN_g_2(g)

        x = self.Conv_x_1(x_in)
        x = self.BN_x_1(x)
        x_int_1 = self.AvgPool(x)

        x = self.Conv_x_2(x_in)
        x = self.BN_x_2(x)
        x_int_2 = self.AvgPool(x)

        x_1 = torch.add(x_int_1, g_int_1)
        x_2 = torch.add(x_int_2, g_int_2)
        x = torch.cat([x_1, x_2], dim=1)
        x_relu = self.ReLU(x)

        x = self.Conv_relu(x_relu)
        x = self.BN_relu(x)
        x = self.Softmax(x)
        x_mask_1 = self.Upsample(x[:, 0:1, :, :, :])
        x_mask_2 = self.Upsample(x[:, 1:2, :, :, :])

        x_out_1 = torch.mul(x_in, x_mask_1)
        x_out_2 = torch.mul(x_in, x_mask_2)

        return x_out_1, x_out_2


class Atten_residual_block(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_num: int,
                 use_checkpoint: bool = False):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.Conv_res = Conv_block(in_channels, out_channels, 1)
        self.Conv = Conv_block(in_channels, out_channels, 3)

        self.Remain_Conv = nn.ModuleList()
        for i in range(conv_num - 1):
            self.Remain_Conv.append(Conv_block(out_channels, out_channels, 3))

    def Residual_forward(self, x_in,atten):
        x_res = self.Conv_res(x_in)
        x = self.Conv(x_in)
        # print("x",x.shape,"atten",atten.shape)
        x=x*atten
        diffY = x_res.size()[2] - x.size()[2]
        diffX = x_res.size()[3] - x.size()[3]
        diffZ = x_res.size()[4] - x.size()[4]
        x = F.pad(x, (diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2,
                      diffZ // 2, diffZ - diffZ // 2))
        x_out = torch.add(x, x_res)

        for Conv in self.Remain_Conv:
            x = Conv(x_out)
            diffY = x_out.size()[2] - x.size()[2]
            diffX = x_out.size()[3] - x.size()[3]
            diffZ = x_out.size()[4] - x.size()[4]
            x = F.pad(x, (diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2,
                          diffZ // 2, diffZ - diffZ // 2))
            x_out = torch.add(x, x_out)

        return x_out

    def forward(self, x_in,atten):

        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.Residual_forward, x_in)
        else:
            x_out = self.Residual_forward(x_in,atten)

        return x_out

class Residual_block(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_num: int,
                 use_checkpoint: bool = False):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.Conv_res = Conv_block(in_channels, out_channels, 1)
        self.Conv = Conv_block(in_channels, out_channels, 3)

        self.Remain_Conv = nn.ModuleList()
        for i in range(conv_num - 1):
            self.Remain_Conv.append(Conv_block(out_channels, out_channels, 3))

    def Residual_forward(self, x_in):
        x_res = self.Conv_res(x_in)
        x = self.Conv(x_in)
        diffY = x_res.size()[2] - x.size()[2]
        diffX = x_res.size()[3] - x.size()[3]
        diffZ = x_res.size()[4] - x.size()[4]
        x = F.pad(x, (diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2,
                      diffZ // 2, diffZ - diffZ // 2))
        x_out = torch.add(x, x_res)

        for Conv in self.Remain_Conv:
            x = Conv(x_out)
            diffY = x_out.size()[2] - x.size()[2]
            diffX = x_out.size()[3] - x.size()[3]
            diffZ = x_out.size()[4] - x.size()[4]
            x = F.pad(x, (diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2,
                          diffZ // 2, diffZ - diffZ // 2))
            x_out = torch.add(x, x_out)

        return x_out

    def forward(self, x_in):

        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.Residual_forward, x_in)
        else:
            x_out = self.Residual_forward(x_in)

        return x_out

class Conv_block(nn.Module):

    def __init__(self, in_channels, out_channels, kernels):
        super().__init__()

        self.Conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernels, stride=1)
        self.BN = nn.BatchNorm3d(out_channels)
        self.ReLU = nn.ReLU()

    def forward(self, x_in):
        # print(x_in.shape)
        x = self.Conv(x_in)
        x = self.BN(x)
        x_out = self.ReLU(x)

        return x_out


# --------------------------------------------------------------------------------------

class SwinTrans_stage_block(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_layers: int,
                 num_heads: int,
                 window_size: list,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
                 cross_atten: bool = False,
                 use_checkpoint: bool = False):
        super().__init__()

        self.window_size = window_size
        self.cross_atten = cross_atten
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            block = SwinTrans_Block(embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    window_size=self.window_size,
                                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    drop=drop,
                                    attn_drop=attn_drop,
                                    cross_atten=cross_atten,
                                    use_checkpoint=use_checkpoint)
            self.blocks.append(block)

    def forward(self, x_in,x_cross=None):

        b, c, d, h, w = x_in.shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]
        attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x_in.device)

        if self.cross_atten:
            x = einops.rearrange(x_in, 'b c d h w -> b d h w c')
            x_cross = einops.rearrange(x_cross, 'b c d h w -> b d h w c')
            for block in self.blocks:
                x = block(x, x_cross, attn_mask)
        else:
            x = einops.rearrange(x_in, 'b c d h w -> b d h w c')
            x_cross = einops.rearrange(x_cross, 'b c d h w -> b d h w c')
            for block in self.blocks:
                x = block(x,x_cross, mask_matrix=attn_mask)

        x = x.view(b, d, h, w, -1)
        x_out = einops.rearrange(x, 'b d h w c -> b c d h w')

        return x_out


class SwinTrans_Block(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 window_size: list,
                 shift_size: list,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
                 cross_atten: bool = False,
                 use_checkpoint: bool = False):
        super().__init__()

        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpoint = use_checkpoint
        self.cross_atten = cross_atten

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attn = MSA_block(embed_dim,
                              window_size=window_size,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              attn_drop=attn_drop,
                              proj_drop=drop)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP_block(hidden_size=embed_dim,
                             mlp_dim=int(embed_dim * mlp_ratio),
                             dropout_rate=drop)

    def forward_part1(self, x_in, x_cross, mask_matrix):

        x = self.norm1(x_in)

        b, d, h, w, c = x.shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, dp, hp, wp, _ = x.shape
        dims = [b, dp, hp, wp]

        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        #print("attn_mask",attn_mask.shape)
        x_cross = nnf.pad(x_cross, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        if any(i > 0 for i in shift_size):
            shifted_x_cross = torch.roll(x_cross, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                                         dims=(1, 2, 3))
        else:
            shifted_x_cross = x_cross
        x_cross_windows = window_partition(shifted_x_cross, window_size)

        x_windows = window_partition(shifted_x, window_size)
        #print("x_windows", x_windows.shape)
        #print("x_cross_windows", x_cross_windows.shape)
        attn_windows = self.attn(x_windows, x_cross_windows)

        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)

        if any(i > 0 for i in shift_size):
            x_out = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x_out = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x_out = x_out[:, :d, :h, :w, :].contiguous()

        return x_out

    def forward_part2(self, x_in):

        x = self.norm2(x_in)
        x_out = self.mlp(x)
        return x_out

    def forward(self, x_in, x_cross=None, mask_matrix=None):

        if self.use_checkpoint and x_in.requires_grad:
            x = x_in + checkpoint.checkpoint(self.forward_part1, x_in, x_cross, mask_matrix)
        else:
            x = x_in + self.forward_part1(x_in, x_cross, mask_matrix)

        if self.use_checkpoint and x.requires_grad:
            x_out = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x_out = x + self.forward_part2(x)

        return x_out


class MSA_block(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 window_size: list,
                 qkv_bias: bool = False,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5
        mesh_args = torch.meshgrid.__kwdefaults__

        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * self.window_size[0] - 1) *
                                                                     (2 * self.window_size[1] - 1) *
                                                                     (2 * self.window_size[2] - 1), num_heads))
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        if mesh_args is not None:
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
        else:
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.Softmax = nn.Softmax(dim=-1)
        self.fuse=nn.Sequential(
            nn.Conv2d(int(embed_dim/2), int(embed_dim/4), 1),
            nn.InstanceNorm2d(int(embed_dim/4)),
            nn.ReLU(),
            nn.Sigmoid())

    def forward(self, x_in,x_in2, mask=None):

        b, n, c = x_in.shape
        qkv = self.qkv(x_in).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v1 = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn1 = q @ k.transpose(-2, -1)

        b, n, c = x_in2.shape
        qkv = self.qkv(x_in2).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v2 = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn2 = q @ k.transpose(-2, -1)
        #print("atten1",attn1.shape,"atten2",attn2.shape)
        # print("v1", v1.shape, "v2", v2.shape)
        v=self.fuse(torch.cat((v1,v2),dim=1))


        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        #print("relative_position_bias",relative_position_bias.shape)
        attn = attn1+attn2 + relative_position_bias.unsqueeze(0)
        #print("attn", attn.shape)
        # if mask is not None:
        #     nw = mask.shape[0]
        #     attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
        #     attn = attn.view(-1, self.num_heads, n, n)
        attn = self.Softmax(attn)
        attn = self.attn_drop(attn).to(v.dtype)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x_out = self.proj_drop(x)

        return x_out




class MLP_block(nn.Module):

    def __init__(self, hidden_size: int, mlp_dim: int, dropout_rate: float = 0.0):
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)

        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)

        self.GELU = nn.GELU()

    def forward(self, x_in):
        x = self.linear1(x_in)
        x = self.GELU(x)
        x = self.drop1(x)

        x = self.linear2(x)
        x_out = self.drop2(x)

        return x_out


# --------------------------------------------------------------------------------------

def compute_mask(dims, window_size, shift_size, device):
    cnt = 0
    d, h, w = dims
    img_mask = torch.zeros((1, d, h, w, 1), device=device)
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


def window_partition(x_in, window_size):
    b, d, h, w, c = x_in.shape
    x = x_in.view(b,
                  d // window_size[0],
                  window_size[0],
                  h // window_size[1],
                  window_size[1],
                  w // window_size[2],
                  window_size[2],
                  c)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2],
                                                                  c)

    return windows


def window_reverse(windows, window_size, dims):
    b, d, h, w = dims
    x = windows.view(b,
                     d // window_size[0],
                     h // window_size[1],
                     w // window_size[2],
                     window_size[0],
                     window_size[1],
                     window_size[2],
                     -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

