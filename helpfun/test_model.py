#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


from yolox.models.network_blocks import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck
# from yolox.models.darknet import
from yolox.models import YOLOFPN  , YOLOXHead
from  yolox.models import YOLOPAFPN
from yolox.models.darknet import Darknet, CSPDarknet

from yolox.models.myyolox import MYYOLOX
class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.conv(x)

    def fuseforward(self, x):
        return self.act(self.conv(x))


# class Darknet(nn.Module):
#     # number of blocks from dark2 to dark5.
#     depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}
#
#     def __init__(
#         self,
#         depth,
#         in_channels=3,
#         stem_out_channels=32,
#         out_features=("dark3", "dark4", "dark5"),
#     ):
#         """
#         Args:
#             depth (int): depth of darknet used in model, usually use [21, 53] for this param.
#             in_channels (int): number of input channels, for example, use 3 for RGB image.
#             stem_out_channels (int): number of output chanels of darknet stem.
#                 It decides channels of darknet layer2 to layer5.
#             out_features (Tuple[str]): desired output layer name.
#         """
#         super().__init__()
#         assert out_features, "please provide output features of Darknet"
#         self.out_features = out_features
#         self.stem = nn.Sequential(
#             BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),
#             *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),
#         )
#         in_channels = stem_out_channels * 2  # 64
#
#         num_blocks = Darknet.depth2blocks[depth]
#         # create darknet with `stem_out_channels` and `num_blocks` layers.
#         # to make model structure more clear, we don't use `for` statement in python.
#         self.dark2 = nn.Sequential(
#             *self.make_group_layer(in_channels, num_blocks[0], stride=2)
#         )
#         in_channels *= 2  # 128
#         self.dark3 = nn.Sequential(
#             *self.make_group_layer(in_channels, num_blocks[1], stride=2)
#         )
#         in_channels *= 2  # 256
#         self.dark4 = nn.Sequential(
#             *self.make_group_layer(in_channels, num_blocks[2], stride=2)
#         )
#         in_channels *= 2  # 512
#
#         self.dark5 = nn.Sequential(
#             *self.make_group_layer(in_channels, num_blocks[3], stride=2),
#             *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
#         )
#
#     def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
#         "starts with conv layer then has `num_blocks` `ResLayer`"
#         return [
#             BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),
#             *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
#         ]
#
#     def make_spp_block(self, filters_list, in_filters):
#         m = nn.Sequential(
#             *[
#                 BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
#                 BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
#                 SPPBottleneck(
#                     in_channels=filters_list[1],
#                     out_channels=filters_list[0],
#                     activation="lrelu",
#                 ),
#                 BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
#                 BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
#             ]
#         )
#         return m
#
#     def forward(self, x):
#         outputs = {}
#         x = self.stem(x)
#         outputs["stem"] = x
#         x = self.dark2(x)
#         outputs["dark2"] = x
#         x = self.dark3(x)
#         outputs["dark3"] = x
#         x = self.dark4(x)
#         outputs["dark4"] = x
#         x = self.dark5(x)
#         outputs["dark5"] = x
#         return {k: v for k, v in outputs.items() if k in self.out_features}


class test(nn.Module):
    def __init__(
        self,
        depth=53,
        in_features=["dark3", "dark4", "dark5"],
    ):
        super().__init__()
        self.backbone = Darknet(depth)
        self.in_features = in_features

        # out 1
        self.out1_cbl = self._make_cbl(512, 256, 1)
        self.out1 = self._make_embedding([256, 512], 512 + 256)

        # out 2
        self.out2_cbl = self._make_cbl(256, 128, 1)
        self.out2 = self._make_embedding([128, 256], 256 + 128)

        # upsample
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def _make_cbl(self, _in, _out, ks):
        return BaseConv(_in, _out, ks, stride=1, act="lrelu")

    def _make_embedding(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                self._make_cbl(in_filters, filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
            ]
        )
        return m
    def forward(self, inputs):
        """
        Args:
            inputs (Tensor): input image.

        Returns:
            Tuple[Tensor]: FPN output features..
        """
        #  backbone
        out_features = self.backbone(inputs)
        x2, x1, x0 = [out_features[f] for f in self.in_features]

        #  yolo branch 1
        x1_in = self.out1_cbl(x0)
        x1_in = self.upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out_dark4 = self.out1(x1_in)

        #  yolo branch 2
        x2_in = self.out2_cbl(out_dark4)
        x2_in = self.upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out_dark3 = self.out2(x2_in)

        outputs = (out_dark3, out_dark4, x0)
        return outputs


class YOLOFPN(nn.Module):
    """
    YOLOFPN module. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=53,
        in_features=["dark3", "dark4", "dark5"],
    ):
        super().__init__()

        self.backbone = Darknet(depth)
        self.in_features = in_features

        # out 1
        self.out1_cbl = self._make_cbl(512, 256, 1)
        self.out1 = self._make_embedding([256, 512], 512 + 256)

        # out 2
        self.out2_cbl = self._make_cbl(256, 128, 1)
        self.out2 = self._make_embedding([128, 256], 256 + 128)

        # upsample
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def _make_cbl(self, _in, _out, ks):
        return BaseConv(_in, _out, ks, stride=1, act="lrelu")

    def _make_embedding(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                self._make_cbl(in_filters, filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
            ]
        )
        return m

    def load_pretrained_model(self, filename="./weights/darknet53.mix.pth"):
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        print("loading pretrained weights...")
        self.backbone.load_state_dict(state_dict)

    def forward(self, inputs):
        """
        Args:
            inputs (Tensor): input image.

        Returns:
            Tuple[Tensor]: FPN output features..
        """
        #  backbone
        out_features = self.backbone(inputs)
        x2, x1, x0 = [out_features[f] for f in self.in_features]

        #  yolo branch 1
        x1_in = self.out1_cbl(x0)
        x1_in = self.upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out_dark4 = self.out1(x1_in)

        #  yolo branch 2
        x2_in = self.out2_cbl(out_dark4)
        x2_in = self.upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out_dark3 = self.out2(x2_in)

        outputs = (out_dark3, out_dark4, x0)
        return outputs


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels *4 , out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        # return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        return self.conv(x)


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class TEST(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        # self.act = get_activation(act, inplace=True)

    def forward(self, x):
        # return self.act(self.bn(self.conv(x)))
        # return self.conv(x)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        # x = self.act(x)
        return x

    # def fuseforward(self, x):
    #     return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        # module_list = [
        #     Bottleneck(
        #         hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
        #     )
        #     for _ in range(n)
        # ]
        # self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        # x_2 = self.conv2(x)
        # x_1 = self.m(x_1)
        # x = torch.cat((x_1, x_2), dim=1)
        # return self.conv3(x)
        return x_1


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)



class CSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}

# class YOLOX(nn.Module):
#     """
#     YOLOX model module. The module list is defined by create_yolov3_modules function.
#     The network returns loss values from three YOLO layers during training
#     and detection results during test.
#     """
#
#     def __init__(self, backbone=None, head=None):
#         super().__init__()
#         if backbone is None:
#             backbone = YOLOPAFPN()
#         if head is None:
#             head = YOLOXHead(80)
#
#         self.backbone = backbone
#         self.head = head
#
#     def forward(self, x, targets=None):
#         # fpn output content features of [dark3, dark4, dark5]
#         fpn_outs = self.backbone(x)
#
#         if self.training:
#             assert targets is not None
#             loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
#                 fpn_outs, targets, x
#             )
#             outputs = {
#                 "total_loss": loss,
#                 "iou_loss": iou_loss,
#                 "l1_loss": l1_loss,
#                 "conf_loss": conf_loss,
#                 "cls_loss": cls_loss,
#                 "num_fg": num_fg,
#             }
#         else:
#             outputs = self.head(fpn_outs)
#
#         return outputs

import logging
import torch.nn
LOGGER = logging.getLogger(__name__)
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv
class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        # self.conv1= nn.Conv2d(3,6,5)
        # self.bn = nn.BatchNorm2d(6)
        self.conv1 = BaseConv()

    def forward(self, x):
        x = self.conv1(x)
        x=  self.bn(x)
        return x

def fuse(model):  # fuse model Conv2d() + BatchNorm2d() layers
    LOGGER.info('Fusing layers... ')
    for m in model.modules():
        if isinstance(m, (BaseConv, DWConv)) and hasattr(m, 'bn'):
            print("merge")
            m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
            delattr(m, 'bn')  # remove batchnorm
            m.forward = m.fuseforward  # update forward
    # self.info()
    return model


class SiLU(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
        # return x * F.leaky_relu(x)  # fix the activate funciton


class Hardswish(nn.Module):  # export-friendly version of nn.Hardswish()
    @staticmethod
    def forward(x):
        # return x * F.hardsigmoid(x)  # for torchscript and CoreML
        return x * F.hardtanh(x + 3, 0., 6.) / 6.  # for torchscript, CoreML and ONNX

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


if __name__ == '__main__':
    # model = Darknet(53)
    # model = CSPDarknet(1.0,1.0)
    # model = test(53)
    # model = MYYOLOX()
    # model = Focus()
    wid_mul =1.0
    dep_mul=1.0
    base_channels = int(wid_mul * 64)  # 64
    base_depth = max(round(dep_mul * 3), 1)  # 3

    # stem
    model = Focus(3, base_channels, ksize=3)
    initialize_weights(model)
    # model = MYYOLOX()
    # wid_mul=1.0
    # dep_mul=1.0
    # depthwise = False
    # act = "silu"
    #
    # base_channels = int(wid_mul * 64)  # 64
    # base_depth = max(round(dep_mul * 3), 1)  # 3
    #
    # model = CSPLayer(
    #             base_channels * 2,
    #             base_channels * 2,
    #             n=base_depth,
    #             depthwise=depthwise,
    #             act=act,
    #         )


    # base_channels = int(wid_mul * 64)
    # in_channels = base_channels * 2
    # out_channels =  base_channels * 2
    # n = 1,
    # shortcut = True
    # expansion = 0.5
    # depthwise = False
    # act = "silu"
    #
    # hidden_channels = int(out_channels * expansion)  # hidden channels
    # model = TEST(in_channels, hidden_channels, 1, stride=1, act=act)
    # [x2, x1, x0] = features

    # model = Focus()
    # model = fuse(model)

    dummy_input = torch.randn(1, 3, 640, 640)

    # model = fuse(model)
    model.eval()
    inplace = False
    # Compatibility updates
    # for m in model.modules():
    #     if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
    #         m.inplace = inplace  # pytorch 1.7.0 compatibility
    #     elif type(m) is Conv:
    #         m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    # for k, m in model.named_modules():
    #     if isinstance(m, Conv):  # assign export-friendly activations
    #         if isinstance(m.act, nn.Hardswish):
    #             m.act = Hardswish()
    #         elif isinstance(m.act, nn.SiLU):
    #             m.act = SiLU()
        # elif isinstance(m, Detect):
        #     m.inplace = inplace
        #     m.onnx_dynamic = dynamic

    # if len(model) == 1:
    #     model=  model[-1]  # return model
    train = False
    dynamic = False

    # model.train()
    f = "test.onnx"
    for _ in range(2):
        y = model(dummy_input)  # dry runs

    # model.eval()
    # torch.onnx._export(model, dummy_input, f, verbose=True, opset_version=11,
    #                       training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
    #                       # training=torch.onnx.TrainingMode.TRAINING,
    #                       do_constant_folding=False,
    #                       input_names=['images'],
    #                       # output_names=['output','output1','output2'],
    #                       output_names=['output'],
    #                       dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
    #                                     'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
    #                                     } if dynamic else None)
    torch.onnx.export(model, dummy_input, f, verbose=True, opset_version=12,
                      training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
                      # training=torch.onnx.TrainingMode.TRAINING,
                      do_constant_folding=True,
                      input_names=['images'],
                      output_names=['output'],
                      dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                                    'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
                                    } if dynamic else None)
    import onnx
    model = onnx.load(f)
    onnx.checker.check_model(model)