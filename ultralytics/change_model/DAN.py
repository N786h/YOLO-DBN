import torch
import torch.nn as nn
from ultralytics.nn.modules.block import C3k, Bottleneck, C2f
from ultralytics.nn.modules import Bottleneck

class DiverseAttentionNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=None, dilation=1, groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False, single_init=False):
        super(DiverseAttentionNet, self).__init__()
        self.deploy = deploy

        # Using optimized activation function
        self.nonlinear = nn.ReLU(inplace=True)

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        # Automatically calculate padding if not provided
        if padding is None:
            padding = kernel_size // 2

        if deploy:
            # Deployable version with single Conv2D
            self.dbb_reparam = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                         dilation=dilation, groups=groups, bias=True)
        else:
            # Original 3-branch structure
            # Branch 1: Standard Conv + BN + ReLU
            self.dbb_origin = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels)
            )

            # Branch 2: Avg Pool + Conv + BN
            self.dbb_avg = nn.Sequential()
            if groups < out_channels:
                self.dbb_avg.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
                self.dbb_avg.add_module('bn', nn.BatchNorm2d(out_channels))
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))
            else:
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))
            self.dbb_avg.add_module('avgbn', nn.BatchNorm2d(out_channels))

            # Branch 3: 1x1 -> KxK Conv sequence
            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels

            self.dbb_1x1_kxk = nn.Sequential(
                nn.Conv2d(in_channels, internal_channels_1x1_3x3, kernel_size=1, stride=1, padding=0, groups=groups, bias=False),
                nn.BatchNorm2d(internal_channels_1x1_3x3),
                nn.ReLU(inplace=True),
                nn.Conv2d(internal_channels_1x1_3x3, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        if single_init:
            self.single_init()

    def forward(self, inputs):
        if self.deploy:
            return self.nonlinear(self.dbb_reparam(inputs))

        out = self.dbb_origin(inputs)

        if hasattr(self, 'dbb_avg'):
            out += self.dbb_avg(inputs)
        if hasattr(self, 'dbb_1x1_kxk'):
            out += self.dbb_1x1_kxk(inputs)

        return self.nonlinear(out)

    def init_gamma(self, gamma_value):
        # Initialize BatchNorm weights
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin[1].weight, gamma_value)
        if hasattr(self, "dbb_avg"):
            torch.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):
            torch.nn.init.constant_(self.dbb_1x1_kxk[4].weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin[1].weight, 1.0)

    def switch_to_deploy(self):
        if hasattr(self, 'dbb_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dbb_reparam = nn.Conv2d(in_channels=self.dbb_origin[0].in_channels,
                                     out_channels=self.dbb_origin[0].out_channels,
                                     kernel_size=self.dbb_origin[0].kernel_size,
                                     stride=self.dbb_origin[0].stride,
                                     padding=self.dbb_origin[0].padding,
                                     groups=self.dbb_origin[0].groups,
                                     bias=True)
        self.dbb_reparam.weight.data = kernel
        self.dbb_reparam.bias.data = bias
        # Remove original branches
        del self.dbb_origin, self.dbb_avg, self.dbb_1x1_kxk

    def get_equivalent_kernel_bias(self):
        # Implement kernel fusion logic as needed
        raise NotImplementedError("Kernel fusion not implemented yet.")



class Bottleneck_DAN(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DiverseAttentionNet(c1, c_, k[0], 1)
        self.cv2 = DiverseAttentionNet(c_, c2, k[1], 1, groups=g)

class C2f_DAN(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_DAN(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))



class C3k2_DAN(C2f_DAN):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_DAN(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )
