import torch.nn as nn

from .rga_modules import RGA_Module

__all__ = ['ShallowRGA']

class ShallowRGA(nn.Module):

    def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=True, cha_ratio=8, spa_ratio=8, down_ratio=8):
        super().__init__()
        self.in_channel = in_channel
        self.in_spatial = in_spatial
        self.use_spatial = use_spatial
        self.use_channel = use_channel
        self.cha_ratio = cha_ratio
        self.spa_ratio = spa_ratio
        self.down_ratio = down_ratio
        #use = args['shallow_rga']
        self._rga_module = rga_module = RGA_Module(self.in_channel, self.in_spatial, self.use_spatial, self.use_channel, self.cha_ratio, self.spa_ratio, self.down_ratio)
        self._rga_module_abc = rga_module  # Forward Compatibility
        """
        if use:
            self._rga_module = rga_module = RGA_Module(self.in_channel, self.in_spatial, self.use_spatial, self.use_channel, self.cha_ratio, self.spa_ratio, self.down_ratio)

            if args['compatibility']:
                self._rga_module_abc = rga_module  # Forward Compatibility
        else:
            self._rga_module = None
        """
    def forward(self, x):

        if self._rga_module is not None:
            self.rga_att = RGA_Module(256, (256//4)*(128//4), use_spatial=True, use_channel=False,
								cha_ratio=8, spa_ratio=8, down_ratio=8)
            x = self.rga_att(x)

        return x

