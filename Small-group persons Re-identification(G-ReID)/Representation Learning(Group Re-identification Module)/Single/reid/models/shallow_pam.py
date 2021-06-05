import torch.nn as nn

from .attention import PAM_Module

__all__ = ['ShallowPAM']

class ShallowPAM(nn.Module):

    def __init__(self, feature_dim: int):

        super().__init__()
        self.input_feature_dim = feature_dim

        #use = args['shallow_pam']
        self._pam_module = pam_module = PAM_Module(self.input_feature_dim)
        #self._pam_module_abc = pam_module  # Forward Compatibility
        """
        if use:
            self._pam_module = pam_module = PAM_Module(self.input_feature_dim)

            if args['compatibility']:
                self._pam_module_abc = pam_module  # Forward Compatibility
        else:
            self._pam_module = None
        """
    def forward(self, x):

        if self._pam_module is not None:
            x = self._pam_module(x)

        return x
