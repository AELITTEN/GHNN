import torch
from ghnn.nets.nnet import NNet
from ghnn.nets.pt_modules import LinearModule, ActivationModule

__all__ = ['LA_SympNet']

class LA_SympNet(NNet):
    """LA-SympNet for data of Hamiltonian systems with a PyTorch backend.

    Args:
        path (str, path-like object): Path where to find a settings file for the NN.

    Attributes:
        settings (dict): All settings.
        model (torch.nn.ModuleDict): The PyTorch modules of the NN.
        dim (int): The number of spatial features and labels.
        dtype (str): Data type for features and labels. 'float' or 'double'.
        device (str): The device to do the computations. 'cpu' or 'gpu'.
    """

    def default_settings(self):
        settings = super().default_settings()
        settings['nn_type'] = 'LA_SympNet'
        settings['modules'] = 10
        settings['sublayer'] = 4
        return settings

    def forward(self, inputs):
        """Defines the computation performed at every call."""
        if inputs.size(-1) == self.dim*2:
            p, q = inputs[..., :self.dim], inputs[..., self.dim:]
        else:
            raise ValueError
        for i in range(self.settings['modules']):
            LinM = self.model['linear_'+str(i+1)]
            ActM = self.model['activation'+str(i+1)]
            p, q = ActM(LinM([p, q]))
        return torch.cat(self.model['output']([p, q]), dim=-1)

    def create_model(self):
        if not isinstance(self.settings['sublayer'], list):
            self.settings['sublayer'] = [self.settings['sublayer']] * (self.settings['modules'] + 1)
        if not isinstance(self.settings['activations'], list):
            self.settings['activations'] = [self.settings['activations']] * self.settings['modules']
        modules = torch.nn.ModuleDict()
        for i in range(self.settings['modules']):
            modules['linear_'+str(i+1)] = LinearModule(self.dim, self.settings['sublayer'][i])
            mode = 'up' if i % 2 == 0 else 'low'
            modules['activation'+str(i+1)] = ActivationModule(self.dim, self.settings['activations'][i], mode)
        modules['output'] = LinearModule(self.dim, self.settings['sublayer'][-1])
        return modules
