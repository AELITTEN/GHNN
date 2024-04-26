import torch

__all__ = ['Module', 'DenseModule', 'LinearModule', 'ActivationModule', 'GradientModule', 'SE_HamiltonModule', 'Double_SE_HamiltonModule', 'SV_HamiltonModule', 'Double_SV_HamiltonModule', 'HamiltonModule', 'HenonModule', 'Double_HenonModule', 'mse_w_loss', 'mae_w_loss', 'mse_symp_loss', 'mae_symp_loss']

def find_act(activation):
    """Identifies the activation function from a string.

    Args:
        activation (str): Identifier for the activation function.

    Returns:
        function: torch activation function.
    """
    if activation == 'sigmoid':
        act = torch.sigmoid
    elif activation == 'relu':
        act = torch.relu
    elif activation == 'tanh':
        act = torch.tanh
    elif activation == 'ant_sigmoid':
        act = ant_sigmoid
    elif activation == 'ant_relu':
        act = ant_relu
    elif activation == 'ant_tanh':
        act = ant_tanh
    elif activation == None:
        act = None
    else:
        raise NotImplementedError
    return act

def find_ant_act(activation):
    """Identifies the antiderivative of the activation function from a string.

    Args:
        activation (str): Identifier for the activation function.

    Returns:
        function: torch antiderivative of the activation function.
    """
    if activation == 'sigmoid':
        ant_act = ant_sigmoid
    elif activation == 'relu':
        ant_act = ant_relu
    elif activation == 'tanh':
        ant_act = ant_tanh
    else:
        raise NotImplementedError
    return ant_act

class Module(torch.nn.Module):
    """My PyTorch base class for all neural network modules.

    Args:
        activation (str): Identifier for the activation function of this layer.

    Attributes:
        act (function): Activation function of this layer.
        device (torch.device): The device on which this module lives (cpu or gpu).
        dtype (torch.dtype): The data type of all the varibale (float or double).
    """

    def __init__(self, activation=None):
        super().__init__()
        self.act = find_act(activation)

        self.__device = None
        self.__dtype = None

    @property
    def device(self):
        return self.__device

    @property
    def dtype(self):
        return self.__dtype

    @device.setter
    def device(self, d):
        if d == 'cpu':
            self.cpu()
            for module in self.modules():
                if isinstance(module, Module):
                    module.__device = torch.device('cpu')
        elif d == 'gpu':
            self.cuda()
            for module in self.modules():
                if isinstance(module, Module):
                    module.__device = torch.device('cuda')
        elif d[:3] == 'gpu':
            index = int(d[3:])
            self.cuda(index)
            for module in self.modules():
                if isinstance(module, Module):
                    module.__device = torch.device(f'cuda:{index}')
        else:
            raise ValueError

    @dtype.setter
    def dtype(self, d):
        if d == 'float':
            self.to(torch.float32)
            for module in self.modules():
                if isinstance(module, Module):
                    module.__dtype = torch.float32
        elif d == 'double':
            self.to(torch.float64)
            for module in self.modules():
                if isinstance(module, Module):
                    module.__dtype = torch.float64
        else:
            raise ValueError

class DenseModule(Module):
    """Dense PyTorch layer.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension.
        activation (str): Activation function of this layer.
        bias (bool): Whether a bias is used or not.

    Attributes:
        dim_in (int): Dimension of the spatial inputs/outputs.
        dim_out (int): Dimension of the spatial inputs/outputs.
        act (torch.function): Activation function of this layer.
        bias (bool): Whether a bias is used or not.
        params (torch.nn.ParameterDict): All the weights and biases.
        device (torch.device): The device on which this module lives (cpu or gpu).
        dtype (torch.dtype): The data type of all the varibale (float or double).
    """

    def __init__(self, dim_in, dim_out, activation, bias=True):
        super().__init__(activation=activation)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.bias = bias

        self.params = self.__init_params()

    def forward(self, inputs):
        """Defines the computation performed at every call."""
        if self.bias:
            inputs = inputs @ self.params['A'] + self.params['b']
        else:
            inputs = inputs @ self.params['A']

        if self.act != None:
            outputs = self.act(inputs)
        else:
            outputs = inputs
        return outputs

    def __init_params(self):
        params = torch.nn.ParameterDict()
        params['A'] = torch.nn.Parameter((torch.randn([self.dim_in, self.dim_out]) * 0.01).requires_grad_(True))
        if self.bias:
            params['b'] = torch.nn.Parameter(torch.zeros([self.dim_out]).requires_grad_(True))
        return params

class LinearModule(Module):
    """Linear symplectic PyTorch layer.

    Args:
        dim (int): Dimension of the spatial inputs/outputs.
        layers (int): Number of sublayers.

    Attributes:
        dim (int): Dimension of the spatial inputs/outputs.
        layers (int): Number of sublayers.
        params (torch.nn.ParameterDict): All the weights and biases.
        device (torch.device): The device on which this module lives (cpu or gpu).
        dtype (torch.dtype): The data type of all the varibale (float or double).
    """

    def __init__(self, dim, layers):
        super().__init__(activation=None)
        self.dim = dim
        self.layers = layers

        self.params = self.__init_params()

    def forward(self, inputs):
        """Defines the computation performed at every call."""
        p, q = inputs
        for i in range(self.layers):
            S = self.params[f'S{i+1}']
            if i % 2 == 0:
                p = p + q @ (S + S.t())
            else:
                q = p @ (S + S.t()) + q
        return p + self.params['b'][0], q + self.params['b'][1]

    def __init_params(self):
        params = torch.nn.ParameterDict()
        for i in range(self.layers):
            params[f'S{i+1}'] = torch.nn.Parameter((torch.randn([self.dim, self.dim]) * 0.01).requires_grad_(True))
        params['b'] = torch.nn.Parameter(torch.zeros([2, self.dim]).requires_grad_(True))
        return params

class ActivationModule(Module):
    """Activation symplectic PyTorch layer.

    Args:
        dim (int): Dimension of the spatial inputs/outputs.
        activation (str): Activation function of this layer.
        mode (str): 'up' or 'down'.

    Attributes:
        act (torch.function): Activation function of this layer.
        dim (int): Dimension of the spatial inputs/outputs.
        mode (str): 'up' or 'down'.
        params (torch.nn.ParameterDict): All the scaling factors.
        device (torch.device): The device on which this module lives (cpu or gpu).
        dtype (torch.dtype): The data type of all the varibale (float or double).
    """

    def __init__(self, dim, activation, mode):
        super().__init__(activation=activation)
        if not self.act:
            raise ValueError

        self.dim = dim
        self.mode = mode
        if self.mode != 'up' and self.mode != 'low':
            raise ValueError

        self.params = self.__init_params()

    def forward(self, inputs):
        """Defines the computation performed at every call."""
        p, q = inputs
        if self.mode == 'up':
            return p + self.act(q) * self.params['a'], q
        else:
            return p, self.act(p) * self.params['a'] + q

    def __init_params(self):
        params = torch.nn.ParameterDict()
        params['a'] = torch.nn.Parameter((torch.randn([self.dim]) * 0.01).requires_grad_(True))
        return params

class GradientModule(Module):
    """Gradient symplectic PyTorch layer.

    Args:
        dim (int): Dimension of the spatial inputs/outputs.
        width (int): Width of this layer.
        activation (str): Activation function of this layer.
        mode (str): 'up' or 'down'.

    Attributes:
        act (torch.function): Activation function of this layer.
        dim (int): Dimension of the spatial inputs/outputs.
        width (int): Width of this layer.
        mode (str): 'up' or 'down'.
        params (torch.nn.ParameterDict): All the weights, biases and scaling factors.
        device (torch.device): The device on which this module lives (cpu or gpu).
        dtype (torch.dtype): The data type of all the varibale (float or double).
    """

    def __init__(self, dim, width, activation, mode):
        super().__init__(activation=activation)
        if not self.act:
            raise ValueError

        self.dim = dim
        self.width = width
        self.mode = mode
        if self.mode != 'up' and self.mode != 'low':
            raise ValueError

        self.params = self.__init_params()

    def forward(self, inputs):
        """Defines the computation performed at every call."""
        p, q = inputs
        if self.mode == 'up':
            gradH = (self.act(q @ self.params['K'] + self.params['b']) * self.params['a']) @ self.params['K'].t()
            return p + gradH, q
        else:
            gradH = (self.act(p @ self.params['K'] + self.params['b']) * self.params['a']) @ self.params['K'].t()
            return p, gradH + q

    def __init_params(self):
        params = torch.nn.ParameterDict()
        params['K'] = torch.nn.Parameter((torch.randn([self.dim, self.width]) * 0.01).requires_grad_(True))
        params['b'] = torch.nn.Parameter(torch.zeros([self.width]).requires_grad_(True))
        params['a'] = torch.nn.Parameter((torch.randn([self.width]) * 0.01).requires_grad_(True))
        return params

class SE_HamiltonModule(Module):
    """Forward Hamilton module with one hidden layer and symplectic Euler.

    Args:
        dim (int): Dimension of the spatial inputs/outputs.
        neurons (int): Number of neurons in the hidden layer.
        activation (str): Activation function of the hidden layer.

    Attributes:
        act (torch.function): Activation function of the hidden layer.
        dim (int): Dimension of the spatial inputs/outputs.
        neurons (int): Number of neurons in the hidden layer.
        params (torch.nn.ParameterDict): All the weights, biases and scaling factors.
        device (torch.device): The device on which this module lives (cpu or gpu).
        dtype (torch.dtype): The data type of all the varibale (float or double).
    """

    def __init__(self, dim, neurons, activation):
        super().__init__(activation=activation)
        if not self.act:
            raise ValueError

        self.dim = dim
        self.neurons = neurons

        self.params = self.__init_params()

    def forward(self, inputs):
        """Defines the computation performed at every call."""
        p, q = inputs
        p = p - (self.act(q @ self.params['K_p'] + self.params['b_p']) * self.params['a_p']) @ self.params['K_p'].t()
        q = q + (self.act(p @ self.params['K_q'] + self.params['b_q']) * self.params['a_q']) @ self.params['K_q'].t()

        return p, q

    def __init_params(self):
        params = torch.nn.ParameterDict()
        params['K_p'] = torch.nn.Parameter((torch.randn([self.dim, self.neurons]) * 0.01).requires_grad_(True))
        params['K_q'] = torch.nn.Parameter((torch.randn([self.dim, self.neurons]) * 0.01).requires_grad_(True))
        params['b_p'] = torch.nn.Parameter(torch.zeros([self.neurons]).requires_grad_(True))
        params['b_q'] = torch.nn.Parameter(torch.zeros([self.neurons]).requires_grad_(True))
        params['a_p'] = torch.nn.Parameter((torch.randn([self.neurons]) * 0.01).requires_grad_(True))
        params['a_q'] = torch.nn.Parameter((torch.randn([self.neurons]) * 0.01).requires_grad_(True))
        return params

class Double_SE_HamiltonModule(Module):
    """Forward Hamilton module with two hidden layers and symplectic Euler.

    Args:
        dim (int): Dimension of the spatial inputs/outputs.
        neurons (int): Number of neurons in the hidden layers.
        activations (str[]): Activation functions of the layers.

    Attributes:
        act (torch.function): Activation functions of the layers.
        dim (int): Dimension of the spatial inputs/outputs.
        neurons (int): Number of neurons in the hidden layers.
        params (torch.nn.ParameterDict): All the weights, biases and scaling factors.
        device (torch.device): The device on which this module lives (cpu or gpu).
        dtype (torch.dtype): The data type of all the varibale (float or double).
    """

    def __init__(self, dim, neurons, activations):
        super().__init__()
        if not isinstance(neurons, list):
            neurons = [neurons] * 2
        if not isinstance(activations, list):
            activations = [activations] * 2

        self.act1 = find_act(activations[0])
        self.act2 = find_act(activations[1])
        self.ant_act = find_ant_act(activations[0])

        self.dim = dim
        self.neurons = neurons

        self.params = self.__init_params()

    def forward(self, inputs):
        """Defines the computation performed at every call."""
        p, q = inputs

        inner = q @ self.params['K1_p'] + self.params['b1_p']
        act_inner = self.ant_act(inner) @ self.params['K2_p'] + self.params['b2_p']
        nex = (self.act2(act_inner) * self.params['a_p']) @ self.params['K2_p'].t()
        p = p - (self.act1(inner) * nex) @ self.params['K1_p'].t()

        inner = p @ self.params['K1_q'] + self.params['b1_q']
        act_inner = self.ant_act(inner) @ self.params['K2_q'] + self.params['b2_q']
        nex = (self.act2(act_inner) * self.params['a_q']) @ self.params['K2_q'].t()
        q = q + (self.act1(inner) * nex) @ self.params['K1_q'].t()

        return p, q

    def __init_params(self):
        params = torch.nn.ParameterDict()

        params['K1_p'] = torch.nn.Parameter((torch.randn([self.dim, self.neurons[0]]) * 0.01).requires_grad_(True))
        params['K2_p'] = torch.nn.Parameter((torch.randn([self.neurons[0], self.neurons[1]]) * 0.01).requires_grad_(True))
        params['b1_p'] = torch.nn.Parameter(torch.zeros([self.neurons[0]]).requires_grad_(True))
        params['b2_p'] = torch.nn.Parameter(torch.zeros([self.neurons[1]]).requires_grad_(True))
        params['a_p'] = torch.nn.Parameter((torch.randn([self.neurons[1]]) * 0.01).requires_grad_(True))

        params['K1_q'] = torch.nn.Parameter((torch.randn([self.dim, self.neurons[0]]) * 0.01).requires_grad_(True))
        params['K2_q'] = torch.nn.Parameter((torch.randn([self.neurons[0], self.neurons[1]]) * 0.01).requires_grad_(True))
        params['b1_q'] = torch.nn.Parameter(torch.zeros([self.neurons[0]]).requires_grad_(True))
        params['b2_q'] = torch.nn.Parameter(torch.zeros([self.neurons[1]]).requires_grad_(True))
        params['a_q'] = torch.nn.Parameter((torch.randn([self.neurons[1]]) * 0.01).requires_grad_(True))

        return params

class SV_HamiltonModule(Module):
    """Forward Hamilton module with one hidden layer and Stoermer-Verlet.

    Args:
        dim (int): Dimension of the spatial inputs/outputs.
        neurons (int): Number of neurons in the hidden layer.
        activation (str): Activation function of the hidden layer.

    Attributes:
        act (torch.function): Activation function of the hidden layer.
        dim (int): Dimension of the spatial inputs/outputs.
        neurons (int): Number of neurons in the hidden layer.
        params (torch.nn.ParameterDict): All the weights, biases and scaling factors.
        device (torch.device): The device on which this module lives (cpu or gpu).
        dtype (torch.dtype): The data type of all the varibale (float or double).
    """

    def __init__(self, dim, neurons, activation):
        super().__init__(activation=activation)
        if not self.act:
            raise ValueError

        self.dim = dim
        self.neurons = neurons

        self.params = self.__init_params()

    def forward(self, inputs):
        """Defines the computation performed at every call."""
        p, q = inputs
        p = p - 0.5*(self.act(q @ self.params['K_p'] + self.params['b_p']) * self.params['a_p']) @ self.params['K_p'].t()
        q = q + (self.act(p @ self.params['K_q'] + self.params['b_q']) * self.params['a_q']) @ self.params['K_q'].t()
        p = p - 0.5*(self.act(q @ self.params['K_p'] + self.params['b_p']) * self.params['a_p']) @ self.params['K_p'].t()

        return p, q

    def __init_params(self):
        params = torch.nn.ParameterDict()
        params['K_p'] = torch.nn.Parameter((torch.randn([self.dim, self.neurons]) * 0.01).requires_grad_(True))
        params['K_q'] = torch.nn.Parameter((torch.randn([self.dim, self.neurons]) * 0.01).requires_grad_(True))
        params['b_p'] = torch.nn.Parameter(torch.zeros([self.neurons]).requires_grad_(True))
        params['b_q'] = torch.nn.Parameter(torch.zeros([self.neurons]).requires_grad_(True))
        params['a_p'] = torch.nn.Parameter((torch.randn([self.neurons]) * 0.01).requires_grad_(True))
        params['a_q'] = torch.nn.Parameter((torch.randn([self.neurons]) * 0.01).requires_grad_(True))
        return params

class Double_SV_HamiltonModule(Module):
    """Forward Hamilton module with two hidden layers and Stoermer-Verlet.

    Args:
        dim (int): Dimension of the spatial inputs/outputs.
        neurons (int): Number of neurons in the hidden layers.
        activations (str[]): Activation functions of the layers.

    Attributes:
        act (torch.function): Activation functions of the layers.
        dim (int): Dimension of the spatial inputs/outputs.
        neurons (int): Number of neurons in the hidden layers.
        params (torch.nn.ParameterDict): All the weights, biases and scaling factors.
        device (torch.device): The device on which this module lives (cpu or gpu).
        dtype (torch.dtype): The data type of all the varibale (float or double).
    """

    def __init__(self, dim, neurons, activations):
        super().__init__()
        if not isinstance(neurons, list):
            neurons = [neurons] * 2
        if not isinstance(activations, list):
            activations = [activations] * 2

        self.act1 = find_act(activations[0])
        self.act2 = find_act(activations[1])
        self.ant_act = find_ant_act(activations[0])

        self.dim = dim
        self.neurons = neurons

        self.params = self.__init_params()

    def forward(self, inputs):
        """Defines the computation performed at every call."""
        p, q = inputs

        inner = q @ self.params['K1_p'] + self.params['b1_p']
        act_inner = self.ant_act(inner) @ self.params['K2_p'] + self.params['b2_p']
        nex = (self.act2(act_inner) * self.params['a_p']) @ self.params['K2_p'].t()
        p = p - 0.5*(self.act1(inner) * nex) @ self.params['K1_p'].t()

        inner = p @ self.params['K1_q'] + self.params['b1_q']
        act_inner = self.ant_act(inner) @ self.params['K2_q'] + self.params['b2_q']
        nex = (self.act2(act_inner) * self.params['a_q']) @ self.params['K2_q'].t()
        q = q + (self.act1(inner) * nex) @ self.params['K1_q'].t()

        inner = q @ self.params['K1_p'] + self.params['b1_p']
        act_inner = self.ant_act(inner) @ self.params['K2_p'] + self.params['b2_p']
        nex = (self.act2(act_inner) * self.params['a_p']) @ self.params['K2_p'].t()
        p = p - 0.5*(self.act1(inner) * nex) @ self.params['K1_p'].t()

        return p, q

    def __init_params(self):
        params = torch.nn.ParameterDict()

        params['K1_p'] = torch.nn.Parameter((torch.randn([self.dim, self.neurons[0]]) * 0.01).requires_grad_(True))
        params['K2_p'] = torch.nn.Parameter((torch.randn([self.neurons[0], self.neurons[1]]) * 0.01).requires_grad_(True))
        params['b1_p'] = torch.nn.Parameter(torch.zeros([self.neurons[0]]).requires_grad_(True))
        params['b2_p'] = torch.nn.Parameter(torch.zeros([self.neurons[1]]).requires_grad_(True))
        params['a_p'] = torch.nn.Parameter((torch.randn([self.neurons[1]]) * 0.01).requires_grad_(True))

        params['K1_q'] = torch.nn.Parameter((torch.randn([self.dim, self.neurons[0]]) * 0.01).requires_grad_(True))
        params['K2_q'] = torch.nn.Parameter((torch.randn([self.neurons[0], self.neurons[1]]) * 0.01).requires_grad_(True))
        params['b1_q'] = torch.nn.Parameter(torch.zeros([self.neurons[0]]).requires_grad_(True))
        params['b2_q'] = torch.nn.Parameter(torch.zeros([self.neurons[1]]).requires_grad_(True))
        params['a_q'] = torch.nn.Parameter((torch.randn([self.neurons[1]]) * 0.01).requires_grad_(True))

        return params

class HamiltonModule(Module):
    """Hamilton module.

    Args:
        dim (int): Dimension of the spatial inputs/outputs.
        neurons (int[]): Neurons of the hidden layer.
        layer (int): Number of hidden layers inside this module.
        activations (str[]): Activation functions of the layers.
        integrator (str): Which integrator should be used.

    Attributes:
        integrator (str): Which integrator should be used.
        layer (int): Number of hidden layers inside this module.
        model (torch.nn.ModuleDict): The PyTorch sub-modules of this module.
        device (torch.device): The device on which this module lives (cpu or gpu).
        dtype (torch.dtype): The data type of all the varibale (float or double).
    """

    def __init__(self, dim, neurons, layer, activations, integrator):
        super().__init__()
        if not isinstance(neurons, list):
            neurons = [neurons] * layer
        if not isinstance(activations, list):
            activations = [activations] * layer

        modules = torch.nn.ModuleDict()
        modules['dense_1_T'] = DenseModule(dim,
                                           neurons[0],
                                           activations[0])
        modules['dense_1_U'] = DenseModule(dim,
                                           neurons[0],
                                           activations[0])
        for i in range(1, layer):
            modules[f'dense_{i+1}_T'] = DenseModule(neurons[i-1],
                                                    neurons[i],
                                                    activations[i])
            modules[f'dense_{i+1}_U'] = DenseModule(neurons[i-1],
                                                    neurons[i],
                                                    activations[i])
        modules['out_T'] = DenseModule(neurons[-1],
                                       1,
                                       None,
                                       bias=False)
        modules['out_U'] = DenseModule(neurons[-1],
                                       1,
                                       None,
                                       bias=False)

        self.model = modules
        self.integrator = integrator
        self.layer = layer

    def forward(self, inputs):
        """Defines the computation performed at every call."""
        volatile = False

        p, q = inputs
        p = p.requires_grad_()
        q = q.requires_grad_()

        if self.integrator == 'symp_euler':
            U = self.model['dense_1_U'](q)
            for i in range(1, self.layer):
                U = self.model[f'dense_{i+1}_U'](U)
            U = self.model['out_U'](U)
            grad_q = torch.autograd.grad(U.sum(), q, create_graph=not volatile)[0]
            p = p - grad_q

            T = self.model['dense_1_T'](p)
            for i in range(1, self.layer):
                T = self.model[f'dense_{i+1}_T'](T)
            T = self.model['out_T'](T)
            grad_p = torch.autograd.grad(T.sum(), p, create_graph=not volatile)[0]
            q = q + grad_p

        elif self.integrator == 'leapfrog':
            U = self.model['dense_1_U'](q)
            for i in range(1, self.layer):
                U = self.model[f'dense_{i+1}_U'](U)
            U = self.model['out_U'](U)
            grad_q = torch.autograd.grad(U.sum(), q, create_graph=not volatile)[0]
            p = p - grad_q

            T = self.model['dense_1_T'](p)
            for i in range(1, self.layer):
                T = self.model[f'dense_{i+1}_T'](T)
            T = self.model['out_T'](T)
            grad_p = torch.autograd.grad(T.sum(), p, create_graph=not volatile)[0]
            q = q + grad_p

        elif self.integrator == 'stoermer_verlet':
            U = self.model['dense_1_U'](q)
            for i in range(1, self.layer):
                U = self.model[f'dense_{i+1}_U'](U)
            U = self.model['out_U'](U)
            grad_q = torch.autograd.grad(U.sum(), q, create_graph=not volatile)[0]
            p = p - 0.5 * grad_q

            T = self.model['dense_1_T'](p)
            for i in range(1, self.layer):
                T = self.model[f'dense_{i+1}_T'](T)
            T = self.model['out_T'](T)
            grad_p = torch.autograd.grad(T.sum(), p, create_graph=not volatile)[0]
            q = q + grad_p

            U = self.model['dense_1_U'](q)
            for i in range(1, self.layer):
                U = self.model[f'dense_{i+1}_U'](U)
            U = self.model['out_U'](U)
            grad_q = torch.autograd.grad(U.sum(), q, create_graph=not volatile)[0]
            p = p - 0.5 * grad_q

        else:
            raise ValueError

        return p, q

class HenonModule(Module):
    """Forward Henon module with one hidden layers.

    Args:
        dim (int): Dimension of the spatial inputs/outputs.
        width (int): Width of the hidden layer.
        activation (str): Activation function of this module.

    Attributes:
        act (torch.function): Activation function of this module.
        dim (int): Dimension of the spatial inputs/outputs.
        width (int): Width of the hidden layer.
        params (torch.nn.ParameterDict): All the weights, biases and scaling factors.
        device (torch.device): The device on which this module lives (cpu or gpu).
        dtype (torch.dtype): The data type of all the varibale (float or double).
    """

    def __init__(self, dim, width, activation):
        super().__init__(activation=activation)
        if not self.act:
            raise ValueError

        self.dim = dim
        self.width = width

        self.params = self.__init_params()

    def forward(self, inputs):
        """Defines the computation performed at every call."""
        p, q = inputs

        q_new = p + self.params['eta']
        p_new = -q + (self.act(p @ self.params['W'] + self.params['b']) * self.params['a']) @ self.params['W'].t()

        q = p_new + self.params['eta']
        p = -q_new + (self.act(p_new @ self.params['W'] + self.params['b']) * self.params['a']) @ self.params['W'].t()

        q_new = p + self.params['eta']
        p_new = -q + (self.act(p @ self.params['W'] + self.params['b']) * self.params['a']) @ self.params['W'].t()

        q = p_new + self.params['eta']
        p = -q_new + (self.act(p_new @ self.params['W'] + self.params['b']) * self.params['a']) @ self.params['W'].t()

        return p, q

    def __init_params(self):
        params = torch.nn.ParameterDict()
        params['W'] = torch.nn.Parameter((torch.randn([self.dim, self.width]) * 0.01).requires_grad_(True))
        params['b'] = torch.nn.Parameter(torch.zeros([self.width]).requires_grad_(True))
        params['a'] = torch.nn.Parameter((torch.randn([self.width]) * 0.01).requires_grad_(True))
        params['eta'] = torch.nn.Parameter((torch.randn([self.dim]) * 0.01).requires_grad_(True))
        return params

class Double_HenonModule(Module):
    """Forward Henon module with two hidden layers.

    Args:
        dim (int): Dimension of the spatial inputs/outputs.
        width1 (int): Width of the first hidden layer.
        width2 (int): Width of the second hidden layer.
        activation (str): Activation function of this module.

    Attributes:
        act (torch.function): Activation function of this module.
        dim (int): Dimension of the spatial inputs/outputs.
        width1 (int): Width of the first hidden layer.
        width2 (int): Width of the second hidden layer.
        params (torch.nn.ParameterDict): All the weights, biases and scaling factors.
        device (torch.device): The device on which this module lives (cpu or gpu).
        dtype (torch.dtype): The data type of all the varibale (float or double).
    """

    def __init__(self, dim, width1, width2, activation):
        if not activation:
            raise ValueError

        super().__init__(activation=activation)
        if activation == 'sigmoid':
            self.ant_act = ant_sigmoid
        elif activation == 'relu':
            self.ant_act = ant_relu
        elif activation == 'tanh':
            self.ant_act = ant_tanh
        else:
            raise NotImplementedError

        self.dim = dim
        self.width1 = width1
        self.width2 = width2

        self.params = self.__init_params()

    def forward(self, inputs):
        """Defines the computation performed at every call."""
        p, q = inputs

        q_new = p + self.params['eta']
        inner = p @ self.params['W1'] + self.params['b1']
        act_inner = self.ant_act(inner) @ self.params['W2'] + self.params['b2']
        nex = (self.act(act_inner) * self.params['a']) @ self.params['W2'].t()
        p_new = -q + (self.act(inner) * nex) @ self.params['W1'].t()

        q = p_new + self.params['eta']
        inner = p_new @ self.params['W1'] + self.params['b1']
        act_inner = self.ant_act(inner) @ self.params['W2'] + self.params['b2']
        nex = (self.act(act_inner) * self.params['a']) @ self.params['W2'].t()
        p = -q_new + (self.act(inner) * nex) @ self.params['W1'].t()

        q_new = p + self.params['eta']
        inner = p @ self.params['W1'] + self.params['b1']
        act_inner = self.ant_act(inner) @ self.params['W2'] + self.params['b2']
        nex = (self.act(act_inner) * self.params['a']) @ self.params['W2'].t()
        p_new = -q + (self.act(inner) * nex) @ self.params['W1'].t()

        q = p_new + self.params['eta']
        inner = p_new @ self.params['W1'] + self.params['b1']
        act_inner = self.ant_act(inner) @ self.params['W2'] + self.params['b2']
        nex = (self.act(act_inner) * self.params['a']) @ self.params['W2'].t()
        p = -q_new + (self.act(inner) * nex) @ self.params['W1'].t()

        return p, q

    def __init_params(self):
        params = torch.nn.ParameterDict()
        params['W1'] = torch.nn.Parameter((torch.randn([self.dim, self.width1]) * 0.01).requires_grad_(True))
        params['W2'] = torch.nn.Parameter((torch.randn([self.width1, self.width2]) * 0.01).requires_grad_(True))
        params['b1'] = torch.nn.Parameter(torch.zeros([self.width1]).requires_grad_(True))
        params['b2'] = torch.nn.Parameter(torch.zeros([self.width2]).requires_grad_(True))
        params['a'] = torch.nn.Parameter((torch.randn([self.width2]) * 0.01).requires_grad_(True))
        params['eta'] = torch.nn.Parameter((torch.randn([self.dim]) * 0.01).requires_grad_(True))
        return params

def ant_sigmoid(inp):
    return torch.log1p(torch.exp(inp))

def ant_relu(inp):
    return 0.5*torch.pow(torch.relu(inp), 2)

def ant_tanh(inp):
    return torch.log(torch.cosh(inp))

class mse_w_loss(Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, output, target):
        loss = torch.mean(((output - target) * self.weights)**2)
        return loss

class mae_w_loss(Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, output, target):
        loss = torch.mean(torch.abs((output - target) * self.weights))
        return loss

class mse_symp_loss(Module):
    def __init__(self, positions, symp_lambda):
        super().__init__()
        self.positions = positions
        self.symp_lambda = symp_lambda

    def forward(self, nn, inp, target):
        output = nn(inp)
        data_loss = torch.mean((output - target)**2)
        symp_loss = nn.calc_symp_mse(self.positions)
        symp_loss = torch.mean(symp_loss)
        return data_loss + self.symp_lambda * symp_loss

class mae_symp_loss(Module):
    def __init__(self, positions, symp_lambda):
        super().__init__()
        self.positions = positions
        self.symp_lambda = symp_lambda

    def forward(self, nn, inp, target):
        output = nn(inp)
        data_loss = torch.mean(torch.abs(output - target))
        symp_loss = nn.calc_symp_mae(self.positions)
        symp_loss = torch.mean(symp_loss)
        return data_loss + self.symp_lambda * symp_loss
