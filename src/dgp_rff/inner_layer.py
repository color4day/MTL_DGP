import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import numpy as np
from pyro.nn import PyroModule, PyroSample
from torch import Tensor

# class FirstLayer(PyroModule):
#     r"""
#     The random feature-based GP is equivalent to a two-layer Bayesian neural network.
#     The first layer refers to generate the random feature $\phi$.
#
#     Attributes
#     ----------
#     J: int
#         The number of random feature.
#     layer: PyroModule
#         The first layer is based on nn.Linear where the weight($\Omega$) is defined by the PyroSample.
#     """
#     def __init__(
#             self,
#             in_dim: int = 1,
#             hid_dim: int = 100,
#             num_layer: int = 1,
#     ) -> None:
#         """
#         :param in_dim: int
#             The input dimension.
#         :param hid_dim: int
#             The hidden state's dimension, which is 2J. The hidden state here refers to the output of
#             the first layer and the input of the second layer.
#         """
#         super().__init__()
#
#         self.J = hid_dim // 2
#         self.layer = PyroModule[nn.Linear](in_dim, self.J, bias=False)
#
#         self.layer.weight = pyro.sample(f"RBF {num_layer}-th Omega", dist.Normal(0., 1.).expand([self.J, in_dim]).to_event(2))
class FirstLayer(PyroModule):
    r"""
    The random feature-based GP is equivalent to a two-layer Bayesian neural network.
    The first layer refers to generate the random feature $\phi$.
    Kernel is RBF kernel.
    Attributes
    ----------
    J: int
        The number of random feature.
    layer: PyroModule
        The first layer is based on nn.Linear where the weight($\Omega$) is defined by the PyroSample.
    """

    def __init__(
            self,
            in_dim: int = 1,
            hid_dim: int = 100,
            # num_layer: int = 1,
    ) -> None:
        """
        :param in_dim: int
            The input dimension.
        :param hid_dim: int
            The hidden state's dimension, which is 2J. The hidden state here refers to the output of
            the first layer and the input of the second layer.
        """
        super().__init__()

        self.J = hid_dim // 2
        self.layer = PyroModule[nn.Linear](in_dim, self.J, bias=False)
        self.layer.weight = PyroSample(dist.Normal(0., 1.0).expand([self.J, in_dim]).to_event(2))
        # self.layer.weight = PyroSample(dist.Normal(0., 1.).expand([self.J, in_dim]).to_event(2))
        #pyro.sample()

    # prior of Omega(5*4): location (5*4): 0.0; scale (5*4): 1.0
    #
    # posterior of Omega:  location (5*4): variational parameter; scale (5*4): variation parameter
    #
    # sample from posterior: N(location, scale)

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        r"""
        :param x: Tensor
            The input to the FirstLayer.
        :return: Tensor
            The output of the FirstLayer, which is $\phi(\Omega \times x)$.
        """
        hid = self.layer(x)
        mu = torch.cat((torch.sin(hid).cuda(), torch.cos(hid).cuda()), dim=-1) / torch.sqrt(torch.tensor(self.J).cuda())

        return mu

    # init_w = None,
    # init_b = None,
    # if init_w is None:
    #     init_w = torch.randn(out_dim, hid_dim).cuda()
class SecondLayertest(PyroModule):
    r"""
    The random feature-based GP is equivalent to a two-layer Bayesian neural network.
    The second layer refers to produce the GP output plus noises.

    Attributes
    ----------
    J: int
        The number of random feature.
    layer: PyroModule
        The second layer is based on nn.Linear where the weight($\Theta$) and bias($\Epsilon$) is defined
        by the PyroSample.
    """

    def __init__(
            self,
            hid_dim: int = 100,
            out_dim: int = 1,
            init_w_mean = None,
            init_w_var = None,
            init_b_mean=None,
            init_b_var=None,
            # num_layer: int = 1,
    ) -> None:
        """
        :param out_dim: int
            The output dimension.
        :param hid_dim: int
            The hidden state's dimension, which is 2J. The hidden state here refers to the output of
            the first layer and the input of the second layer.
        """
        super().__init__()
        if init_w_mean is None:
            init_w_mean = torch.zeros(out_dim, hid_dim).cuda()
        else:
            init_w_mean = init_w_mean.reshape(out_dim, hid_dim).cuda()
        if init_w_var is None:
            init_w_var = torch.ones(out_dim, hid_dim).cuda()
        else:
            init_w_var = init_w_var.reshape(out_dim, hid_dim).cuda()
        if init_b_mean is None:
            init_b_mean = torch.zeros(out_dim, hid_dim).cuda()
        else:
            init_b_mean = init_b_mean.reshape(out_dim, hid_dim).cuda()
        if init_b_var is None:
            init_b_var = torch.ones(out_dim, hid_dim).cuda()
        else:
            init_b_var = init_b_var.reshape(out_dim, hid_dim).cuda()

        self.J = hid_dim // 2
        self.layer = PyroModule[nn.Linear](hid_dim, out_dim, bias=True)
        # self.layer.weight = pyro.sample(f"{num_layer}-th Weight",
        #                                 dist.Normal(0., torch.tensor(1.0,device = 'cuda')).expand([out_dim, hid_dim]).to_event(2))
        self.layer.weight = PyroSample(dist.Normal(init_w_mean, init_w_var))#torch.ones(out_dim,hid_dim)).to_event(2))
        # self.layer.bias = pyro.sample(f"{num_layer}-th bias",dist.Normal(0., torch.tensor(1.0,device = 'cuda')).expand([out_dim]).to_event(1))
        self.layer.bias = PyroSample(dist.Normal(init_b_mean, init_b_var))
        #self.layer.bias = PyroSample(dist.Normal(0., torch.tensor(1.0, device='cuda')).expand([out_dim]).to_event(1))
    # known: phi(Omega x)
    # want: phi(Omega x) W + epsilon
    # epsilon : (2,)
    # Secondlayer:
    #     layer.weight: W;
    #     layer.bias: epsilon

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        r"""
        :param x: Tensor
            The input to the SecondLayer.
        :return: Tensor
            The output of the SecondLayer, which is $\phi(\Omega \times x)\Theta + \Epsilon$.
        """
        mu = self.layer(x)

        return mu

class SecondLayer(PyroModule):
    r"""
    The random feature-based GP is equivalent to a two-layer Bayesian neural network.
    The second layer refers to produce the GP output plus noises.

    Attributes
    ----------
    J: int
        The number of random feature.
    layer: PyroModule
        The second layer is based on nn.Linear where the weight($\Theta$) and bias($\Epsilon$) is defined
        by the PyroSample.
    """

    def __init__(
            self,
            hid_dim: int = 100,
            out_dim: int = 1,
            # num_layer: int = 1,
    ) -> None:
        """
        :param out_dim: int
            The output dimension.
        :param hid_dim: int
            The hidden state's dimension, which is 2J. The hidden state here refers to the output of
            the first layer and the input of the second layer.
        """
        super().__init__()

        self.J = hid_dim // 2
        self.layer = PyroModule[nn.Linear](hid_dim, out_dim, bias=True)
        # self.layer.weight = pyro.sample(f"{num_layer}-th Weight",
        #                                 dist.Normal(0., torch.tensor(1.0,device = 'cuda')).expand([out_dim, hid_dim]).to_event(2))
        self.layer.weight = PyroSample(dist.Normal(1., torch.tensor(1.0,device = 'cuda')).expand([out_dim, hid_dim]).to_event(2))
        # self.layer.bias = pyro.sample(f"{num_layer}-th bias",dist.Normal(0., torch.tensor(1.0,device = 'cuda')).expand([out_dim]).to_event(1))

        self.layer.bias = PyroSample(dist.Normal(0., torch.tensor(1.0, device='cuda')).expand([out_dim]).to_event(1))
    # known: phi(Omega x)
    # want: phi(Omega x) W + epsilon
    # epsilon : (2,)
    # Secondlayer:
    #     layer.weight: W;
    #     layer.bias: epsilon

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        r"""
        :param x: Tensor
            The input to the SecondLayer.
        :return: Tensor
            The output of the SecondLayer, which is $\phi(\Omega \times x)\Theta + \Epsilon$.
        """
        mu = self.layer(x)

        return mu

class SingleLayerFix(PyroModule):
    r"""
    The random feature-based GP is equivalent to a two-layer Bayesian neural network.
    The first layer refers to generate the random feature $\phi$.
    Kernel is RBF kernel.
    Attributes
    ----------
    J: int
        The number of random feature.
    layer: PyroModule
        The first layer is based on nn.Linear where the weight($\Omega$) is defined by the PyroSample.
    """

    def __init__(
            self,
            in_dim: int = 1,
            hid_dim: int = 100,
            out_dim: int = 1,
            seed = 1
            # num_layer: int = 1,
    ) -> None:
        """
        :param in_dim: int
            The input dimension.
        :param hid_dim: int
            The hidden state's dimension, which is 2J. The hidden state here refers to the output of
            the first layer and the input of the second layer.
        """
        super().__init__()

        # self.J = hid_dim // 2
        # self.layer = PyroModule[nn.Linear](in_dim, self.J, bias=False)
        # torch.manual_seed(seed)
        # self.layer.weight =  torch.rand(self.J,in_dim)#PyroSample(dist.Normal(0., 1.).expand([self.J, in_dim]).to_event(2))
        # #pyro.sample()

    # prior of Omega(5*4): location (5*4): 0.0; scale (5*4): 1.0
    #
    # posterior of Omega:  location (5*4): variational parameter; scale (5*4): variation parameter
    #
    # sample from posterior: N(location, scale)

    # def forward(
    #         self,
    #         x: Tensor,
    # ) -> Tensor:
    #     r"""
    #     :param x: Tensor
    #         The input to the FirstLayer.
    #     :return: Tensor
    #         The output of the FirstLayer, which is $\phi(\Omega \times x)$.
    #     """
    #     hid = self.layer(x)
    #     mu = torch.cat((torch.sin(hid).cuda(), torch.cos(hid).cuda()), dim=-1) / torch.sqrt(torch.tensor(self.J).cuda())

        torch.manual_seed(seed)
        self.J = hid_dim // 2
        self.in_dim = in_dim
        self.layer = PyroModule[nn.Linear](hid_dim, out_dim, bias=True)
        # self.layer.weight = pyro.sample(f"{num_layer}-th Weight",
        #                                 dist.Normal(0., torch.tensor(1.0,device = 'cuda')).expand([out_dim, hid_dim]).to_event(2))
        self.layer.weight = PyroSample(dist.Normal(0., torch.tensor(1.0,device = 'cuda')).expand([out_dim, hid_dim]).to_event(2))
        # self.layer.bias = pyro.sample(f"{num_layer}-th bias",dist.Normal(0., torch.tensor(1.0,device = 'cuda')).expand([out_dim]).to_event(1))

        self.layer.bias = PyroSample(dist.Normal(0., torch.tensor(1.0, device='cuda')).expand([out_dim]).to_event(1))
    # known: phi(Omega x)
    # want: phi(Omega x) W + epsilon
    # epsilon : (2,)
    # Secondlayer:
    #     layer.weight: W;
    #     layer.bias: epsilon

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        r"""
        :param x: Tensor
            The input to the SecondLayer.
        :return: Tensor
            The output of the SecondLayer, which is $\phi(\Omega \times x)\Theta + \Epsilon$.
        """

        hid = x @ torch.rand(self.J, self.in_dim).T
        x1 = torch.cat((torch.sin(hid).cuda(), torch.cos(hid).cuda()), dim=-1) / torch.sqrt(torch.tensor(self.J).cuda())
        mu = self.layer(x1)

        return mu

class SecondLayerNoBias(PyroModule):
    r"""
    The random feature-based GP is equivalent to a two-layer Bayesian neural network.
    The second layer refers to produce the GP output plus noises.

    Attributes
    ----------
    J: int
        The number of random feature.
    layer: PyroModule
        The second layer is based on nn.Linear where the weight($\Theta$) and bias($\Epsilon$) is defined
        by the PyroSample.
    """

    def __init__(
            self,
            hid_dim: int = 100,
            out_dim: int = 1,
            # num_layer: int = 1,
    ) -> None:
        """
        :param out_dim: int
            The output dimension.
        :param hid_dim: int
            The hidden state's dimension, which is 2J. The hidden state here refers to the output of
            the first layer and the input of the second layer.
        """
        super().__init__()

        self.J = hid_dim // 2
        self.layer = PyroModule[nn.Linear](hid_dim, out_dim, bias=False)
        # self.layer.weight = pyro.sample(f"{num_layer}-th Weight", dist.Normal(0., torch.tensor(1.0,device = 'cuda')).expand([out_dim, hid_dim]).to_event(2))
        self.layer.weight = PyroSample(dist.Normal(0., torch.tensor(1.0,device = 'cuda')).expand([out_dim, hid_dim]).to_event(2))
        # self.layer.bias = PyroSample(dist.Normal(0., torch.tensor(1.0,device = 'cuda')).expand([out_dim]).to_event(1))

    # known: phi(Omega x)
    # want: phi(Omega x) W + epsilon
    # epsilon : (2,)
    # Secondlayer:
    #     layer.weight: W;
    #     layer.bias: epsilon

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        r"""
        :param x: Tensor
            The input to the SecondLayer.
        :return: Tensor
            The output of the SecondLayer, which is $\phi(\Omega \times x)\Theta + \Epsilon$.
        """
        mu = self.layer(x)

        return mu

class FirstLaplacianLayer(PyroModule):
    r"""
    The random feature-based GP is equivalent to a two-layer Bayesian neural network.
    The first layer refers to generate the random feature $\phi$.
    The kernel here is Laplacian kernel
    Attributes
    ----------
    J: int
        The number of random feature.
    layer: PyroModule
        The first layer is based on nn.Linear where the weight($\Omega$) is defined by the PyroSample.
    """

    def __init__(
            self,
            in_dim: int = 1,
            hid_dim: int = 100,
            # num_layer: int = 1,
    ) -> None:
        """
        :param in_dim: int
            The input dimension.
        :param hid_dim: int
            The hidden state's dimension, which is 2J. The hidden state here refers to the output of
            the first layer and the input of the second layer.
        """
        super().__init__()

        self.J = hid_dim // 2
        self.layer = PyroModule[nn.Linear](in_dim, self.J, bias=False)
        # self.layer.weight = pyro.sample(f"Laplacian {num_layer}-th Omega",
        #                                 dist.Cauchy(0., 1.).expand([self.J, in_dim]).to_event(2))
        self.layer.weight = PyroSample(dist.Cauchy(0., 1.).expand([self.J, in_dim]).to_event(2))
        #pyro.sample()

    # prior of Omega(5*4): location (5*4): 0.0; scale (5*4): 1.0
    #
    # posterior of Omega:  location (5*4): variational parameter; scale (5*4): variation parameter
    #
    # sample from posterior: N(location, scale)

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        r"""
        :param x: Tensor
            The input to the FirstLayer.
        :return: Tensor
            The output of the FirstLayer, which is $\phi(\Omega \times x)$.
        """
        hid = self.layer(x)
        mu = torch.cat((torch.sin(hid).cuda(), torch.cos(hid).cuda()), dim=-1) / torch.sqrt(torch.tensor(self.J).cuda())

        return mu

class FirstCauchyLayer(PyroModule):
    r"""
    The random feature-based GP is equivalent to a two-layer Bayesian neural network.
    The first layer refers to generate the random feature $\phi$.
    Cauchy kernel is better on complex dataset, not linearly separable data.
    Attributes
    ----------
    J: int
        The number of random feature.
    layer: PyroModule
        The first layer is based on nn.Linear where the weight($\Omega$) is defined by the PyroSample.
    """

    def __init__(
            self,
            in_dim: int = 1,
            hid_dim: int = 100,
            # num_layer: int = 1,
    ) -> None:
        """
        :param in_dim: int
            The input dimension.
        :param hid_dim: int
            The hidden state's dimension, which is 2J. The hidden state here refers to the output of
            the first layer and the input of the second layer.
        """
        super().__init__()

        self.J = hid_dim // 2
        self.layer = PyroModule[nn.Linear](in_dim, self.J, bias=False)
        # self.layer.weight = pyro.sample(f"Laplacian {num_layer}-th Omega",
        #                                 dist.Laplace(0., 1.).expand([self.J, in_dim]).to_event(2))
        self.layer.weight = PyroSample(dist.Laplace(0., 1.).expand([self.J, in_dim]).to_event(2))
        #pyro.sample()

    # prior of Omega(5*4): location (5*4): 0.0; scale (5*4): 1.0
    #
    # posterior of Omega:  location (5*4): variational parameter; scale (5*4): variation parameter
    #
    # sample from posterior: N(location, scale)

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        r"""
        :param x: Tensor
            The input to the FirstLayer.
        :return: Tensor
            The output of the FirstLayer, which is $\phi(\Omega \times x)$.
        """
        hid = self.layer(x)
        mu = torch.cat((torch.sin(hid).cuda(), torch.cos(hid).cuda()), dim=-1) / torch.sqrt(torch.tensor(self.J).cuda())

        return mu
