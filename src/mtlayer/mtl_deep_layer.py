import torch
import torch.nn as nn
from torch import Tensor
from pyro.nn import PyroModule
from src.dgp_rff.inner_layer import FirstLayer, SecondLayer, FirstLaplacianLayer, FirstCauchyLayer,SecondLayerNoBias
from src.dgp_rff.outer_layer import SingleGP, SingleCauchyGP, SingleLaplacianGP, SingleGPFix,SingleGPNoBias
from src.dgp_rff.ensemble_layer import EnsembleGP, FinalLayer


class DeepGPNoBias(PyroModule):
    def __init__(
            self,
            dim_list = None,
            J_list = None,
            #这里不会写变量类型 搞定了
    ) -> None:
        super().__init__()
        in_dim_list = dim_list[:-1]
        out_dim_list = dim_list[1:]

        assert min(in_dim_list) > 0 and min(out_dim_list) > 0 and min(J_list) > 0  # make sure the dimensions are valid

        # Define the PyroModule layer list
        layer_list = []
        for i in range(len(in_dim_list)):
            layer_list.append(SingleGPNoBias(in_dim_list[i], out_dim_list[i], J_list[i]))
            #还没写singleGPnobias
            # layer_list.append(SecondLayerNoBias(2 * J_list[i], out_dim_list[i],i))
        #layer_list = [FirstLayer(in_dim_list[i], 2 * J_list[i]), SecondLayer(2 * J_list[i], out_dim_list[i]) for i in range(len(in_dim_list))]
        #layer_list.append(PyroModule[nn.Linear](out_dim_list[-1], out_dim_list[-1], bias=True))
        print(layer_list)
        self.layers = PyroModule[torch.nn.ModuleList](layer_list)

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        """
        :param x: Tensor
            The input into the Single GP
        :return:
            The output of the Single GP
        """
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        mu = x

        return mu

class MtlDeepGP(PyroModule):
    def __init__(
            self,
            dim_list = None,
            dim1_list = None,
            dim2_list = None,
            J_list = None,
            J1_list = None,
            J2_list = None,
            #这里不会写变量类型 搞定了
    ) -> None:
        super().__init__()
        in_dim_list = dim_list[:-1]
        out_dim_list = dim_list[1:]

        assert min(in_dim_list) > 0 and min(out_dim_list) > 0 and min(J_list) > 0  # make sure the dimensions are valid

        # Define the PyroModule layer list
        layer_list = []
        for i in range(len(in_dim_list)):
            layer_list.append(SingleGP(in_dim_list[i], out_dim_list[i], J_list[i]))
        print(layer_list)
        self.layers = PyroModule[torch.nn.ModuleList](layer_list)

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        """
        :param x: Tensor
            The input into the Single GP
        :return:
            The output of the Single GP
        """
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        mu = x

        return mu


class MtlDeepGP(PyroModule):
    def __init__(self, dim_list=[1, 1, 1], dim1_list = [1,1], dim2_list = [1,1], J_list=[50, 10],J1_list=[5], J2_list=[5]):
        super().__init__()

        self.out_dim = dim_list[-1]
        self.GPcommon = DeepGPNoBias(dim_list, J_list)
        self.GP1 = DeepGPNoBias(dim_list=dim1_list, J_list=J1_list)
        self.GP2 = DeepGPNoBias(dim_list=dim2_list, J_list=J2_list)
        self.model.to('cuda')

    def forward(self, x1,x2, y=None):
        z1 = self.GPcommon(x1)
        z2 = self.GPcommon(x2)
        z = 1/2 * (z1 + z2)
        y1 = self.GP1(z)
        y2 = self.GP2(z)
        y = torch.cat((y1, y2), dim=1)
        mu = y

        return mu