import torch
import torch.nn as nn
from torch import Tensor
from pyro.nn import PyroModule
from src.dgp_rff.inner_layer import FirstLayer, SecondLayer, FirstLaplacianLayer, FirstCauchyLayer,SecondLayerNoBias
from src.dgp_rff.outer_layer import SingleGP, SingleCauchyGP, SingleLaplacianGP, SingleGPFix,SingleGPNoBias
from src.dgp_rff.ensemble_layer import EnsembleGP, FinalLayer


class DeepGP(PyroModule):

    def __init__(
            self,
            dim_list = None,
            J_list = None,
            init_w = None,
            #这里不会写变量类型 搞定了
    ) -> None:
        super().__init__()
        in_dim_list = dim_list[:-1]
        out_dim_list = dim_list[1:]

        # if in_dim_list is None:
        #     in_dim_list = [1, 1, 1]
        assert min(in_dim_list) > 0 and min(out_dim_list) > 0 and min(J_list) > 0  # make sure the dimensions are valid

        # Define the PyroModule layer list
        layer_list = []
        for i in range(len(in_dim_list)):
            layer_list.append(SingleGP(in_dim_list[i],out_dim_list[i], J_list[i],init_w))
            # layer_list.append(SecondLayer(2 * J_list[i], out_dim_list[i],i))
        #layer_list = [FirstLayer(in_dim_list[i], 2 * J_list[i]), SecondLayer(2 * J_list[i], out_dim_list[i]) for i in range(len(in_dim_list))]
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

class DeepGPFix(PyroModule):
    # deep gp with fixed features
    def __init__(
            self,
            dim_list = None,
            J_list = None,
            #这里不会写变量类型 搞定了
    ) -> None:
        super().__init__()
        in_dim_list = dim_list[:-1]
        out_dim_list = dim_list[1:]

        # if in_dim_list is None:
        #     in_dim_list = [1, 1, 1]
        assert min(in_dim_list) > 0 and min(out_dim_list) > 0 and min(J_list) > 0  # make sure the dimensions are valid

        # Define the PyroModule layer list
        layer_list = []
        for i in range(len(in_dim_list)):
            layer_list.append(SingleGPFix(in_dim_list[i],out_dim_list[i], 2 * J_list[i]))
            # layer_list.append(SecondLayer(2 * J_list[i], out_dim_list[i],i))
        #layer_list = [FirstLayer(in_dim_list[i], 2 * J_list[i]), SecondLayer(2 * J_list[i], out_dim_list[i]) for i in range(len(in_dim_list))]
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

class DeepEnsembleGP(PyroModule):
    def __init__(
            self,
            dim_list = None,
            J_list = None,
            nkernel = 2,
            #这里不会写变量类型 搞定了
    ) -> None:
        super().__init__()
        in_dim_list = [dim_list[0]] + [2 * x for x in dim_list[1:-1]]
        out_dim_list = dim_list[1:]

        # if in_dim_list is None:
        #     in_dim_list = [1, 1, 1]
        print(in_dim_list)
        print(out_dim_list)
        print(J_list)
        assert min(in_dim_list) > 0 and min(out_dim_list) > 0 and min(J_list) > 0  # make sure the dimensions are valid

        # Define the PyroModule layer list
        layer_list = []
        for i in range(len(in_dim_list)):
            layer_list.append(EnsembleGP(in_dim_list[i], out_dim_list[i], J_list[i],nkernel))
        layer_list.append(FinalLayer(out_dim_list[-1], out_dim_list[-1],nkernel))
        print(layer_list)
        # layer_list.append(SecondLayerNoBias(2 * out_dim_list[-1], out_dim_list[-1],i+1))
        #加上num_layer和参数名字 所有文件所有层
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

