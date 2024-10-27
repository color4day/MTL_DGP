def add_layer(oldmodel,new_dim,new_J):
    #old model has l layers
    #new model has l+1 layers
    #first get parameter for old model
    #pass param to new model
    #prevent update of newmodel
    old_dim_list =
    old_J_list =
    l = get_num_layer(mymodel)
    #old model: 3 omegalayer 3 W layer
    new_dim_list = old_dim_list.insert(-2,new_dim)
    new_J_list = old_J_list.append(new_J)
    newmodel = ModelDGP(new_dim_list,new_J_list)
mymodel = ModelDGP(dim_list=[x_train.shape[1],10,20,y_train.shape[1]], J_list=[50,10,20])

def getvalue(
        mymodel,
        ensemble = False
):
    #this model has num_layer layers, every layer is an ensemble layer.
    #In each ensemble layer, get omegas using layers_kernelname[0].omega
    #get weight and bias using layer.weight/bias.squeeze()
    num_layer = len(mymodel.model.layers)

    for i in range(num_layer-1):
        print(i,"th (ensemble) layer:")
        print(i,"th RBF Omega = ", mymodel.model.layers[i].layers_RBF[0].layer.weight.T)#Omega
        print(i,"th RBF Weight = ", mymodel.model.layers[i].layers_RBF[1].layer.weight.T)#Weight
        print(i,"th Cauchy Omega = ", mymodel.model.layers[i].layers_Cauchy[0].layer.weight.T)#Omega
        print(i,"th Cauchy Weight = ", mymodel.model.layers[i].layers_Cauchy[1].layer.weight.T)#Weight
    if ensemble:
        print(num_layer-1, "Final Layer Weight = ", mymodel.model.layers[-1].layer.weight.T)

class ModelDGP(PyroModule):
    def __init__(self, dim_list=[1, 1, 1], J_list=[50, 10]):
        super().__init__()

        self.out_dim = dim_list[-1]
        self.model = DeepGP(dim_list, J_list)
        self.model.to('cuda')

    def forward(self, x, y=None):
        mu = self.model(x).squeeze()  # 10000*6

        # batch shape | event shapt
        # 10000       |
        # scale = torch.FloatTensor([1e-8]*self.out_dim).cuda()
        scale = pyro.sample("sigma",
                            dist.Gamma(torch.tensor(0.5, device='cuda'), torch.tensor(1.0, device='cuda'))).expand(
            self.out_dim)  # Infer the response noise

        # Sampling model
        with pyro.plate("data", x.shape[0]):  # x.shape[0]=10000
            # obs = xxx("obs", mu, obs=y)
            obs = pyro.sample("obs", dist.MultivariateNormal(mu.cuda(), torch.diag(scale * scale).cuda()), obs=y)

        #         f1: phi(Omega x)W (+ epsilon1)
        #         f2: phi(Omega f1)W (+ epsilon2)

        #         f2 + epsilon ~ N(0, Sigma)

        return mu