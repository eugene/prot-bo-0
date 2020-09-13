import torch
import pandas as pd
import numpy as np
import math, time, sys
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.test_functions import Branin
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement, NoisyExpectedImprovement
from botorch.optim import optimize_acqf
from scipy.stats import spearmanr
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior

torch.manual_seed(0)

class SimpleCustomGP(ExactGP, GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API
    
    def __init__(self, train_X, train_Y):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=RBFKernel(ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X)
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

covar_module = ScaleKernel(
    MaternKernel(
        nu=2.5,
        ard_num_dims=1, #train_x.shape[-1],
        batch_shape=torch.Size(),
        lengthscale_prior=GammaPrior(3.0, 6.0), # GammaPrior(3.0, 6.0),
    ),
    batch_shape=torch.Size(),
    outputscale_prior=GammaPrior(1.0, 0.15)     # GammaPrior(2.0, 0.15),
)

def train(dim = 8, beta = 1e-2):
    global X, dfo, df, promising

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double


    dfo     = pd.read_csv(f"data/df-{dim}.csv").set_index('mutation')
    df      = dfo.copy()
    mu, std = df.mean(axis = 0), df.std(axis=0)

    df -= df.mean(axis = 0)
    df /= std

    print("Spearman cor:", spearmanr(df.value, df.model)[0])

    X, y, mutant_ids = df.drop(columns=['value']), df.value, []
    min_max = list(map(list, list(zip(X.min().tolist(), X.max().tolist()))))
    bounds = torch.tensor(min_max, dtype=dtype, device=device)

    muts, vals = [], []
    def blackbox(x):
        ret = []
        for row in x:
            x_ = row.to('cpu')
            euclid = (x_ - X.values).pow(2).sum(1).pow(0.5)
            idx = euclid.argmin()
            ret.append(y[idx])
            muts.append(df.iloc[idx.item()].name)
            vals.append(y[idx]*std.value + mu.value)

        return torch.tensor(ret)

    #promising = X.sort_values('model', ascending=False)[0:100]
    promising = X.sample(100)
    train_x = torch.tensor(promising.values, dtype=dtype, device=device)
    train_obj = blackbox(train_x).unsqueeze(-1)

    for i in range(256):
        # model = FixedNoiseGP(train_x, train_obj, train_Yvar=torch.full_like(train_obj, 0.2))
        model = SingleTaskGP(train_X=train_x, train_Y=train_obj)#, covar_module=covar_module) #SimpleCustomGP(train_x, train_obj) #
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        model = model.to(device)
        mll = mll.to(device)
        
        fit_gpytorch_model(mll)
        
        best_value = train_obj.max()
        
        # EI = ExpectedImprovement(model=model, best_f=best_value-0.2, maximize = True)
        # POI = ProbabilityOfImprovement(model=model, best_f=best_value-0.1)
        UCB = UpperConfidenceBound(model=model, beta=beta)
        # NE  = NoisyExpectedImprovement(model=model, X_observed=train_x, num_fantasies=32, maximize = True)

        new_point_analytic, _ = optimize_acqf(
            acq_function=UCB, # EI, POI
            # acq_function=IUC,
            bounds=bounds.T,
            q=1,
            num_restarts=20,    #20,
            raw_samples=20*50,  #20*50,
            options={},
        )

        train_x   = torch.cat((train_x, new_point_analytic), 0)
        train_obj = torch.cat((train_obj, blackbox(new_point_analytic).unsqueeze(-1)), 0)

        star = "\t***YES**" if vals[-1] >= 0.15 else ''
        print("Iteration", i, "len(muts)", len(muts), "\t", muts[-1], "\t", vals[-1], star)
        
    res = pd.DataFrame(data={'mutant': muts, 'value':vals})
    res.to_csv(f"results/res-ucb3-dim:{dim}-beta-{str(beta).replace('.',',')}.csv")
    return res

if __name__ == "__main__":    
    for beta in [1e3*0.5, 1e3, 1e4*0.5, 1e4, 1e5*0.5, 1e5, 1e6*0.5, 1e6]:
        train(dim=sys.argv[1], beta=beta)
