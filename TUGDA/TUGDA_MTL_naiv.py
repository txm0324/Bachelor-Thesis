# for current TUGDA_rep
# pip uninstall torch torchvision torchaudio
# pip install torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# now: conda activate TUGDA_server 

import pandas as pd
import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
#call pytorch lightning functions
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning import Trainer, seed_everything

import torch.nn.functional as F

# for analysis
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from scipy.stats import skew


# get list of drugs to be trained and predicted
folder = 'data/'
drug_list = pd.read_csv('{}/cl_y_test_o_k1.csv'.format(folder), index_col=0)
drug_list = drug_list.columns

# get data for biological input: 
# dgi_matrix = pd.read_csv('{}/interaktionsmatrix_all.csv'.format(folder), index_col=0).values
# pathway_matrix =  pd.read_csv('{}/drug_pathway_matrix_all.csv'.format(folder), index_col=0).values


# 3-fold training and test data;
train_data_report = {}
test_data_report = {}

for k in range(1,4):
    train_data_report['x_k_fold{}'.format(k)] = pd.read_csv('{}/cl_x_train_o_k{}.csv'.format(folder, k), index_col=0)
    train_data_report['y_k_fold{}'.format(k)] = pd.read_csv('{}/cl_y_train_o_k{}.csv'.format(folder, k), index_col=0)
    
    test_data_report['x_k_fold{}'.format(k)] = pd.read_csv('{}/cl_x_test_o_k{}.csv'.format(folder, k), index_col=0)
    test_data_report['y_k_fold{}'.format(k)] = pd.read_csv('{}/cl_y_test_o_k{}.csv'.format(folder, k), index_col=0)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_test_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)

class tugda_mtl(pl.LightningModule):
    def __init__(self, params, train_data, y_train,
                 test_data, y_test
                ):
        super(tugda_mtl, self).__init__()
        
        self.learning_rate = params['lr']
        self.batch_size = params['bs']
        self.mu = params['mu']
        self.lambda_ = params['lambda_']
        self.gamma = params['gamma']
        self.num_tasks = params['num_tasks']
        self.passes = params['passes']

        self.lambda_graph = params.get('lambda_graph', 0.1) 
        self.lambda_pathway_graph = params.get('lambda_pathway_graph', 0.1)


        self.train_data = train_data
        self.y_train = y_train
        self.test_data = test_data
        self.y_test = y_test

        input_dim = self.train_data.shape[1]
        
        feature_extractor = [nn.Linear(input_dim, params['hidden_units_1']), 
                             nn.Dropout(p=params['dropout']),
                             nn.ReLU()]
        
        self.feature_extractor = nn.Sequential(*feature_extractor)

        latent_basis =  [nn.Linear(params['hidden_units_1'], params['latent_space']),
                         nn.Dropout(p=params['dropout']),
                         nn.ReLU()]
        
        self.latent_basis = nn.Sequential(*latent_basis)
        
        #task-specific weights
        self.S = nn.Linear(params['latent_space'], self.num_tasks)
        
        #decoder weights
        A = [nn.Linear( self.num_tasks , params['latent_space']), nn.ReLU()]
        self.A = nn.Sequential(*A)
        
        #uncertainty (aleatoric)
        self.log_vars = torch.zeros(self.num_tasks, requires_grad=True, device=device)

    def forward(self, input_data):

        #### Normalisierung ####
        # Effekt: Jeder sample vektor so skaliert, dass Gesamtnorm = 1 ist, große Ausreißer gedämpft und alle Features gleichwertig behandelt
        # x = F.normalize(input_data, p=2, dim=1)
        

        #### Feature Combiner ####
        x = self.feature_extractor(input_data)
        h = self.latent_basis(x)
        preds = self.S(h)
        h_hat = self.A(preds)
        return preds, h, h_hat

    def prepare_data(self):
        train_dataset = TensorDataset(torch.FloatTensor(self.train_data),
                                      torch.FloatTensor(self.y_train))

        test_dataset = TensorDataset(torch.FloatTensor(self.test_data),
                                     torch.FloatTensor(self.y_test))

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False, num_workers=4)

    def configure_optimizers(self):
        params = ([p for p in self.parameters()] + [self.log_vars])
        optimizer = torch.optim.Adagrad(params, lr=self.learning_rate)
        return optimizer
    
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, 
                       second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # update params
        optimizer.step()
        optimizer.zero_grad()
        
    def mse_ignore_nan(self, preds, labels):
        mse_loss = torch.nn.MSELoss(reduction='none')
        per_task_loss = torch.zeros(labels.size(1), device=device)

        for k in range(labels.size(1)):
            precision = torch.exp(-self.log_vars[k])
            diff = mse_loss(preds[~torch.isnan(labels[:,k]), k], labels[~torch.isnan(labels[:,k]), k])
            per_task_loss[k] = torch.mean(precision * diff + self.log_vars[k])

        return torch.mean(per_task_loss[~torch.isnan(per_task_loss)]), per_task_loss
    
    def mse_ignore_nan_test(self, preds, labels):
        mse_loss = torch.nn.MSELoss(reduction='mean')
        per_task_loss = torch.zeros(labels.size(1), device=device)
        
        for k in range(labels.size(1)):
            per_task_loss[k] = mse_loss(preds[~torch.isnan(labels[:,k]), k], labels[~torch.isnan(labels[:,k]), k])
            
        return torch.mean(per_task_loss[~torch.isnan(per_task_loss)]), per_task_loss 
    
    #autoencoder loss
    def MSE_loss(self, x, x_hat):
        mse_loss = torch.nn.MSELoss()
        return mse_loss(x, x_hat)
    
    def forward_pass(self, fw_batch, batch_idx):
        
        x, y = fw_batch
        
        preds_simulation = torch.torch.zeros(y.size(0),y.size(1), self.passes, device=device)

        for simulation in range(self.passes):
            
            preds, h, h_hat = self.forward(x)
            preds_simulation[:,:, simulation]=preds
        
        preds_mean = torch.mean(preds_simulation, axis=2)
        preds_var = torch.var(preds_simulation, axis=2)
        total_unc = torch.mean(preds_var, axis=0)
            
        #prediction loss
        local_loss, task_loss = self.mse_ignore_nan(preds_mean, y)
        #autoencoder loss
        recon_loss = self.gamma * self.MSE_loss(h, h_hat)
        
        a = 1 + (total_unc + torch.sum(torch.abs(self.A[0].weight.T),1))
        loss_weight = ( a[~torch.isnan(task_loss)]  ) * task_loss[~torch.isnan(task_loss)] 
        loss_weight = torch.sum(loss_weight)
        l1_S = self.mu * self.S.weight.norm(1)
        L = self.latent_basis[0].weight.norm(2) + self.feature_extractor[0].weight.norm(2)
        l2_L = self.lambda_ * L

        #total loss
        total_loss = loss_weight + recon_loss + l1_S + l2_L
        return total_loss, task_loss
        
    def training_step(self, train_batch, batch_idx):
        
        loss, task_loss = self.forward_pass(train_batch, batch_idx)
        
        logs = {'train_loss': loss, 'task_loss': task_loss}
        return {'loss': loss, 'log': logs}
    
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        
        #enable dropouts
        self.feature_extractor[1].train()
        self.latent_basis[1].train()
        
        preds_simulation = torch.torch.zeros(y.size(0),y.size(1), self.passes, device=device)
        for simulation in range(self.passes):
            
            seed = simulation
            #to reproduce predictions
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            preds, _, _  = self.forward(x)
            preds_simulation[:,:, simulation]=preds
        
        preds_mean = torch.mean(preds_simulation, axis=2)
        loss, task_losses_per_class = self.mse_ignore_nan_test(preds_mean, y)
        
        #disable dropouts
        self.feature_extractor[1].eval()
        self.latent_basis[1].eval()
        
        return {'test_loss': loss,
               'test_task_losses_per_class': task_losses_per_class.detach().cpu().numpy(),
               'test_preds': preds_mean.detach().cpu().numpy(),
                }

## for Drug-Gen-Interactions (DGI), Drug-Pathway-Interactions (DPI) and Protein-Protein-Interactions (PPI)

def extension_with_multiple_task_features(X, y, task_feature_matrices, weights=None):
    """
    Erweiterung der Feature-Matrix X mit DGI-Vektoren auf Basis von Labels in y.

    Inputs:
    - X: np.array [N_samples (Zellinien) x n_genes], Genexpression

    X_train: (536, 1780) = [N_samples (Zellinien von 2 Folds) x n_genes]
    X_test: (269, 1780) = [N_samples (Zellinien von 1 Fold) x n_genes]

    - y: np.array [N_samples (Zellinien) x n_tasks (drugs)], Arzneimittelantworten (z. B. IC50, AUC)

    Y_train: (536, 200) = [N_samples (Zellinien von 2 Folds) x n_tasks (200 drugs)]
    Y_test: (269, 200) = [N_samples (Zellinien von 1 Fold) x n_tasks (200 drugs)]

    - dgi_matrix: torch.Tensor [n_tasks x n_genes], Drug-Gene-Interaktionen (0/1 oder normiert)

    dgi_matrix [n_tasks (200 drugs) x n_genes (1780)]
    
    Output:
    - X_extension: np.array [N_samples x (n_genes + n_genes)]

    X_train_extension: (536, 3560) = [N_samples (Zellinien von 2 Folds) + 2* n_genes]
    X_test_extension: (269, 3560) = [N_samples (Zellinien von 1 Folds) + 2* n_genes]

    """
    if weights is None:
        n_weights = len(task_feature_matrices)
        weights = [1.0 / n_weights for _ in range(n_weights)]
    elif len(weights) != len(task_feature_matrices):
        raise ValueError("The number of weights must match the number of matrices")

    X_ext = []

    for i in range(X.shape[0]):
        task_indices = np.where(~np.isnan(y[i]))[0]
        if len(task_indices) == 0:
            raise ValueError(f"Sample {i} does not have a valid label assignment.")
        task_idx = task_indices[0]

        # Sammle alle Feature-Vektoren & gewichte diese für diesen Task 
        feature_vecs = [
            weight * matrix.iloc[task_idx].values
            for weight, matrix in zip(weights, task_feature_matrices)
        ]

        # Kombiniere mit ursprünglichem X
        x_aug = np.concatenate([X[i]] + feature_vecs)
        X_ext.append(x_aug)

    return np.stack(X_ext)

# best set of hyperparamters found on this dataset setting (GDSC)
net_params = {
 # tunned hyperparameters
 'hidden_units_1': 1024,
 'latent_space': 700,
 'lr': 0.001,
 'dropout': 0.1,
 'mu': 0.01,
 'lambda_': 0.001,
 'gamma': 0.0001,
 'bs': 300,
 'passes': 50,
 'num_tasks': 200,
 'epochs': 100}

# training and testing
error_list = []
pcorr_list = []

metrics_callback = MetricsCallback()

dgi_matrix = pd.read_csv("./data/global_gene_interaction_matrix.csv", index_col=0).astype(np.float32)
pathway_matrix = pd.read_csv("./data/drug_pathway_binary_matrix.csv", index_col=0).astype(np.float32)

for k in range(1,4):
    
    X_train = train_data_report['x_k_fold{}'.format(k)].values
    X_test = test_data_report['x_k_fold{}'.format(k)].values

    y_train = train_data_report['y_k_fold{}'.format(k)].values
    y_test = test_data_report['y_k_fold{}'.format(k)].values

    # Extension 
    # Gene-Interaction:
    X_train = extension_with_multiple_task_features(X_train, y_train, task_feature_matrices=[dgi_matrix], weights=[1])
    X_test = extension_with_multiple_task_features(X_test, y_test, task_feature_matrices=[dgi_matrix], weights=[1])
    # Pathway-Interaction:
    # X_train = extension_with_multiple_task_features(X_train, y_train, task_feature_matrices=[pathway_matrix], weights=[1])
    # X_test = extension_with_multiple_task_features(X_test, y_test, task_feature_matrices=[pathway_matrix], weights=[1])
    # Combination
    # X_train = extension_with_multiple_task_features(X_train, y_train, task_feature_matrices=[dgi_matrix, pathway_matrix], weights=[0.5,0.5])
    # X_test = extension_with_multiple_task_features(X_test, y_test, task_feature_matrices=[dgi_matrix, pathway_matrix], weights=[0.5,0.5])

    # Input dimensions
    net_params['input_dim'] = X_train.shape[1]

    trainer = pl.Trainer(
        max_epochs=net_params['epochs'],
        gpus=1 if torch.cuda.is_available() else None,
        callbacks=[metrics_callback],
        deterministic=True,
        reload_dataloaders_every_epoch=True
    )


    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    seed_everything(seed)
    model = tugda_mtl(net_params, X_train, y_train,
                  X_test, y_test)
    
    trainer.fit(model)
    
    # use model after training or load weights
    results = trainer.test(model)

    # get error per drug
    error_mtl_nn_results = np.concatenate((np.array(drug_list, ndmin=2).T,
                            np.array(results[0]['test_task_losses_per_class'], ndmin=2).T), axis=1)


## Analysis by MSE and Pearson Correlation 

# MSE
predictions = results[0]['test_preds']  # [n_samples, n_tasks]
task_mses = results[0]['test_task_losses_per_class']

print("Durchschnittlicher MSE über Tasks:", np.nanmean(task_mses))
print("Median-MSE über Tasks:", np.nanmedian(task_mses))

# Save as csv file 
df_errors = pd.DataFrame({'MSE_gene': task_mses}, index=drug_list)
print(df_errors.head())
df_errors.to_csv("task_mses_gene.csv", index_label='Drug')

# df_errors = pd.DataFrame({'MSE_pathway': task_mses}, index=drug_list)
# print(df_errors.head())
# df_errors.to_csv("task_mses_pathway.csv", index_label='Drug')

# df_errors = pd.DataFrame({'MSE_combination': task_mses}, index=drug_list)
# print(df_errors.head())
# df_errors.to_csv("task_mses_combination.csv", index_label='Drug')


# Pearson Correlation 
num_tasks = y_test.shape[1]  # Number of Task
pearson_corrs = []

for i in range(num_tasks):
    true_vals = y_test[:, i]
    pred_vals = predictions[:, i]
    
    # NaN-Werte aussortieren, falls vorhanden
    mask = ~np.isnan(true_vals) & ~np.isnan(pred_vals)
    if np.sum(mask) > 0:
        corr, _ = pearsonr(true_vals[mask], pred_vals[mask])
        pearson_corrs.append(corr)
    else:
        pearson_corrs.append(np.nan)

pearson_corrs = np.array(pearson_corrs)

print("Median Pearson Correlation:", np.nanmedian(pearson_corrs))

# Save as csv file 
df_pearson = pd.DataFrame({'corr_gene': pearson_corrs}, index=drug_list)
df_pearson.to_csv("task_pearson_corrs_gene.csv", index_label='Drug')

# df_pearson = pd.DataFrame({'corr_pathway': pearson_corrs}, index=drug_list)
# df_pearson.to_csv("task_pearson_corrs_pathway.csv", index_label='Drug')

# df_pearson = pd.DataFrame({'corr_combination': pearson_corrs}, index=drug_list)
# df_pearson.to_csv("task_pearson_corrs_combination.csv", index_label='Drug')

'''
np.savez_compressed(
    'result_new_feature_combiner_L2.npz',
    predictions=predictions,
    task_mses=task_mses,
    median_mse=median_mse,
    pearson_corrs=pearson_corrs
)

data = np.load('results.npz')

predictions = data['predictions']
task_mses = data['task_mses']
median_mse = data['median_mse']
pearson_corrs = data['pearson_corrs']

'''