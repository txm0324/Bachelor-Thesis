# for current TUGDA_rep
# pip uninstall torch torchvision torchaudio
# pip install torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# now: conda activate TUGDA_server 

# library for data analysis and table processing 
import pandas as pd
# library for numerical calculations and arrays
import numpy as np
# built-in Python module for random numbers
import random

## PyTorch:
# tourch is the main library for deep learning with PyTorch
import torch
# nn is a submodule of torch for neural networks (e.g. layers, loss functions)
from torch import nn
# DataLoader: enables data to be loaded in batches
# TensorDataset: enables features and labels to be easily combined in a dataset 
from torch.utils.data import DataLoader, TensorDataset

## PyTorch Lightning:
# call pytorch lightning functions

# lightweight PyTorch wrapper that helps you focus on the what (model architecture, training logic) rather than the how (logging, GPUs, checkpoints)
# lightning use structured classes (Trainer, Callback, etc.), automated training (backpropagation, optimizer, checkpointing), hardware management (automatic training on GPU/TPU)
import pytorch_lightning as pl

# enables user-defined actions during training (e.g. EarlyStopping, logging, custom hooks).
from pytorch_lightning import Callback
# Trainer: central class for training and evaluating models  
# seed_everything: sets randoms seeds for reproducibility 
from pytorch_lightning import Trainer, seed_everything

from scipy.stats import pearsonr # to calculate the correlation

# get list of 200 drugs to be used for training and prediction (here: 200 drugs)
folder = 'data/'
drug_list = pd.read_csv('{}/cl_y_test_o_k1.csv'.format(folder), index_col=0 )
drug_list = drug_list.columns

# 3-fold training and test data
# Features (X), Labels (Y)
train_data_report = {}
test_data_report = {}

# save test and training data for each fold
for k in range(1,4):
    train_data_report['x_k_fold{}'.format(k)] = pd.read_csv('{}/cl_x_train_o_k{}.csv'.format(folder, k), index_col=0)
    train_data_report['y_k_fold{}'.format(k)] = pd.read_csv('{}/cl_y_train_o_k{}.csv'.format(folder, k), index_col=0)
    
    test_data_report['x_k_fold{}'.format(k)] = pd.read_csv('{}/cl_x_test_o_k{}.csv'.format(folder, k), index_col=0)
    test_data_report['y_k_fold{}'.format(k)] = pd.read_csv('{}/cl_y_test_o_k{}.csv'.format(folder, k), index_col=0)


# Extension data (from Networks/Network_Construction.py and Pathway_Construction.ipynb)

# 1. Drug-Gene-Interaction
dgi_matrix_direct = pd.read_csv("./data/Targets/direct/direct_targets.csv", index_col=0).astype(np.float32)
dgi_matrix_indirect_02 = pd.read_csv("./data/Targets/indirect/indirect_targets_0.2.csv", index_col=0).astype(np.float32)
dgi_matrix_indirect_03 = pd.read_csv("./data/Targets/indirect/indirect_targets_0.3.csv", index_col=0).astype(np.float32)
dgi_matrix_indirect_04 = pd.read_csv("./data/Targets/indirect/indirect_targets_0.4.csv", index_col=0).astype(np.float32)
dgi_matrix_indirect_05 = pd.read_csv("./data/Targets/indirect/indirect_targets_0.5.csv", index_col=0).astype(np.float32)
dgi_matrix_indirect_06 = pd.read_csv("./data/Targets/indirect/indirect_targets_0.6.csv", index_col=0).astype(np.float32)
dgi_matrix_indirect_07 = pd.read_csv("./data/Targets/indirect/indirect_targets_0.7.csv", index_col=0).astype(np.float32)

# 2. Drug-Pathway-Interaction
pathway_matrix = pd.read_csv("./data/Pathways/drug_pathway_binary_matrix.csv", index_col=0).astype(np.float32)
pathway_matrix_count = pd.read_csv("./data/Pathways/gene_count_zscore.csv", index_col=0).astype(np.float32)
pathway_matrix_frequency = pd.read_csv("./data/Pathways/gene_frequency.csv", index_col=0).astype(np.float32)
pathway_matrix_weights = pd.read_csv("./data/Pathways/pathway_weights_zscore.csv", index_col=0).astype(np.float32)

# preprocessing file 
# gene and drug order should be the same!!!!
# here you habe to change with approach you want to use
dgi_matrix = dgi_matrix_indirect_05.T.fillna(0.0).astype(float).values

# Checks whether a GPU with CUDA support is available and recognized by PyTorch
# Automatically selects 'cuda' or 'cpu' so you don't have to manually adjust the code
# depending on whether you're running locally (CPU) or on a GPU-enabled server
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Automatically saves test metrics to a list after each test run
class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_test_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)

class tugda_mtl(pl.LightningModule):
    # Constructor __init__
    # Initializes the model with parameters, data and network architecture
    def __init__(self, params, train_data, y_train,
                 test_data, y_test, dgi_matrix=None
                ):
        super(tugda_mtl, self).__init__()
        
        # Hyperparameters are taken from params
        self.learning_rate = params['lr']
        self.batch_size = params['bs']
        self.mu = params['mu']
        self.lambda_ = params['lambda_']
        self.gamma = params['gamma']
        self.num_tasks = params['num_tasks']
        self.passes = params['passes']
        self.lambda_dgi = params['lambda_dgi']

        # Training and test data are saved 
        self.train_data = train_data
        self.y_train = y_train
        self.test_data = test_data
        self.y_test = y_test

        # dgi_tensor is stored in the module as a non-trainable tensor
        self.register_buffer('dgi_tensor', torch.FloatTensor(dgi_matrix) if dgi_matrix is not None else None)
        
        # Three main network components are defined: 

        input_dim = self.train_data.shape[1]
        
        # 1. extracts features from input data 
        # Feedforward-Netzwork: 
            # nn.Linear: full connected layer: Input --> Hidden Layer
            # nn.Dropout: randomly deactives neurons during training 
            # nn.ReLU(): Activation function: manes the network non-linear
        feature_extractor = [nn.Linear(input_dim, params['hidden_units_1']), 
                             nn.Dropout(p=params['dropout']),
                             nn.ReLU()]
        
        # to combine several layers into a single module (You don't need to write each layer in forward())
        # PyTorch processes: Input --> Linear Layer --> Dropout --> ReLU --> Output
        self.feature_extractor = nn.Sequential(*feature_extractor)

        # 2. Compresses the features into a latent space
        # takes the features from feature_extractor
        # Compresses them into a smaller, more meaningful space
        # create representative intermediate representation (latent basis) --> makes task predictions, attempts to reconstruct thee later using a decoder
        latent_basis =  [nn.Linear(params['hidden_units_1'], params['latent_space']),
                         nn.Dropout(p=params['dropout']),
                         nn.ReLU()]
        
        self.latent_basis = nn.Sequential(*latent_basis)
        
        # 3. task-specific weight matrix for predictions 
        # takes the latent representation and computes predictions for each drug
        self.S = nn.Linear(params['latent_space'], self.num_tasks)
        
        # decoder weights to back-project the latent space (for the autoencoder)
        A = [nn.Linear( self.num_tasks , params['latent_space']), nn.ReLU()]
        self.A = nn.Sequential(*A)
        
        # uncertainty (aleatoric)
        # estimates the uncertainty of the predictions 
        # stores the logarithmic variance for each drug (high variance means that the mdoel is very uncertain (high dispersion of predictions))
        self.log_vars = torch.zeros(self.num_tasks, requires_grad=True, device=device)
        
    def forward(self, input_data):
        x = self.feature_extractor(input_data) # Extract features from input file
        h = self.latent_basis(x) # Compresses the features into the latent space 
        preds = self.S(h) # Prediction for each drug
        h_hat = self.A(preds) # Reconstructs latent representation from prediction 
        return preds, h, h_hat

    def prepare_data(self):
        # TensorDataSet is a PyTourch class that combines input data and drug values, data is then stored as tensors 
        train_dataset = TensorDataset(torch.FloatTensor(self.train_data),
                                      torch.FloatTensor(self.y_train))

        test_dataset = TensorDataset(torch.FloatTensor(self.test_data),
                                     torch.FloatTensor(self.y_test))

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset


    # Dataloader is a PyTorch class that helps to create and iterates batches from training data 
    def train_dataloader(self):
        # shuffle = TRUE: trainig fata is randomly shuffled for each run to prevent overfitting 
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
    
    def test_dataloader(self):
        # batch size: entire dataset is transferred at once as a batch in order to evalute all data 
        # shuffle = FALSE: test data is not mixed, as the order of the data is irrelevant here 
        return DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False, num_workers=8)

    def configure_optimizers(self):
        # to update the model parameters so that the error/loss is minimized 
        params = ([p for p in self.parameters()] + [self.log_vars])
        # Adagrad: adjust learning rates for each parameter based on the previous gradients (parameters that are updated frequently are given a lower learning rate)
        optimizer = torch.optim.Adagrad(params, lr=self.learning_rate)
        return optimizer
    
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, 
                       optimizer_closure, second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # update params of the model based on the current gradient

        # based on gradients that were previously calculated during the back propagation 
        optimizer.step(closure=optimizer_closure)
        # gradient is accumlated by default (gradients are added to previous value) --> prevents that the gradient of different batches being added together 
        optimizer.zero_grad()
        
    def mse_ignore_nan(self, preds, labels):
        # creates an MSE loss function that does not average (redcution='none'), it returns the error per drug
        mse_loss = torch.nn.MSELoss(reduction='none')
        per_task_loss = torch.zeros(labels.size(1), device=device) # preprared to save the errors per drug

        for k in range(labels.size(1)):
            # calculate precision (1/variance) --> prediction for k is uncertain (large variance), the precision becomes small and the error is weighted less
            precision = torch.exp(-self.log_vars[k])

            # MSE is only calculated on data points where there is no NaN in the label 
            diff = mse_loss(preds[~torch.isnan(labels[:,k]), k], labels[~torch.isnan(labels[:,k]), k])
            # MES is weighted with the unvertainty: higher uncertainty --> less influence (“learned loss weighting”)
            per_task_loss[k] = torch.mean(precision * diff + self.log_vars[k])

        return torch.mean(per_task_loss[~torch.isnan(per_task_loss)]), per_task_loss
    
    def mse_ignore_nan_test(self, preds, labels):
        # this version create MSE loss function without uncertainty weighting --> error is averaged directly (standard MSE)
        mse_loss = torch.nn.MSELoss(reduction='mean')
        per_task_loss = torch.zeros(labels.size(1), device=device) # preprared to save the errors per drug
        
        for k in range(labels.size(1)):
            per_task_loss[k] = mse_loss(preds[~torch.isnan(labels[:,k]), k], labels[~torch.isnan(labels[:,k]), k])
            
        # return the average test error across all drug and the MSE per drug
        return torch.mean(per_task_loss[~torch.isnan(per_task_loss)]), per_task_loss 
    
    # autoencoder loss (MSE between two vectors: x (original latent representation from latent_basis), x_hat_ reconstructed version from decoder A(preds))
    def MSE_loss(self, x, x_hat):
        mse_loss = torch.nn.MSELoss()
        # Purpose: ,easures the reconstruction error of the autoencoder, small error indicates that the latent representation contains enough information to reconstruct itself from predictions
        return mse_loss(x, x_hat)
    
    # calculates the total loss for a batch 
    def forward_pass(self, fw_batch, batch_idx):
        
        x, y = fw_batch
        
        # Multiple predictions are made, Goal: estimation of uncertainty due to missing data
        preds_simulation = torch.torch.zeros(y.size(0),y.size(1), self.passes, device=device)
        for simulation in range(self.passes):
            
            preds, h, h_hat = self.forward(x)
            preds_simulation[:,:, simulation]=preds
        
        # average prediction
        preds_mean = torch.mean(preds_simulation, axis=2)
        # Variance over the simulations --> uncertainty due to missing data averaged over all samples
        preds_var = torch.var(preds_simulation, axis=2)
        # mean uncertainty per drug 
        total_unc = torch.mean(preds_var, axis=0)
        
        # prediction loss
        local_loss, task_loss = self.mse_ignore_nan(preds_mean, y)
        # autoencoder loss
        # measures how well the latent feature h can be reconstructed from predictions, gamma: weight for this loss 
        recon_loss = self.gamma * self.MSE_loss(h, h_hat)

        # loss for new data
        dgi_reg_loss = 0.0
        if self.dgi_tensor is not None: 

        # Example:
        # Genes: G1, G2, G3, G4 with Tasks: T1, T2
        # DGI: (G1,T1), (G2,T1), (G2,T2), (G3,T2) are pairs with interactions (1)
        # After the first layer we have a weight for each gene (mean of cell-line values):
        # G1: 0.8, G2: 0.5, G3: 0.2, G4: 0.1
        #
        # Penalty T1:
        # - G1: 0.8 --> (1 - 0.8)² = 0.04
        # - G2: 0.5 --> (1 - 0.5)² = 0.25
        # Mean of penalty: 0.145
        #
        # Penalty T2:
        # - G2: 0.5 --> (1 - 0.5)² = 0.25
        # - G3: 0.2 --> (1 - 0.2)² = 0.64
        # Mean of penalty: 0.445

            # Weight matrix of genes from the first layer
            W = self.feature_extractor[0].weight  # [hidden, genes]
            gene_importance = W.abs().mean(dim=0)  # [n_genes]

            for k in range(self.num_tasks):
                if k >= self.dgi_tensor.shape[1]:
                    continue
                dgi_mask = self.dgi_tensor[:, k] # [n_genes]
                if dgi_mask.sum() > 0: # dgi_mask: Boolean tensor that indicates which genes are important for task k according to DGI 
                    # Penalty: important genes should also have high weights
                    imp_selected = gene_importance[dgi_mask.bool()]
                    penalty = ((1.0 - imp_selected).clamp(min=0)) ** 2 # penalty not negative 
                    dgi_reg_loss += penalty.mean()
                
        dgi_reg_loss = self.lambda_dgi * dgi_reg_loss

        # loss weighting based on uncertainty and decoder structure 
        # a: weight per drug, depending one: decoder complexity (decoder A) and uncertainty due to missing data
        a = 1 + (total_unc + torch.sum(torch.abs(self.A[0].weight.T),1))
        loss_weight = ( a[~torch.isnan(task_loss)]  ) * task_loss[~torch.isnan(task_loss)] 
        loss_weight = torch.sum(loss_weight)

        # L1: sum of absolute vales of the weights in the matrix S, mu: how strong this penalty is 
        # Effects: many weightd are set to 0, the model learns which drug really benefit from each ohter and ignores irrelevnat connections
        l1_S = self.mu * self.S.weight.norm(1)
        L = self.latent_basis[0].weight.norm(2) + self.feature_extractor[0].weight.norm(2)
        # L2: square root of the sum of the squares of the weights, lambda_: regularization factor
        # Effect: penalizes large weight values 
        l2_L = self.lambda_ * L

        # total loss
        total_loss = loss_weight + recon_loss + l1_S + l2_L + dgi_reg_loss
        return total_loss, task_loss
    

    # per training_step:
    def training_step(self, train_batch, batch_idx):
        
        # performs simulations, autoencoder reconstruction and regularizations
        loss, task_loss = self.forward_pass(train_batch, batch_idx)
        
        # records the total loss + individual losses per task 
        logs = {'train_loss': loss, 'task_loss': task_loss}
        return {'loss': loss, 'log': logs}
    
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        
        # enable dropouts to remain active during testing in order to obtain multiple slightly different predictions (for uncertainty measurement)
        self.feature_extractor[1].train()
        self.latent_basis[1].train()
        
        # 3D-Tensor for storing predicitions (dimensions: Batch size, number of targets, number of simulations)
        preds_simulation = torch.torch.zeros(y.size(0),y.size(1), self.passes, device=device)
        # performs self.passes predictions with different noise
        for simulation in range(self.passes):
            
            seed = simulation
            # to reproduce predictions
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # the modell is called multiple times on same inputs x
            # because the seed activates different random factors (droupout, noise), the predicitions vary 
            preds, _, _  = self.forward(x)
            preds_simulation[:,:, simulation]=preds
        
        # Mean across all simulations
        preds_mean = torch.mean(preds_simulation, axis=2)
        # Error between average predicted value and the actual target values y 
        loss, task_losses_per_class = self.mse_ignore_nan_test(preds_mean, y)
        
        # disable dropouts
        self.feature_extractor[1].eval()
        self.latent_basis[1].eval()
        
        # Output 
        return {'test_loss': loss,
               'test_task_losses_per_class': task_losses_per_class.detach().cpu().numpy(),
               'test_preds': preds_mean.detach().cpu().numpy(),
                }

# best set of hyperparamters found on this dataset setting (GDSC)
net_params = {
 # tunned hyperparameters
 'hidden_units_1': 1024,
 'latent_space': 700,
 'lr': 0.001,
 'dropout': 0.1,
 'mu': 0.01,
 'lambda_': 0.001,
 'lambda_dgi': 0.001, # add regulation for new data
 'gamma': 0.0001,
 'bs': 300,
 'passes': 50,
 'num_tasks': 200,
 'epochs': 100}

# training and testing
all_predictions = []
all_y_true = []

metrics_callback = MetricsCallback()

for k in range(1, 4):
    print(f"\n--- Fold {k} ---\n")
    
    X_train = train_data_report[f'x_k_fold{k}'].values
    X_test = test_data_report[f'x_k_fold{k}'].values

    y_train = train_data_report[f'y_k_fold{k}'].values
    y_test = test_data_report[f'y_k_fold{k}'].values
    
    net_params['input_dim'] = X_train.shape[1]

    # PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=net_params['epochs'],
        gpus=1 if torch.cuda.is_available() else None, # trains with GPU, if available
        callbacks=[metrics_callback],
        deterministic=True, # same results for every run (reproducible)
        reload_dataloaders_every_epoch=True
    )

    # Seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # creates the model with current fold data, fit: starts the training
    seed_everything(seed)

    # Create and train model
    model = tugda_mtl(net_params, X_train, y_train, X_test, y_test, dgi_matrix=dgi_matrix)
    trainer.fit(model)
    results = trainer.test(model, verbose=False)

    # get predictions and save them
    predictions = results[0]['test_preds']  # [n_samples, n_tasks]
    all_predictions.append(predictions)
    all_y_true.append(y_test)


# === FROM HERE: AGGREGATION OVER ALL FOLDS ===

# combine all predictions and labels
all_predictions = np.concatenate(all_predictions, axis=0)  # [n_total_samples, n_tasks]
all_y_true = np.concatenate(all_y_true, axis=0)            # [n_total_samples, n_tasks]

# Test NaN
mask_total = ~np.isnan(all_y_true) & ~np.isnan(all_predictions)

# Metrics per task across all folds
num_tasks = all_y_true.shape[1]
task_mses = []
task_pearsons = []

for i in range(num_tasks):
    true_vals = all_y_true[:, i]
    pred_vals = all_predictions[:, i]
    mask = ~np.isnan(true_vals) & ~np.isnan(pred_vals)
    if np.sum(mask) > 0:
        mse = np.mean((pred_vals[mask] - true_vals[mask]) ** 2)
        corr, _ = pearsonr(true_vals[mask], pred_vals[mask])
    else:
        mse = np.nan
        corr = np.nan
    task_mses.append(mse)
    task_pearsons.append(corr)

print("\n--- MEAN METRICS PER TASK ACROSS ALL FOLDS ---")
print("Average MSE across tasks:", np.nanmean(task_mses))
print("Median MSE across tasks:", np.nanmedian(task_mses))
print("Average Pearson correlation across tasks:", np.nanmean(task_pearsons))
print("Median Pearson correlation across tasks:", np.nanmedian(task_pearsons))


# Save them as csv file
mean_df = pd.DataFrame({
    'Task': drug_list,
    'Mean_MSE': task_mses,
    'Mean_Pearson': task_pearsons
})
mean_df.to_csv("./results/MTL/mean_metrics_per_task_indirect_05.csv", index=False)