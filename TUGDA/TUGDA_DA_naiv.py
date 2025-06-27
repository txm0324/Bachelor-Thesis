# library for data analysis and table processing 
import pandas as pd
# library for numerical calculations and arrays
import numpy as np
# built-in Python module for random numbers
import random
# cycle creates an iterator that repeats a given seqeunce an infinite number of times
from itertools import cycle


## PyTorch:
# tourch is the main library for deep learning with PyTorch
import torch
# nn is a submodule of torch for neural networks (e.g. layers, loss functions)
from torch import nn
# DataLoader: enables data to be loaded in batches
# TensorDataset: enables features and labels to be easily combined in a dataset 
from torch.utils.data import DataLoader, TensorDataset
# to define manually forward and backward passes
from torch.autograd import Function

## PyTorch Lightning:
# call pytorch lightning functions

# lightweight PyTorch wrapper that helps you focus on the what (model architecture, training logic) rather than the how (logging, GPUs, checkpoints)
# lightning use structured classes (Trainer, Callback, etc.), automated training (backpropagation, optimizer, checkpointing), hardware management (automatic training on GPU/TPU)
import pytorch_lightning as pl

# enables user-defined actions during training (e.g. EarlyStopping, logging, custom hooks).
from pytorch_lightning import Callback
# seed_everything: sets randoms seeds for reproducibility 
from pytorch_lightning import seed_everything

# Checks whether a GPU with CUDA support is available and recognized by PyTorch
# Automatically selects 'cuda' or 'cpu' so you don't have to manually adjust the code
# depending on whether you're running locally (CPU) or on a GPU-enabled server
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# flip discriminator gradient during the backpropagation process 
# Purpose: Gradient Reversal Layer: a layer that does not change anything in the forward pass, but reverse gradients in the backward pass (models learns adversarially)
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x) # retunr the same tensor as input (no change)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha # Invert gradients and scale with alpha 

        return output, None # second return value is for alpha, which does not need a degree

# Automatically saves test metrics to a list after each test run
class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    # Constructor to initialize the metric 
    def __init__(self):
        super().__init__()
        self.metrics = []

    # called when the test process is complete, access trainer.callback_metrics: dictionary which containing all metrics logged by the model during testing
    def on_test_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)

    # called when the training process is complete, access trainer.callback_metrics: dictionary which containing all metrics logged by the model during training
    def on_train_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)

class tugda_da(pl.LightningModule):
    # Constructor __init__
    # Initializes the model with parameters, data and network architecture
    def __init__(self, params, train_data, y_train, x_target_unl,
                 test_data, y_test, y_test_da
                ):
        super(tugda_da, self).__init__()
        
        # Hyperparameters are taken from params
        self.learning_rate = params['lr']
        self.batch_size = params['bs_source']
        self.mu = params['mu']
        self.lambda_ = params['lambda_']
        self.gamma = params['gamma']
        self.lambda_disc = params['lambda_disc']
        self.bs_disc = params['bs_disc']
        self.n_epoch = params['n_epochs']
        self.passes = params['passes']
        self.num_tasks = params['num_tasks']

        # Training and test data are saved 
        self.train_data = train_data
        self.y_train = y_train
        self.x_target_unl = x_target_unl
        
        # Three main network components are defined: 

        # 1. extracts features from input data 
        # Feedforward-Netzwork: 
            # nn.Linear: full connected layer: Input --> Hidden Layer
            # nn.Dropout: randomly deactives neurons during training 
            # nn.ReLU(): Activation function: manes the network non-linear

        input_dim = self.train_data.shape[1]
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
         # to combine several layers into a single module (You don't need to write each layer in forward())
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
        
        # discriminator
        # Input: vector from the latent space, Output_ intermediate layer with n_units_disc
        domain_classifier = [nn.Linear(params["latent_space"], params["n_units_disc"]), 
                             nn.Dropout(p=params['dropout']), # prevents overfitting 
                             nn.ReLU(), # activation function, introduces non-linearity
                             nn.Linear(params["n_units_disc"], 1), # Final layer outputs a single value --> classification logit
                             nn.Sigmoid() # Converts the logit into a probability between 0 and 1 
                            ]
         # to combine several layers into a single module (You don't need to write each layer in forward())
        self.domain_classifier = nn.Sequential(*domain_classifier)
        
    def forward(self, input_data, alpha):
        x = self.feature_extractor(input_data) # Extract features from input file
        h = self.latent_basis(x) #  Compresses the features into the latent space 
        # AUC between 0 and 1, in just cases over 1, wrong after definition, that's i set an sigmoid function here 
        preds = torch.sigmoid(self.S(h)) # Prediction for each drug
        h_hat = self.A(preds) # Reconstructs latent representation from prediction 
        # domain classfier attempts to determine which domain (source/target) the reverse_feature originates from
        # However, since reverse_feature has reverse gradient flow, the encoder is trained to counteract this goal --> domaininvariant 
        reverse_feature = ReverseLayerF.apply(h, alpha)
        domain_output = self.domain_classifier(reverse_feature)
        return preds, h, h_hat, domain_output

    def prepare_data(self):
        # TensorDataSet is a PyTourch class that combines input data and drug values, data is then stored as tensors 
        train_dataset = TensorDataset(torch.FloatTensor(self.train_data),
                                      torch.FloatTensor(self.y_train))

        
        target_unl_dataset = TensorDataset(torch.FloatTensor(self.x_target_unl))
        
        self.train_dataset = train_dataset
        self.target_unl_dataset = target_unl_dataset

    # Dataloader is a PyTorch class that helps to create and iterates batches from training data 
    def train_dataloader(self):
        # shuffle = TRUE: trainig fata is randomly shuffled for each run to prevent overfitting 
        # dataloader1: loads the source data (trained, labeled dataset)
        dataloader1 = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        # dataloader2: loads the target data, unlabeled (unlabeled target domain)
        dataloader2 = DataLoader(self.target_unl_dataset, batch_size=self.bs_disc, shuffle=True)
        # stores the length of the smaller loader (used to know when a shared training epoch ends)
        self.len_dataloader = min(len(dataloader1), len(dataloader2))
        # zip: combines both dataloader (for each step you get: a batch of labeled source data and a batch of unlabeled target data)
        # cycle: ensures that the smaller data loader is repeated indefinitely if one ends earlier 
        return zip(dataloader1, cycle(dataloader2))

    def test_dataloader(self):
        # batch size: entire dataset is transferred at once as a batch in order to evalute all data 
        # shuffle = FALSE: test data is not mixed, as the order of the data is irrelevant here 
        dataloader = DataLoader(self.target_unl_dataset, batch_size=len(self.target_unl_dataset), shuffle=False)
        return dataloader

    def configure_optimizers(self):
        
        # Adagrad_ optimization algorithm that adjusts the learning rate for each parameter individually (based on the frequency of gradient)
        # Parameters that are updated frequently receive a smaller learning rate 
        # Parameters that are updated infrequently receive a large learning rate 
        opt_cla = torch.optim.Adagrad([
            {'params': self.feature_extractor.parameters()},
            {'params': self.latent_basis.parameters()},
            {'params': self.S.parameters()},
            {'params': self.A.parameters()},
            {'params': self.domain_classifier.parameters()},
            {'params': self.log_vars}
        ], lr=self.learning_rate)

        return opt_cla
    
    # Binary Cross-Entropy Loss
    def binary_classification_loss(self, preds, labels):
        # measures the difference between the % of the preds and the actual target values (labels)
        # BCE Loss=−[y⋅log(p)+(1−y)⋅log(1−p)] (y: label (0/1), probability predicted by the model that y=1)
        bin_loss = torch.nn.BCELoss()
        return bin_loss(preds, labels)

        
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
        per_task_loss = torch.zeros(labels.size(1), device=device) #  stores MSE for each column (e.g. per drug)
        per_sample_loss = torch.zeros(labels.size(0), device=device) # stores MSE for each row (e.g. per patient)

        
        # MSE only for entries where the label is not NaN
        for k in range(labels.size(1)):
            per_task_loss[k] = mse_loss(preds[~torch.isnan(labels[:,k]), k], labels[~torch.isnan(labels[:,k]), k])
        
        # per class loss
        for k in range(labels.size(0)):
            per_sample_loss[k] = mse_loss(preds[k, ~torch.isnan(labels[k,:])], labels[k, ~torch.isnan(labels[k, :])])
        
        
        return torch.mean(per_task_loss[~torch.isnan(per_task_loss)]), per_task_loss,per_sample_loss 

    # autoencoder loss (MSE between two vectors: x (original latent representation from latent_basis), x_hat_ reconstructed version from decoder A(preds))
    def MSE_loss(self, x, x_hat):
        mse_loss = torch.nn.MSELoss()
        # Purpose: ,easures the reconstruction error of the autoencoder, small error indicates that the latent representation contains enough information to reconstruct itself from predictions
        return mse_loss(x, x_hat)
    
     # calculates the total loss for a batch 
    def forward_pass(self, fw_batch, batch_idx):
        x, y = fw_batch[0]
        # unlabelled data (target)
        unl = fw_batch[1][0]
        
        # warm-up
        # Goal: make training more stable by gradually adjusting parameters (such as alpha) rather than changing them suddenly
        # p: current training progress, related to the batch index and epoch 
        p = float(batch_idx +  self.current_epoch * self.len_dataloader) / self.n_epoch / self.len_dataloader
        # alpha: sigmoid-like function controls a gradual change in alpha, starting at a low value (close to -1) and increasing to a higher value (close to +1) over the course of training
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
         # Multiple predictions are made, Goal: estimation of uncertainty due to missing data
        preds_simulation = torch.torch.zeros(y.size(0),y.size(1), self.passes, device=device)
        for simulation in range(self.passes):
            # the modell is called multiple times (forward) on same inputs x 
            # because the seed activates different random factors (droupout, noise), the predicitions vary 
            preds, h, h_hat, domain_out_source = self.forward(x, alpha)
            # stores the predictions for the current simulation in the matrix preds_simulation 
            preds_simulation[:,:, simulation]=preds

        # average prediction
        preds_mean = torch.mean(preds_simulation, axis=2)
        # Variance over the simulations --> uncertainty due to missing data averaged over all samples
        preds_var = torch.var(preds_simulation, axis=2)
        # mean uncertainty per drug 
        total_unc = torch.mean(preds_var, axis=0)


        m_loss, task_loss = self.mse_ignore_nan(preds_mean, y)
        recon_loss = self.gamma * self.MSE_loss(h, h_hat)

        # loss weighting based on uncertainty and decoder structure 
        # a: weight per drug, depending one: decoder complexity (decoder A) and uncertainty due to missing data
        a = 1 + (total_unc + torch.sum(torch.abs(self.A[0].weight.T),1))
        loss_weight = ( a[~torch.isnan(task_loss)] ) * task_loss[~torch.isnan(task_loss)]
        loss_weight = torch.sum(loss_weight)

        # L1: sum of absolute vales of the weights in the matrix S, mu: how strong this penalty is 
        # Effects: many weightd are set to 0, the model learns which drug really benefit from each ohter and ignores irrelevnat connections
        l1_S = self.mu * self.S.weight.norm(1)
        L = self.latent_basis[0].weight.norm(2) + self.feature_extractor[0].weight.norm(2)
        
        # L2: square root of the sum of the squares of the weights, lambda_: regularization factor
        # Effect: penalizes large weight values 
        l2_L = self.lambda_ * L

        # domain discriminator source
        # Source domain as class "0" 
        # A binary cross entropy loss is calculated
        zeros = torch.zeros(y.size(0), device=self.device)
        d_loss_source = self.binary_classification_loss(domain_out_source, zeros)
        
        # domain discriminator target
        # unl = unlabeled data from target-domain 
        # Target domain as class "1"
         # A binary cross entropy loss is calculated
        preds, h, h_hat, domain_out_target = self.forward(unl, alpha)
        ones = torch.ones(unl.size(0), device=self.device)
        d_loss_target = self.binary_classification_loss(domain_out_target, ones)
        
        # Combination of source and target domain errors 
        # Goal: discriminator should correctly seperate between source (0) and target (1)
        d_loss = d_loss_source + d_loss_target

        # total loss
        loss = loss_weight + recon_loss + l1_S + l2_L + (self.lambda_disc *d_loss)
        return loss, m_loss, d_loss

    # per training_step:
    def training_step(self, train_batch, batch_idx):

        # performs simulations, autoencoder reconstruction and regularizations
        loss, task_loss, disc_loss = self.forward_pass(train_batch, batch_idx)

        # records the total loss + source losses per task + Discriminator Loss
        logs = {'total_loss': loss, 'source_loss': task_loss, 'disc_loss': disc_loss}
        return {'loss': loss, 'log': logs}
    

    def test_step(self, test_batch, batch_idx):
    
        # Input data of the current test batch 
        x_unl = test_batch[0]

        # dropout on
        # enable dropouts to remain active during testing in order to obtain multiple slightly different predictions (for uncertainty measurement)
        self.feature_extractor[1].train()
        self.latent_basis[1].train()
        
        # TARGET STEPS
        # get model preds 
        # 3D-Tensor for storing predicitions (dimensions: Batch size of current batch, number of tasks, number of simulations)
        preds_simulation = torch.torch.zeros(x_unl.size(0),self.num_tasks, self.passes, device=device)
    
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
            preds, h_target_ls, _, _  = self.forward(x_unl, 0)
            preds_simulation[:,:, simulation]=preds
        
        # Mean across all simulations
        preds_mean = torch.mean(preds_simulation, axis=2)
        
        # disable dropouts: train() # resets dropout to train
        self.feature_extractor[1].eval()
        self.latent_basis[1].eval()
        
        # Output
        return {'preds': preds_mean.detach().cpu().numpy()
                }

# best set of hyperparameters found on validation settings;
net_params = {
 'hidden_units_1': 1500, # Number of neurons in the first hidden layer 
 'latent_space': 800, # size of latent_space (e.g. for embeddings or autoencoder components)
 'lr': 0.001, # Learning rate - how much the modul adjusts the weights at each step 
 'dropout': 0.1, # Droupout rate - percentage of neurons that are randomly deactivated to avoid overfitting 
 'mu': 1,
 'lambda_': 1,
 'gamma': 0.01,
 'n_units_disc': 500,# Number of neurons in the Discriminator-Network 
 'n_epochs': 50, # How often the model runs through the entire training data set
 'bs_disc': 64, # Batches size for discriminator (how many training examples are entered inot the discriminator per step)
 'bs_source': 300, # Batch Size - how many training examples are processed at once 
 'lambda_disc': 0.3,
 'num_tasks': 200, # Number of drug variables
 'passes': 50} # how often the models goes over the datra per epoch 

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


# training and testing
print(net_params)

metrics_callback = MetricsCallback()

# cell-line dataset;
gdsc_dataset = pd.read_csv('./data/GDSCDA_fpkm_AUC_all_drugs.zip', index_col=0)
# gene set range (first 1780 columns represent genes, from the 1781st column to the end represent data on drugs)
gene_list = gdsc_dataset.columns[0:1780]
drug_list = gdsc_dataset.columns[1780:]

# pdx novartis dataset;
pdx_dataset = pd.read_csv('./data/PDX_MTL_DA.csv', index_col=0)
drugs_pdx = pdx_dataset.columns[1780:]

# genes and drug are extracted from the gdsc_dataset
gene_list = gdsc_dataset.columns[0:1780]
drug_list = gdsc_dataset.columns[1780:]

# Input data (X_train) and the target values (y_train)
# X_train contains the genes (features), y_train contains the drug effects (target values)
X_train = gdsc_dataset[gene_list].values
y_train = gdsc_dataset[drug_list].values

# Extension 
dgi_matrix = pd.read_csv("./data/global_gene_interaction_matrix.csv", index_col=0).astype(np.float32)
pathway_matrix = pd.read_csv("./data/drug_pathway_binary_matrix.csv", index_col=0).astype(np.float32)

# Gene-Interaction:
X_train = extension_with_multiple_task_features(X_train, y_train, task_feature_matrices=[dgi_matrix], weights=[1])
# Pathway-Interaction:
# X_train = extension_with_multiple_task_features(X_train, y_train, task_feature_matrices=[pathway_matrix], weights=[1])
# Combination
# X_train = extension_with_multiple_task_features(X_train, y_train, task_feature_matrices=[dgi_matrix, pathway_matrix], weights=[0.5, 0.5])

# Test data also extracted from another dataset (pdx_dataset)
X_test = pdx_dataset[gene_list].values
y_test = pdx_dataset[drugs_pdx].values

X_train_unl = pdx_dataset[gene_list].values

# Trainer configuration
trainer = pl.Trainer(
    max_epochs=net_params['n_epochs'],
    gpus=1 if torch.cuda.is_available() else None, # trains with GPU, if available
    callbacks=[metrics_callback],
    deterministic=True,  # same results for every run (reproducible)
    reload_dataloaders_every_epoch=True
)

# same results for every run (reproducible)
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# creates the model with current fold data, fit: starts the training
seed_everything(seed)
model = tugda_da(net_params, X_train, y_train, X_train_unl,
              X_train, y_train, y_test)

trainer.fit(model)

 # use model after training or load weights
results = trainer.test(model)

# get prediction
preds = results[0]['preds']

# Save preds into a dataframe
drug_list = list(gdsc_dataset.columns[1780:])
assert preds.shape[1] == len(drug_list), f"Mismatch: {preds.shape[1]} vs {len(drug_list)}"

df_preds = pd.DataFrame(preds, columns=drug_list)
df_preds.index = [f"Patient_{i}" for i in range(399)]
df_genes_pdx = pdx_dataset.iloc[:, :1780].copy()
df_preds.index = df_genes_pdx.index

# Name for specific variant
df_preds.to_csv('./results/preds_AUC_naiv_gene_level.csv', index=True)
df_preds.to_csv('./results/preds_AUC_naiv_pathway_level.csv', index=True)
df_preds.to_csv('./results/preds_AUC_naiv_combination.csv', index=True)