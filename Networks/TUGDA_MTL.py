from Train_Test_Data import train_datasets, test_datasets
from Train_Test_Data import drug_list

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.data import DataLoader
from torch.utils.data import TensorDataset
import random
import numpy as np

#call pytorch lightning functions
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning import Trainer, seed_everything

from torch_geometric.data import DataLoader as GeoDataLoader

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TopKPooling, global_max_pool as gmp, global_mean_pool as gap

from torch_geometric.data import Batch

class GNNEncoder(nn.Module):
    def __init__(self, feature_size=3, embedding_size=512, output_dim=256):
        super(GNNEncoder, self).__init__()

        # Block 1
        self.conv1 = GATConv(feature_size, embedding_size, heads=3, dropout=0.6)
        self.head_transform1 = nn.Linear(embedding_size * 3, embedding_size)
        self.bn1 = nn.BatchNorm1d(embedding_size)
        self.pool1 = TopKPooling(embedding_size, ratio=0.8)

        # Block 2
        self.conv2 = GATConv(embedding_size, embedding_size, heads=3, dropout=0.6)
        self.head_transform2 = nn.Linear(embedding_size * 3, embedding_size)
        self.bn2 = nn.BatchNorm1d(embedding_size)
        self.pool2 = TopKPooling(embedding_size, ratio=0.5)

        # Final projection
        self.linear1 = nn.Linear(embedding_size * 2, 512)
        self.linear2 = nn.Linear(512, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Block 1
        x = self.conv1(x, edge_index)
        x = F.relu(self.head_transform1(x))
        x = self.bn1(x)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Block 2
        x = self.conv2(x, edge_index)
        x = F.relu(self.head_transform2(x))
        x = self.bn2(x)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Combine
        x = x1 + x2

        # Final layers
        x = self.linear1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        graph_embedding = self.linear2(x)

        return graph_embedding

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_test_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

class tugda_mtl_gnn(pl.LightningModule):
    def __init__(self, params, train_dataset, y_train, test_dataset, y_test):
        super(tugda_mtl_gnn, self).__init__()
        
        # Hyperparameters
        self.learning_rate = params['lr']
        self.batch_size = params['bs']
        self.mu = params['mu']
        self.lambda_ = params['lambda_']
        self.gamma = params['gamma']
        self.num_tasks = params['num_tasks']
        self.passes = params['passes']

        # Data
        self.train_data = train_dataset
        self.y_train = y_train
        self.test_data = test_dataset
        self.y_test = y_test
        
        # GNN Encoder (your existing implementation)
        self.gnn_encoder = GNNEncoder(
            feature_size=params.get('feature_size', 3),
            embedding_size=params.get('embedding_size', 128),
            output_dim=params.get('gnn_output_dim', 64)
        )

        # Projection layer
        projection = [
            nn.Linear(params.get('gnn_output_dim', 256), params['hidden_units_1']),
            nn.ReLU()
        ]
        self.projection = nn.Sequential(*projection)
        
        # Latent basis
        latent_basis = [
            nn.Linear(params['hidden_units_1'], params['latent_space']),
            nn.Dropout(p=params['dropout']),
            nn.ReLU()
        ]
        self.latent_basis = nn.Sequential(*latent_basis)
        
        # Task-specific weights
        self.S = nn.Linear(params['latent_space'], self.num_tasks)
        
        # Decoder weights
        A = [nn.Linear(self.num_tasks, params['latent_space']), nn.ReLU()]
        self.A = nn.Sequential(*A)
        
        # Uncertainty (aleatoric)
        self.log_vars = torch.zeros(self.num_tasks, requires_grad=True, device=device)
        
    def forward(self, drug_graphs):
        """
        drug_graphs: Liste von Data-Objekten (je ein Graph pro Drug)
        returns:
            preds: [num_tasks] - Vorhersage für jede Task/Drug
            h:     [latent_dim] - Latente Repräsentation
            h_hat: [num_tasks] - Rekonstruierte Tasks durch Autoencoder
        """
        embeddings = []

        for graph in drug_graphs:
            # Encode each drug-specific graph
            graph_emb = self.gnn_encoder(graph)
            embeddings.append(graph_emb)

        # Stack and average embeddings (or concatenate if you want task-specific features)
        x = torch.stack(embeddings).mean(dim=0).squeeze(0)  # shape: [gnn_output_dim]

        # Project to hidden space
        x = self.projection(x)  # shape: [hidden_units_1]

        # Latent basis
        h = self.latent_basis(x)  # shape: [latent_space]

        # Task predictions
        preds = self.S(h)  # shape: [num_tasks]

        # Decoder (Autoencoder regularization)
        h_hat = self.A(preds)  # shape: [latent_space]

        return preds, h, h_hat

    def prepare_data(self):
        self.train_dataset = self.train_data
        self.test_dataset = self.test_data

    def train_dataloader(self):
        from torch_geometric.loader import DataLoader
        return DataLoader(self.train_data, 
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=0)

    def test_dataloader(self):
        from torch_geometric.loader import DataLoader
        return DataLoader(self.test_data,
                        batch_size=self.batch_size,
                        shuffle=False,
                        num_workers=0)
    
    def configure_optimizers(self):
        params = ([p for p in self.parameters()] + [self.log_vars])
        optimizer = torch.optim.Adagrad(params, lr=self.learning_rate)
        return optimizer
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """
        Proper implementation of optimizer_step that handles the closure
        """
        # Call the closure to compute the loss and gradients
        optimizer_closure()
        
        # Perform the actual optimization step
        optimizer.step()
        
        # Zero gradients
        optimizer.zero_grad()
        
    def mse_ignore_nan(self, preds, labels):
        """
        preds: [batch_size, num_tasks]
        labels: [batch_size, num_tasks]
        """
        print("preds shape:", preds.shape)
        print("labels shape:", labels.shape)

        # assert preds.shape == labels.shape, f"Shape mismatch: {preds.shape} vs {labels.shape}"

        if preds.dim() == 1:
            preds = preds.unsqueeze(0)  # [1, num_tasks]
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)  # [1, num_tasks]

        mse_loss = torch.nn.MSELoss(reduction='none')
        per_task_loss = torch.zeros(labels.size(1), device=device)

        for k in range(labels.size(1)):
            mask = ~torch.isnan(labels[:, k])
            if not mask.any():
                per_task_loss[k] = torch.nan
                continue  # Überspringe Tasks ohne Label
            diff = mse_loss(preds[mask, k], labels[mask, k])
            precision = torch.exp(-self.log_vars[k])
            per_task_loss[k] = torch.mean(precision * diff + self.log_vars[k])

        return torch.mean(per_task_loss[~torch.isnan(per_task_loss)]), per_task_loss
    
    def mse_ignore_nan_test(self, preds, labels):
        """
        preds: [batch_size, num_tasks] - Vorhersage des Modells
        labels: [batch_size, num_tasks] - Wahre Werte mit möglichen NaNs
        returns:
            mean_loss: Durchschnittlicher Loss über alle Tasks
            per_task_loss: Loss pro Task/Drug
        """

        if preds.dim() == 1:
            preds = preds.unsqueeze(0)  # [1, num_tasks]
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)  # [1, num_tasks]


        mse_loss = torch.nn.MSELoss(reduction='mean')
        batch_size, num_tasks = labels.shape
        per_task_loss = torch.zeros(num_tasks, device=device)

        for k in range(num_tasks):
            # Maske für nicht-NaN Werte im aktuellen Task
            mask = ~torch.isnan(labels[:, k])
            
            if not mask.any():
                per_task_loss[k] = torch.nan  # Überspringe Tasks ohne Label
                continue
            
            # Berechne Loss nur für Samples mit gültigem Label
            pred_task = preds[mask, k]
            label_task = labels[mask, k]
            per_task_loss[k] = mse_loss(pred_task, label_task)
        
        # Mittelwert über alle Tasks (ignoriere NaNs)
        mean_loss = torch.mean(per_task_loss[~torch.isnan(per_task_loss)])
        
        return mean_loss, per_task_loss
    
    def MSE_loss(self, x, x_hat):
        mse_loss = torch.nn.MSELoss()
        return mse_loss(x, x_hat)
    
    def forward_pass(self, batch, batch_idx):
        """
        batch = (drug_graphs_list, labels_tensor)
        drug_graphs_list = [Data, Data, ...]  # je ein Graph pro Drug
        labels_tensor = [batch_size, num_tasks]
        """

        print("Starting forward_pass")
        drug_graphs_list, labels_tensor = batch
        print(f"Number of drug graphs: {len(drug_graphs_list)}")

        labels_tensor = labels_tensor.squeeze(1)
        print("labels_tensor shape:", labels_tensor.shape)

        # Wir erwarten: drug_graphs_list ist eine Liste von Graphen einer Zelllinie
        # Also: Iteriere über jeden Drug-Graphen und encode ihn
        preds, h, h_hat = self.forward(drug_graphs_list)

        preds = preds.unsqueeze(0)
        h = h.unsqueeze(0)
        h_hat = h_hat.unsqueeze(0)

        # Monte Carlo Simulation (optional)
        preds_simulation = [preds.clone() for _ in range(self.passes)]
        preds_simulation = torch.stack(preds_simulation)  # [passes, 1, num_tasks]

        preds_mean = preds_simulation.mean(dim=0)  # [1, num_tasks]
        preds_var = preds_simulation.var(dim=0)    # [1, num_tasks]
        total_unc = preds_var.mean(dim=0)          # [num_tasks]

        print("preds_mean.shape:", preds_mean.shape)
        print("preds_var.shape:", preds_var.shape)
        print("total_unc.shape:", total_unc.shape)

        # Loss Calculation
        local_loss, task_loss = self.mse_ignore_nan(preds_mean, labels_tensor)
        recon_loss = self.gamma * self.MSE_loss(h, h_hat)

        total_unc = preds_var.mean(dim=0)
        a = 1 + (total_unc + torch.sum(torch.abs(self.A[0].weight.T), dim=1))
        loss_weight = (a[~torch.isnan(task_loss)] * task_loss[~torch.isnan(task_loss)]).sum()

        l1_S = self.mu * self.S.weight.norm(1)
        l2_L = self.lambda_ * (
            self.gnn_encoder.conv1.lin.weight.norm(2) +
            self.gnn_encoder.conv2.lin.weight.norm(2)
        )

        total_loss = loss_weight + recon_loss + l1_S + l2_L

        return total_loss, task_loss
    
    def training_step(self, batch, batch_idx):
        # batch is a PyG Data object, no need to unpack
        loss, task_loss = self.forward_pass(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    
    def test_step(self, test_batch, batch_idx):
        cell_line_graphs, labels = test_batch
        labels = labels.squeeze(1)

        preds_simulation = []

        for simulation in range(self.passes):
            pl.seed_everything(simulation)
            preds_per_pass = []

            for graph in cell_line_graphs:
                pred, _, _ = self.forward([graph])  # Wir rufen forward mit einer einzelnen Drug-Liste auf
                preds_per_pass.append(pred)

            preds_simulation.append(torch.cat(preds_per_pass, dim=0))

        preds_mean = torch.mean(torch.stack(preds_simulation), dim=0).unsqueeze(0)
        loss, task_losses = self.mse_ignore_nan_test(preds_mean, labels.unsqueeze(0))

        return {
            'test_loss': loss,
            'test_preds': preds_mean.detach().cpu().numpy(),
            'test_task_losses_per_class': task_losses.detach().cpu().numpy()
        }
    
""" Test mit einem Batch """    
net_params_test = {
    'lr': 0.001,
    'bs': 1,
    'mu': 0.01,
    'lambda_': 0.001,
    'gamma': 0.0001,
    'num_tasks': 200,
    'passes': 5,
    'epochs': 1,
    'feature_size': 3,
    'embedding_size': 64,
    'gnn_output_dim': 128,
    'hidden_units_1': 128,
    'latent_space': 128,
    'dropout': 0.1
}

# # Wähle ein Dataset (z.B. Fold 1)
# train_dataset = train_datasets[0]
# print("train dataset")
# test_dataset = test_datasets[0]
# print("test dataset")

# # Labels als numpy arrays oder Tensoren
# y_train = train_dataset.labels_df.values
# print("y train")
# y_test = test_dataset.labels_df.values
# print("y test")

# # Erstelle ein minimales Modell mit Dummy-Parametern
# model = tugda_mtl_gnn(
#     params=net_params_test,
#     train_dataset=train_dataset,
#     y_train=y_train,
#     test_dataset=test_dataset,
#     y_test=y_test
# )
# print("Modell")

# # Simuliere einen Batch aus einem Datensatz
# batch = train_dataset[0]  # ein Datensatz: (drug_graphs, label)
# drug_graphs, label = batch
# print("Batch")
# print("Batch type:", type(batch))  # class tuple

# # Führe forward() manuell aus
# with torch.no_grad():
#     preds, h, h_hat = model(drug_graphs)

# print("Preds:", preds.shape)   # Sollte [200] sein
# print("Latent Space:", h.shape)      # z.B. [700]
# print("Reconstruct Latent Space:", h_hat.shape)  # z.B. [700]

# # print("Testing with gradients...")
# # try:
# #     preds, h, h_hat = model(drug_graphs)
# #     loss = preds.mean()  # Simple loss for testing
# #     loss.backward()
# #     print("Backward pass successful")
# # except RuntimeError as e:
# #     print(f"Error during backward: {str(e)}")
# #     if "CUDA out of memory" in str(e):
# #         print("Out of memory error - reduce model size or batch size")

# # Führe forward_pass mit diesem Batch aus
# loss, task_loss = model.forward_pass((drug_graphs, label.unsqueeze(0)), 0)
# print("Loss:", loss.item())
# loss.backward()
# print("Backward-Pass OK")
# """ Ende """

# Erstes Training 

# Seeds setzen für Reproduzierbarkeit
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
pl.seed_everything(seed)

# Wähle ein Dataset (z.B. Fold 1)
train_dataset = train_datasets[0]
print("train dataset")
test_dataset = test_datasets[0]
print("test dataset")

# Labels als numpy arrays oder Tensoren
y_train = train_dataset.labels_df.values
print("y train")
y_test = test_dataset.labels_df.values
print("y test")

# Modell initialisieren
model = tugda_mtl_gnn(
    params=net_params_test,
    train_dataset=train_dataset,
    y_train=y_train,
    test_dataset=test_dataset,
    y_test=y_test
)
print("Modell initialisiert.")

metrics_callback = MetricsCallback()

trainer = pl.Trainer(
    max_epochs=net_params_test['epochs'],
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,  # Oder 2, wenn du Multi-GPU brauchst
    callbacks=[metrics_callback],
    log_every_n_steps=5,
    deterministic=True
)

trainer.fit(model)
results = trainer.test(model)

error_per_task = results[0]['test_task_losses_per_class']
error_mtl_nn_results = np.concatenate((
    np.array(drug_list, ndmin=2).T,
    np.array(error_per_task, ndmin=2).T
), axis=1)

print("Error per task:")
print(error_mtl_nn_results)



# #best set of hyperparamters found on this dataset setting (GDSC)
# net_params = {
#  #tunned hyperparameters
#  'hidden_units_1': 512,
#  'latent_space': 256,
#  'lr': 0.001,
#  'dropout': 0.1,
#  'mu': 0.01,
#  'lambda_': 0.001,
#  'gamma': 0.0001,
#  'bs': 8,
#  'passes': 10,
#  'num_tasks': 200,
#  'epochs': 10}

# pcorr_list = []

# metrics_callback = MetricsCallback()

# for k in range(1, 4):  # 3-fold
    
#     print(f"\n--- Fold {k} ---")
    
#     # Original-Daten laden
#     full_train_dataset = train_datasets[k - 1]
#     full_test_dataset = test_datasets[k - 1]

#     full_y_train = full_train_dataset.labels_df.values
#     full_y_test = full_test_dataset.labels_df.values

#     X_train = full_train_dataset
#     X_test = full_test_dataset
#     y_train = full_y_train
#     y_test = full_y_test

#     # Trainer erstellen
#     trainer = pl.Trainer(
#         max_epochs=net_params['epochs'],
#         accelerator='gpu' if torch.cuda.is_available() else 'cpu',
#         devices=2,
#         callbacks=[metrics_callback],
#         log_every_n_steps=5,
#         deterministic=True,
#         # reload_dataloaders_every_epoch=True
#     )

#     # Seeds setzen
#     seed = 42
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     pl.seed_everything(seed)

#     # Modell erstellen
#     model = tugda_mtl_gnn(
#         params=net_params,
#         train_dataset=full_train_dataset,
#         y_train=y_train,
#         test_dataset=full_test_dataset,
#         y_test=y_test
#     )

#     # Training starten
#     trainer.fit(model)

#     # Testen
#     results = trainer.test(model)

#     # Fehler pro Drug speichern
#     error_per_task = results[0]['test_task_losses_per_class']
#     error_mtl_nn_results = np.concatenate((
#         np.array(drug_list, ndmin=2).T,
#         np.array(error_per_task, ndmin=2).T
#     ), axis=1)

#     print("Error per task:")
#     print(error_mtl_nn_results)