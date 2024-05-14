from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.autograd import Variable

from data.data_reader import TSPReader
from utils.plot import plot_tsp, plot_predictions
from model.model import ResidualGatedGCNModel
from utils.train import train_one_epoch, test, update_learning_rate

# testing

num_nodes = 10
num_neighbors = -1  # when set to -1, it considers all the connections instead of k nearest neighbors
base_dir = Path(__file__).parent.resolve().as_posix()
train_filepath = f"{base_dir}/data/tsp/tsp{num_nodes}_train_concorde.txt"

dataset = TSPReader(num_nodes, num_neighbors, 1, train_filepath)
batch = dataset.test_batch()

#
idx = 0
f = plt.figure(figsize=(5, 5))
a = f.add_subplot(111)
plot_tsp(a, batch.nodes_coord[idx], batch.edges[idx], batch.edges_values[idx], batch.edges_target[idx])

#
num_nodes = 10 #@param # Could also be 10, 20, or 30!
num_neighbors = -1 # Could increase it!
train_filepath = f"{base_dir}/data/tsp/tsp{num_nodes}_train_concorde.txt"
hidden_dim = 50 #@param
num_layers = 3 #@param
mlp_layers = 2 #@param
learning_rate = 0.01 #@param
max_epochs = 40 #@param
batches_per_epoch = 1000

variables = {'train_filepath': f'{base_dir}/data/tsp/tsp{num_nodes}_train_concorde.txt',
             'val_filepath': f'{base_dir}/data/tsp/tsp{num_nodes}_val_concorde.txt',
             'test_filepath': f'{base_dir}/data/tsp/tsp{num_nodes}_test_concorde.txt',
             'num_nodes': num_nodes,
             'num_neighbors': num_neighbors,
             'node_dim': 2 ,
             'voc_nodes_in': 2,
             'voc_nodes_out': 2,
             'voc_edges_in': 3,
             'voc_edges_out': 2,
             'hidden_dim': hidden_dim,
             'num_layers': num_layers,
             'mlp_layers': mlp_layers,
             'aggregation': 'mean',
             'max_epochs': max_epochs,
             'val_every': 5,
             'test_every': 5,
             'batches_per_epoch': batches_per_epoch,
             'accumulation_steps': 1,
             'learning_rate': learning_rate,
             'decay_rate': 1.01
             }

net = nn.DataParallel(ResidualGatedGCNModel(variables, torch.FloatTensor, torch.LongTensor))
net.cpu()

# Compute number of network parameters
nb_param = 0
for param in net.parameters():
    nb_param += np.prod(list(param.data.size()))
print('Number of parameters:', nb_param)

#
####
num_nodes = variables['num_nodes']
num_neighbors = variables['num_neighbors']
batches_per_epoch = 1 # variables['batches_per_epoch']
val_filepath = variables['val_filepath']
test_filepath = variables['test_filepath']

# Initially set loss class weights as None
edge_cw = None

# Initialize running data
running_loss = 0.0
running_nb_data = 0

dtypeLong = torch.LongTensor
dtypeFloat = torch.FloatTensor

# Convert batch to torch Variables
x_edges = Variable(torch.LongTensor(batch.edges).type(dtypeLong), requires_grad=False)
x_edges_values = Variable(torch.FloatTensor(batch.edges_values).type(dtypeFloat), requires_grad=False)
x_nodes = Variable(torch.LongTensor(batch.nodes).type(dtypeLong), requires_grad=False)
x_nodes_coord = Variable(torch.FloatTensor(batch.nodes_coord).type(dtypeFloat), requires_grad=False)
y_edges = Variable(torch.LongTensor(batch.edges_target).type(dtypeLong), requires_grad=False)
y_nodes = Variable(torch.LongTensor(batch.nodes_target).type(dtypeLong), requires_grad=False)

# Compute class weights
edge_labels = y_edges.cpu().numpy().flatten()
edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)

# Forward and backward pass
y_preds, loss = net.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw)
loss = loss.mean()
loss.backward()

## Train 
# Define optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=variables["learning_rate"])
val_loss_old = None
train_losses = []
val_losses = []
test_losses = []

for epoch in range(variables["max_epochs"]):

    # Train
    train_time, train_loss = train_one_epoch(net, optimizer, variables, TSPReader)

    # Print metrics
    train_losses.append(train_loss)

    print(f"Epoch: {epoch}, Train Loss: {train_loss}")

    if epoch % variables["val_every"] == 0 or epoch == variables["max_epochs"]-1:

        # Validate
        val_time, val_loss = test(net, variables, TSPReader, mode='val')
        val_losses.append(val_loss)
        print(f"Epoch: {epoch}, Val Loss; {val_loss}")

        # Update learning rate
        if val_loss_old != None and val_loss > 0.99 * val_loss_old:
            variables["learning_rate"] /= variables["decay_rate"]
            optimizer = update_learning_rate(optimizer, variables["learning_rate"])

        val_loss_old = val_loss  # Update old validation loss

    if epoch % variables["test_every"] == 0 or epoch == variables["max_epochs"]-1:

        # Test
        test_time, test_loss = test(net, variables, TSPReader, mode='test')
        test_losses.append(test_loss)
        print(f"Epoch: {epoch}, Test Loss; {test_loss}\n")

#
net.eval()

num_samples = 10
num_nodes = variables['num_nodes']
num_neighbors = variables['num_neighbors']
test_filepath = variables['test_filepath']
dataset = iter(TSPReader(num_nodes, num_neighbors, 1, test_filepath))


x_edges = []
x_edges_values = []
x_nodes = []
x_nodes_coord = []
y_edges = []
y_nodes = []
y_preds = []

with torch.no_grad():
    for i in range(num_samples):
        sample = next(dataset)
        # Convert batch to torch Variables
        x_edges.append(Variable(torch.LongTensor(sample.edges).type(dtypeLong), requires_grad=False))
        x_edges_values.append(Variable(torch.FloatTensor(sample.edges_values).type(dtypeFloat), requires_grad=False))
        x_nodes.append(Variable(torch.LongTensor(sample.nodes).type(dtypeLong), requires_grad=False))
        x_nodes_coord.append(Variable(torch.FloatTensor(sample.nodes_coord).type(dtypeFloat), requires_grad=False))
        y_edges.append(Variable(torch.LongTensor(sample.edges_target).type(dtypeLong), requires_grad=False))
        y_nodes.append(Variable(torch.LongTensor(sample.nodes_target).type(dtypeLong), requires_grad=False))

        # Compute class weights
        edge_labels = (y_edges[-1].cpu().numpy().flatten())
        edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)

        # Forward pass
        y_pred, loss = net.forward(x_edges[-1], x_edges_values[-1], x_nodes[-1], x_nodes_coord[-1], y_edges[-1], edge_cw)
        y_preds.append(y_pred)


y_preds = torch.squeeze(torch.stack(y_preds))
# Plot prediction visualizations
plot_predictions(x_nodes_coord, x_edges, x_edges_values, y_edges, y_preds, num_plots=num_samples)