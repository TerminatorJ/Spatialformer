import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from datasets import load_from_disk
import os
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
p_path = current_file_path.parents[2]
script_path = os.path.join(p_path, "scripts")
sys.path.append(script_path)
# import pdb; pdb.set_trace()
from get_embeddings import sample_dataset
from train import *

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
# Define the MLP model
# class SimpleMLP(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(SimpleMLP, self).__init__()
#         self.layer1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.layer2 = nn.Linear(hidden_size, output_size)
    
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.relu(x)
#         x = self.layer2(x)
#         return x
    
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.5):
        super(SimpleMLP, self).__init__()
        
        # Define layers: Linear, BatchNorm, ReLU, Dropout
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Apply layers with Batch Normalization and Dropout included
        x = self.layer1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


def standardization(data):
    # Calculate mean and std from training data
    # import pdb; pdb.set_trace()
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # Standardize the data
    new_data = (data - mean) / std
    return new_data



def initialize(train_X_path, train_label_path, test_X_path, test_label_path, val_X_path, val_label_path, batch_size = 64, input_size = 128):
    # Load data
    # import pdb; pdb.set_trace()
    train_data = np.load(train_X_path)  # Load your train data
    valid_data = np.load(val_X_path)  # Load your validation data
    test_data = np.load(test_X_path)    # Load your test data

    train_data = standardization(train_data)
    valid_data = standardization(valid_data)
    test_data = standardization(test_data)

    # Load labels
    train_labels = np.load(train_label_path)
    valid_labels = np.load(val_label_path)
    test_labels = np.load(test_label_path)
    # import pdb; pdb.set_trace()
    # Encode labels
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    valid_labels_encoded = label_encoder.transform(valid_labels)
    test_labels_encoded = label_encoder.transform(test_labels)

    # Convert to torch tensors
    X_train = torch.tensor(train_data, dtype=torch.float32)
    y_train = torch.tensor(train_labels_encoded, dtype=torch.long)

    X_valid = torch.tensor(valid_data, dtype=torch.float32)
    y_valid = torch.tensor(valid_labels_encoded, dtype=torch.long)

    X_test = torch.tensor(test_data, dtype=torch.float32)
    y_test = torch.tensor(test_labels_encoded, dtype=torch.long)

    # Create DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    input_size = input_size
    hidden_size = 256
    output_size = len(np.unique(train_labels))  # Number of classes
    model = SimpleMLP(input_size, hidden_size, output_size)

    return train_loader, test_loader, valid_loader, model




def train_with_early_stopping(model, train_loader, valid_loader, epoch=10, patience=2, n_tasks = None, objective = None, lr = 0.0001, monitor = "train_loss"):
    # Set up training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    valid_losses = []

    best_loss = float('inf')
    patience_counter = 0

    num_epochs = epoch
    for epoch in range(num_epochs):
        model.train()  # Ensure model is in training mode
        epoch_loss = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()  # Accumulate loss

        # Calculate average losses
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()  # Switch to evaluation mode
        valid_loss = 0
        with torch.no_grad():
            for X_val, y_val in valid_loader:
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                valid_loss += loss.item()

        avg_valid_loss = valid_loss / len(valid_loader)
        valid_losses.append(avg_valid_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}')
        if monitor == "val_loss":
            # Early stopping check
            if avg_valid_loss < best_loss:
                best_loss = avg_valid_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        elif monitor == "train_loss":
            # Early stopping check
            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break

    # Plotting the training and validation curves
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(valid_losses)+1), valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Curve with Early Stopping')
    plt.legend()
    plt.savefig(f"/home/sxr280/Spatialformer/downstream/cell_types_nich_annotation/figures/training_valid_curve_{n_tasks}_{objective}_{current_time}.png", dpi=300)
    plt.show()

    return model
# Evaluation function with metrics
def evaluate_metrics(model, loader, n_tasks, objective = None):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    np.save(f"/home/sxr280/Spatialformer/downstream/cell_types_nich_annotation/data/all_preds_{n_tasks}_{objective}_{current_time}.npy", np.array(all_preds))
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return precision, recall, f1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='probe model for cell type annotation')
    parser.add_argument('--n_tasks', type = int, default=2, help='The number of tasks that is used to pre-train the model')
    parser.add_argument('--objective', type = str, default=None, help='You should define this parameter when n_tasks = 1 with two options: exp and spatial')
    parser.add_argument('--epoch', type = int, default=1000, help='How many epoch you want to train the probe network')
    parser.add_argument('--patience', type = int, default=10, help='The patience you want to stand for the loss that not going down')
    parser.add_argument('--lr', type = float, default=0.0001, help='The learning rate of the model training')
    parser.add_argument('--batch_size', type = int, default=64, help='The number of samples for each step')
    parser.add_argument('--input_size', type = int, default=128, help='The hidden dim of the model output')
    # parser.add_argument('--mini_batch', action = 'store_true', help = 'whether use minibatch')

    args = parser.parse_args()

    if args.n_tasks == 2:
        train_X_path = '/home/sxr280/Spatialformer/data/train_embedding_True_False_8_0.1_2_None_20241005_145010.npy'
        train_label_path = '/home/sxr280/Spatialformer/data/train_nichelabels_True_False_8_0.1_2_None_20241005_145010.npy'
        test_X_path = '/home/sxr280/Spatialformer/data/test_embedding_True_False_8_0.1_2_None_20241005_145010.npy'
        test_label_path = '/home/sxr280/Spatialformer/data/test_nichelabels_True_False_8_0.1_2_None_20241005_145010.npy'
        val_X_path = '/home/sxr280/Spatialformer/data/val_embedding_True_False_8_0.1_2_None_20241005_145010.npy'
        val_label_path = '/home/sxr280/Spatialformer/data/val_nichelabels_True_False_8_0.1_2_None_20241005_145010.npy'
    if args.n_tasks == 1:
        if args.objective == "spatial":
            train_X_path = '/home/sxr280/Spatialformer/data/train_embedding_True_False_8_0.1_1_spatial.npy'
            train_label_path = '/home/sxr280/Spatialformer/data/train_labels_True_False_8_0.1_1_spatial.npy'
            test_X_path = '/home/sxr280/Spatialformer/data/test_embedding_True_False_8_0.1_1_spatial.npy'
            test_label_path = '/home/sxr280/Spatialformer/data/test_labels_True_False_8_0.1_1_spatial.npy'
            val_X_path = '/home/sxr280/Spatialformer/data/val_embedding_True_False_8_0.1_1_spatial.npy'
            val_label_path = '/home/sxr280/Spatialformer/data/val_labels_True_False_8_0.1_1_spatial.npy'
        elif args.objective == "exp":
            train_X_path = '/home/sxr280/Spatialformer/data/train_embedding_True_False_8_0.1_1_exp.npy'
            train_label_path = '/home/sxr280/Spatialformer/data/train_labels_True_False_8_0.1_1_exp.npy'
            test_X_path = '/home/sxr280/Spatialformer/data/test_embedding_True_False_8_0.1_1_exp.npy'
            test_label_path = '/home/sxr280/Spatialformer/data/test_labels_True_False_8_0.1_1_exp.npy'
            val_X_path = '/home/sxr280/Spatialformer/data/val_embedding_True_False_8_0.1_1_exp.npy'
            val_label_path = '/home/sxr280/Spatialformer/data/val_labels_True_False_8_0.1_1_exp.npy'
        elif args.objective == "mt":
            train_X_path = '/home/sxr280/Spatialformer/data/train_embedding_True_False_8_0.1_1_mt_20240907_163220.npy'
            train_label_path = '/home/sxr280/Spatialformer/data/train_labels_True_False_8_0.1_1_mt_20240907_163220.npy'
            test_X_path = '/home/sxr280/Spatialformer/data/test_embedding_True_False_8_0.1_1_mt_20240907_163220.npy'
            test_label_path = '/home/sxr280/Spatialformer/data/test_labels_True_False_8_0.1_1_mt_20240907_163220.npy'
            val_X_path = '/home/sxr280/Spatialformer/data/val_embedding_True_False_8_0.1_1_mt_20240907_163220.npy'
            val_label_path = '/home/sxr280/Spatialformer/data/val_labels_True_False_8_0.1_1_mt_20240907_163220.npy'
        elif args.objective == "baseline":
            train_X_path = '/home/sxr280/Spatialformer/data/train_embedding_True_False_8_0.1_1_baseline.npy'
            train_label_path = '/home/sxr280/Spatialformer/data/train_labels_True_False_8_0.1_1_baseline.npy'
            test_X_path = '/home/sxr280/Spatialformer/data/test_embedding_True_False_8_0.1_1_baseline.npy'
            test_label_path = '/home/sxr280/Spatialformer/data/test_labels_True_False_8_0.1_1_baseline.npy'
            val_X_path = '/home/sxr280/Spatialformer/data/val_embedding_True_False_8_0.1_1_baseline.npy'
            val_label_path = '/home/sxr280/Spatialformer/data/val_labels_True_False_8_0.1_1_baseline.npy'
        elif args.objective == "scFoundataion":
            train_X_path = '/home/sxr280/scFoundation/model/examples/enhancement/David1M_0.1fra_train_01B-resolution_singlecell_cell_embedding_f1_resolution.npy'
            train_label_path = '/home/sxr280/Spatialformer/data/train_frac10_nicheann.npy'
            test_X_path = '/home/sxr280/scFoundation/model/examples/enhancement/David1M_0.1fra_test_01B-resolution_singlecell_cell_embedding_f1_resolution.npy'
            test_label_path = '/home/sxr280/Spatialformer/data/test_frac10_nicheann.npy'
            val_X_path = '/home/sxr280/scFoundation/model/examples/enhancement/David1M_0.1fra_val_01B-resolution_singlecell_cell_embedding_f1_resolution.npy'
            val_label_path = '/home/sxr280/Spatialformer/data/val_frac10_nicheann.npy'
        elif args.objective == "scGPT":
            train_X_path = '/home/sxr280/Spatialformer/data/train_embeddings_scGPT_niche.npy'
            train_label_path = '/home/sxr280/Spatialformer/data/train_frac10_nicheann.npy'
            test_X_path = '/home/sxr280/Spatialformer/data/test_embeddings_scGPT_niche.npy'
            test_label_path = '/home/sxr280/Spatialformer/data/test_frac10_nicheann.npy'
            val_X_path = '/home/sxr280/Spatialformer/data/val_embeddings_scGPT_niche.npy'
            val_label_path = '/home/sxr280/Spatialformer/data/val_frac10_nicheann.npy'



    train_loader, test_loader, valid_loader, model = initialize(train_X_path, train_label_path, test_X_path, test_label_path, val_X_path, val_label_path, args.batch_size, input_size = args.input_size)


    model = train_with_early_stopping(model, train_loader, valid_loader, epoch = args.epoch, patience = args.patience, n_tasks = args.n_tasks, objective = args.objective, lr = args.lr, monitor = "val_loss")
    # Save the model
    torch.save(model, f'/home/sxr280/Spatialformer/downstream/model/model_{args.n_tasks}_{args.objective}.pth')

    # Evaluate the model
    # train_precision, train_recall, train_f1 = evaluate_metrics(model, train_loader)
    # valid_precision, valid_recall, valid_f1 = evaluate_metrics(model, valid_loader)
    test_precision, test_recall, test_f1 = evaluate_metrics(model, test_loader, args.n_tasks, args.objective)
    print(f'Test Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}')
   



# python cell_type_annotation.py --n_tasks 2 --lr 0.001 --epoch 10000 --patience 5 --batch_size 64


#python cell_type_annotation.py --n_tasks 1 --lr 0.0001 --epoch 1000 --objective spatial --patience 40
# python cell_type_annotation.py --n_tasks 1 --lr 0.001 --epoch 10000 --patience 40 --objective exp --batch_size 64
# python cell_type_annotation.py --n_tasks 1 --lr 0.001 --epoch 10000 --patience 40 --objective mt --batch_size 64

# python cell_type_annotation.py --n_tasks 1 --lr 0.001 --epoch 10000 --patience 10 --objective baseline --batch_size 64 --input_size 343

# python cell_type_annotation.py --n_tasks 1 --lr 0.001 --epoch 10000 --patience 10 --objective scFoundataion --batch_size 64 --input_size 3072
# python cell_type_annotation.py --n_tasks 1 --lr 0.001 --epoch 10000 --patience 10 --objective scGPT --batch_size 64 --input_size 512