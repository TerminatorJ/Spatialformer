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



current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
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
    # import pdb; pdb.set_trace()
    train_data = standardization(train_data)
    valid_data = standardization(valid_data)
    test_data = standardization(test_data)

    # Load labels
    train_labels = np.load(train_label_path, allow_pickle=True)
    valid_labels = np.load(val_label_path, allow_pickle=True)
    test_labels = np.load(test_label_path, allow_pickle=True)
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, prefetch_factor=3, persistent_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    input_size = input_size
    hidden_size = 64
    output_size = len(np.unique(train_labels))  # Number of classes
    model = SimpleMLP(input_size, hidden_size, output_size)

    return train_loader, test_loader, valid_loader, model, label_encoder.classes_




def train_with_early_stopping(
    model,
    train_loader,
    valid_loader,
    epoch=10,
    patience=2,
    n_tasks=None,
    objective=None,
    lr=1e-4,
    monitor="train_loss"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    valid_losses = []

    best_loss = float("inf")
    patience_counter = 0

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    for ep in range(epoch):
        # ---------- Training ----------
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ---------- Validation ----------
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in valid_loader:
                X_val = X_val.to(device, non_blocking=True)
                y_val = y_val.to(device, non_blocking=True)

                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                valid_loss += loss.item()

        avg_valid_loss = valid_loss / len(valid_loader)
        valid_losses.append(avg_valid_loss)

        print(
            f"Epoch [{ep+1}/{epoch}] "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Valid Loss: {avg_valid_loss:.4f}"
        )

        # ---------- Early stopping ----------
        monitored_loss = avg_valid_loss if monitor == "val_loss" else avg_train_loss

        if monitored_loss < best_loss:
            best_loss = monitored_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {ep+1}")
                break

    # ---------- Plot ----------
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(valid_losses, label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    save_path = (
        f"/scratch/project_465001820/Spatialformer/downstream/"
        f"cell_types_nich_annotation/figures/"
        f"training_valid_curve_{n_tasks}_{objective}_{current_time}.png"
    )
    plt.savefig(save_path, dpi=300)
    plt.show()

    return model
# Evaluation function with metrics
def evaluate_metrics(model, loader, n_tasks, unique_class, objective = None):
    model.eval()
    all_preds = []
    all_labels = []
    all_outputs = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            #concating all the predicted results
            # import pdb; pdb.set_trace()
            try:
                all_outputs.extend(torch.cat(outputs).cpu().numpy())
            except:
                all_outputs.extend(outputs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    np.save(f"/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/all_outputs_{n_tasks}_{objective}_{current_time}.npy", np.array(all_outputs))
    np.save(f"/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/all_preds_{n_tasks}_{objective}_{current_time}.npy", np.array(all_preds))
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    #print the result of each class
    
    precision_pc = precision_score(all_labels, all_preds, average=None)
    recall_pc = recall_score(all_labels, all_preds, average=None)
    f1_pc = f1_score(all_labels, all_preds, average=None)
    
    # for i,c in enumerate(unique_class):
    #     print(f'{c}:   Precision: {precision_pc[i]}, Recall: {recall_pc[i]}, F1: {f1_pc[i]}')
    
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
        train_X_path = "/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_train_spatialformer_gene_Colon_Disease.npy"
        train_label_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_train_label.npy'
        test_X_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_test_spatialformer_gene_Colon_Disease.npy'
        test_label_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_test_label.npy'
        val_X_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_val_spatialformer_gene_Colon_Disease.npy'
        val_label_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_val_label.npy'
    
    if args.n_tasks == 3:
        # train_X_path = "/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_train_spatialformer_gene_Colon_Disease.npy"
        # train_label_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_train_label.npy'
        # test_X_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_test_spatialformer_gene_Colon_Disease.npy'
        # test_label_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_test_label.npy'
        # val_X_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_val_spatialformer_gene_Colon_Disease.npy'
        # val_label_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_val_label.npy'
        train_X_path = "/scratch/project_465001820/Spatialformer_main_practice/downstream/cell_types_nich_annotation/data/spa_single_ct_concat_train_embed2.npy"
        train_label_path = '/scratch/project_465001820/Spatialformer_main_practice/downstream/cell_types_nich_annotation/data/spa_train_ct.npy'
        test_X_path = '/scratch/project_465001820/Spatialformer_main_practice/downstream/cell_types_nich_annotation/data/spa_single_ct_concat_test_embed2.npy'
        test_label_path = '/scratch/project_465001820/Spatialformer_main_practice/downstream/cell_types_nich_annotation/data/spa_test_ct.npy'
        val_X_path = '/scratch/project_465001820/Spatialformer_main_practice/downstream/cell_types_nich_annotation/data/spa_single_ct_concat_val_embed2.npy'
        val_label_path = '/scratch/project_465001820/Spatialformer_main_practice/downstream/cell_types_nich_annotation/data/spa_val_ct.npy'

    if args.n_tasks == 1:
        if args.objective == "spatial":
            train_X_path = '/home/sxr280/Spatialformer/data/train_embedding_True_False_8_0.1_1_spatial.npy'
            train_label_path = '/home/sxr280/Spatialformer/data/train_labels_True_False_8_0.1_1_spatial.npy'
            test_X_path = '/home/sxr280/Spatialformer/data/test_embedding_True_False_8_0.1_1_spatial.npy'
            test_label_path = '/home/sxr280/Spatialformer/data/test_labels_True_False_8_0.1_1_spatial.npy'
            val_X_path = '/home/sxr280/Spatialformer/data/val_embedding_True_False_8_0.1_1_spatial.npy'
            val_label_path = '/home/sxr280/Spatialformer/data/val_labels_True_False_8_0.1_1_spatial.npy'
        elif args.objective == "scFoundataion":
            # train_X_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_train_01B-resolution_singlecell_cell_embedding_f1_resolution.npy'
            # train_label_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_train_label.npy'
            # test_X_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_test_01B-resolution_singlecell_cell_embedding_f1_resolution.npy'
            # test_label_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_test_label.npy'
            # val_X_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_val_01B-resolution_singlecell_cell_embedding_f1_resolution.npy'
            # val_label_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_val_label.npy'
            train_X_path = '/scratch/project_465001820/Spatialformer_main_practice/downstream/cell_types_nich_annotation/data/VUILD110_0.1fra_train_celltype_01B-resolution_singlecell_cell_embedding_f1_resolution.npy'
            train_label_path = '/scratch/project_465001820/Spatialformer_main_practice/downstream/cell_types_nich_annotation/data/spa_train_ct.npy'
            test_X_path = '/scratch/project_465001820/Spatialformer_main_practice/downstream/cell_types_nich_annotation/data/VUILD110_0.1fra_test_celltype_01B-resolution_singlecell_cell_embedding_f1_resolution.npy'
            test_label_path = '/scratch/project_465001820/Spatialformer_main_practice/downstream/cell_types_nich_annotation/data/spa_test_ct.npy'
            val_X_path = '/scratch/project_465001820/Spatialformer_main_practice/downstream/cell_types_nich_annotation/data/VUILD110_0.1fra_val_celltype_01B-resolution_singlecell_cell_embedding_f1_resolution.npy'
            val_label_path = '/scratch/project_465001820/Spatialformer_main_practice/downstream/cell_types_nich_annotation/data/spa_val_ct.npy'
        elif args.objective == "scGPT":
            train_X_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_train_embeddings_scGPT.npy'
            train_label_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_train_label.npy'
            test_X_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_test_embeddings_scGPT.npy'
            test_label_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_test_label.npy'
            val_X_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_val_embeddings_scGPT.npy'
            val_label_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_val_label.npy'
        elif args.objective == "Geneformer":
            # train_X_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_train_geneformer.npy'
            # train_label_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_train_label.npy'
            # test_X_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_test_geneformer.npy'
            # test_label_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_test_label.npy'
            # val_X_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_val_geneformer.npy'
            # val_label_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_val_label.npy'
            train_X_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/Geneformer_VUILD110_ct_train_embed_316M.npy'
            train_label_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/spa_train_ct.npy'
            test_X_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/Geneformer_VUILD110_ct_test_embed_316M.npy'
            test_label_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/spa_test_ct.npy'
            val_X_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/Geneformer_VUILD110_ct_val_embed_316M.npy'
            val_label_path = '/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/spa_val_ct.npy'



    train_loader, test_loader, valid_loader, model, unique_class = initialize(train_X_path, train_label_path, test_X_path, test_label_path, val_X_path, val_label_path, args.batch_size, input_size = args.input_size)


    model = train_with_early_stopping(model, train_loader, valid_loader, epoch = args.epoch, patience = args.patience, n_tasks = args.n_tasks, objective = args.objective, lr = args.lr, monitor = "val_loss")
    # Save the model
    torch.save(model, f'/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/checkpoints/model_{args.n_tasks}_{args.objective}.pth')

    # Evaluate the model
    # train_precision, train_recall, train_f1 = evaluate_metrics(model, train_loader)
    # valid_precision, valid_recall, valid_f1 = evaluate_metrics(model, valid_loader)
    test_precision, test_recall, test_f1 = evaluate_metrics(model, test_loader, args.n_tasks, unique_class, args.objective)
    print(f'Test Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}')
   



# python cell_type_annotation.py --n_tasks 2 --lr 0.001 --epoch 10000 --patience 10 --batch_size 64 --input_size 512 --objective SpatialFormer

#three tasks
# python cell_type_annotation.py --n_tasks 3 --lr 0.001 --epoch 10000 --patience 10 --batch_size 64 --input_size 512


#python cell_type_annotation.py --n_tasks 1 --lr 0.0001 --epoch 1000 --objective spatial --patience 40
# python cell_type_annotation.py --n_tasks 1 --lr 0.001 --epoch 10000 --patience 40 --objective exp --batch_size 64
# python cell_type_annotation.py --n_tasks 1 --lr 0.001 --epoch 10000 --patience 40 --objective mt --batch_size 64

# python cell_type_annotation.py --n_tasks 1 --lr 0.001 --epoch 10000 --patience 10 --objective baseline --batch_size 64 --input_size 343

# scFoundation
# python cell_type_annotation.py --n_tasks 1 --lr 0.001 --epoch 10000 --patience 10 --objective scFoundataion --batch_size 64 --input_size 3072
# scGPT
# python cell_type_annotation.py --n_tasks 1 --lr 0.001 --epoch 10000 --patience 10 --objective scGPT --batch_size 64 --input_size 512
# Geneformer
# python cell_type_annotation.py --n_tasks 1 --lr 0.0001 --epoch 10000 --patience 10 --objective Geneformer --batch_size 64 --input_size 896
# python cell_type_annotation.py --n_tasks 1 --lr 0.0001 --epoch 10000 --patience 10 --objective Geneformer --batch_size 64 --input_size 1152