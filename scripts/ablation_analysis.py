from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datasets import DatasetDict, load_dataset, concatenate_datasets

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)  # First hidden layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_size, output_size)  # Output layer

    def forward(self, x):
        # Forward pass
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x



class ProbAnalysis:
    def __init__(self, dataset, model, n_splits=5, target_num = 38):
        self.dataset = dataset
        self.target_num = 0
        self.model = model
        self.n_splits = n_splits

    def dataprocess(self, batch_size=32):
        my_df = self.dataset.to_pandas()
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(my_df['Annotations'])
        X = np.vstack(my_df["Embeddings_Norm"])

        # Scale the data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return X, y

    def cross_validate(self, X, y, num_epochs=100, patience=10, batch_size=32):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        fold = 1

        for train_index, val_index in skf.split(X, y):
            print(f'Fold {fold}/{self.n_splits}')
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Convert to PyTorch tensors
            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.long)
            y_val = torch.tensor(y_val, dtype=torch.long)

            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)

            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            self.train(num_epochs, patience)
            self.eval()
            fold += 1

    def train(self, num_epochs=100, patience=10):
        self.model = SimpleNN(target_num=self.target_num)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            for X_batch, y_batch in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                epoch_loss += loss.item()
                num_batches += 1

                loss.backward()
                optimizer.step()

            avg_loss = epoch_loss / num_batches
            print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

            # Early stopping logic
            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve == patience:
                print(f'Early stopping triggered after {epoch+1} epochs with best loss: {best_loss:.4f}')
                break

    def eval(self):
        self.model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        all_features = []
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                outputs = self.model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
                all_features.extend(X_batch.cpu().numpy())

        # Calculate accuracy
        accuracy = correct / total
        print(f'Validation Accuracy: {accuracy:.4f}')

        # Calculate NMI
        NMIcell = normalized_mutual_info_score(all_labels, all_predictions)
        print(f'NMI Score: {NMIcell:.4f}')

        # Calculate ARI
        ARIcell = adjusted_rand_score(all_labels, all_predictions)
        print(f'ARI Score: {ARIcell:.4f}')

        # Calculate ASW (using Euclidean distance by default)
        ASWcell = silhouette_score(np.array(all_features), np.array(all_predictions))
        print(f'ASW Score: {ASWcell:.4f}')

        AveBIO = np.mean([NMIcell, ARIcell, ASWcell])
        print(f"AveBIO score: {AveBIO:.4f}")


if __name__ == "__main__":
# Usage
    hf_cache = "/home/sxr280/Spatialformer/cache"
    dataset = load_dataset("TerminatorJ/xenium_25_lung_dataset_update2", cache_dir = hf_cache, num_proc = 1)
    model = SimpleNN(128, 64, 38)
    analysis = ProbAnalysis(dataset, model)
    X, y = analysis.dataprocess()
    analysis.cross_validate(X, y)
