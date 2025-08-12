import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class AudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]]).squeeze()


class AudioCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(AudioCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=5)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        # adaptive pooling to handle different sizes
        x = nn.AdaptiveAvgPool1d(1)(x)
        x = x.view(x.size(0), -1)
        
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def train_model(train_features, train_labels, val_features, val_labels, epochs=20):
    """Train the CNN model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # create datasets
    train_dataset = AudioDataset(train_features, train_labels)
    val_dataset = AudioDataset(val_features, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # create model
    input_size = train_features[0].shape[0]
    num_classes = len(set(train_labels))
    model = AudioCNN(input_size, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_accs = []
    val_accs = []
    
    print(f"Training model with {input_size} input features, {num_classes} classes")
    
    for epoch in range(epochs):
        # training
        model.train()
        train_correct = 0
        train_total = 0
        
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_correct += (outputs.argmax(1) == batch_labels).sum().item()
            train_total += batch_labels.size(0)
        
        train_acc = train_correct / train_total
        
        # validation
        model.eval()
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                outputs = model(batch_features)
                preds = outputs.argmax(1).cpu().numpy()
                
                val_preds.extend(preds)
                val_true.extend(batch_labels.numpy())
        
        val_acc = accuracy_score(val_true, val_preds)
        
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1:2d}: Train {train_acc:.3f}, Val {val_acc:.3f}")
    
    # plot results
    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training Progress')
    plt.show()
    
    return model


def load_data(feature_name):
    """Load features and labels for training"""
    
    print(f"Loading {feature_name} data...")
    
    # load files
    train_features = np.load(f'train_features/{feature_name}_features.npy', allow_pickle=True)
    train_labels = np.load('train_features/labels.npy')
    val_features = np.load(f'val_features/{feature_name}_features.npy', allow_pickle=True)
    val_labels = np.load('val_features/labels.npy')
    
    # simple preprocessing - just average across time
    train_features = [np.mean(f, axis=1) if f.ndim > 1 else f for f in train_features]
    val_features = [np.mean(f, axis=1) if f.ndim > 1 else f for f in val_features]
    
    print(f"Loaded {len(train_features)} train samples, {len(val_features)} val samples")
    
    return train_features, train_labels, val_features, val_labels


if __name__ == "__main__":
    # train models for different features
    features = ['mfcc', 'cqt', 'lpc']
    
    for feature in features:
        print(f"\n--- Training {feature.upper()} model ---")
        
        try:
            train_feat, train_lab, val_feat, val_lab = load_data(feature)
            model = train_model(train_feat, train_lab, val_feat, val_lab)
            torch.save(model.state_dict(), f'{feature}_model.pth')
            print(f"Saved {feature} model")
        except Exception as e:
            print(f"Error training {feature}: {e}")
    
    print("\nDone!")