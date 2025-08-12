import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

class FeatureDataset(Dataset):
    def __init__(self, feature_file, label_file, max_len=None):
        self.features = np.load(feature_file, allow_pickle=True)
        self.labels = np.load(label_file)
        
        # make features 2D
        processed_features = []
        for feat in self.features:
            if feat.ndim == 1:
                feat = feat[np.newaxis, :]
            processed_features.append(feat)
        self.features = processed_features
        
        # get max length
        if max_len is None:
            lengths = [feat.shape[1] for feat in self.features]
            self.max_len = int(np.percentile(lengths, 95))
        else:
            self.max_len = max_len
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        feat = self.features[idx]
        label = self.labels[idx]
        
        # pad or cut
        c, t = feat.shape
        if t > self.max_len:
            feat = feat[:, :self.max_len]
        elif t < self.max_len:
            pad_width = self.max_len - t
            feat = np.pad(feat, ((0, 0), (0, pad_width)), mode='constant')
        
        return torch.FloatTensor(feat), torch.LongTensor([label]).squeeze()


class SimpleCNN(nn.Module):
    def __init__(self, n_channels, max_len, num_classes):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.relu(self.conv3(x))
        x = self.global_pool(x)
        x = x.flatten(1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def train_model(feature_name, epochs=30):
    """Train a model for one feature type"""
    
    print(f"Training {feature_name}...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load data
    train_features = f'train_features_corrected/{feature_name}_features.npy'
    train_labels = f'train_features_corrected/labels.npy'
    dev_features = f'dev_features_corrected/{feature_name}_features.npy'
    dev_labels = f'dev_features_corrected/labels.npy'
    
    # check files exist
    for f in [train_features, train_labels, dev_features, dev_labels]:
        if not os.path.exists(f):
            print(f"File not found: {f}")
            return None
    
    train_dataset = FeatureDataset(train_features, train_labels)
    dev_dataset = FeatureDataset(dev_features, dev_labels, max_len=train_dataset.max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
    
    # create model
    sample_feat, _ = train_dataset[0]
    n_channels, max_len = sample_feat.shape
    num_classes = len(np.unique(train_dataset.labels))
    
    print(f"Input: {n_channels} x {max_len}, Classes: {num_classes}")
    
    model = SimpleCNN(n_channels, max_len, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0.0
    train_accs = []
    dev_accs = []
    
    # training loop
    for epoch in range(epochs):
        # train
        model.train()
        train_correct = 0
        train_total = 0
        
        for batch_feat, batch_labels in train_loader:
            batch_feat, batch_labels = batch_feat.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_feat)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_correct += (outputs.argmax(1) == batch_labels).sum().item()
            train_total += batch_labels.size(0)
        
        train_acc = train_correct / train_total
        
        # validation
        model.eval()
        dev_correct = 0
        dev_total = 0
        
        with torch.no_grad():
            for batch_feat, batch_labels in dev_loader:
                batch_feat, batch_labels = batch_feat.to(device), batch_labels.to(device)
                outputs = model(batch_feat)
                
                dev_correct += (outputs.argmax(1) == batch_labels).sum().item()
                dev_total += batch_labels.size(0)
        
        dev_acc = dev_correct / dev_total
        
        train_accs.append(train_acc)
        dev_accs.append(dev_acc)
        
        print(f"Epoch {epoch+1:2d}: Train {train_acc:.3f}, Dev {dev_acc:.3f}")
        
        # save best model
        if dev_acc > best_acc:
            best_acc = dev_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'n_channels': n_channels,
                    'max_len': max_len,
                    'num_classes': num_classes
                },
                'best_acc': best_acc
            }, f'{feature_name}_model_corrected.pth')
    
    print(f"Best accuracy: {best_acc:.3f}")
    
    # simple plot
    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, label='Train')
    plt.plot(dev_accs, label='Dev')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{feature_name.upper()} Training')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{feature_name}_training.png')
    plt.show()
    
    return best_acc


def evaluate_model(feature_name):
    """Test model on eval set"""
    
    print(f"Evaluating {feature_name}...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load model
    model_path = f'{feature_name}_model_corrected.pth'
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return 0
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['model_config']
    
    model = SimpleCNN(config['n_channels'], config['max_len'], config['num_classes']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # load eval data
    eval_features = f'eval_features_corrected/{feature_name}_features.npy'
    eval_labels = f'eval_features_corrected/labels.npy'
    
    if not os.path.exists(eval_features):
        print(f"Eval file not found: {eval_features}")
        return 0
    
    eval_dataset = FeatureDataset(eval_features, eval_labels, max_len=config['max_len'])
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
    
    # evaluate
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_feat, batch_labels in eval_loader:
            batch_feat = batch_feat.to(device)
            outputs = model(batch_feat)
            preds = outputs.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
    
    eval_acc = accuracy_score(all_labels, all_preds)
    print(f"Eval accuracy: {eval_acc:.3f}")
    
    return eval_acc


if __name__ == "__main__":
    print("Training models with corrected labels")
    
    features = ['mfcc', 'cqt', 'lpc']
    results = {}
    
    # train all features
    for feature in features:
        print(f"\n--- {feature.upper()} ---")
        dev_acc = train_model(feature)
        if dev_acc:
            eval_acc = evaluate_model(feature)
            results[feature] = {'dev': dev_acc, 'eval': eval_acc}
    
    # summary
    print("\nResults:")
    print("Feature  Dev    Eval")
    print("-" * 20)
    for feature, accs in results.items():
        print(f"{feature:<8} {accs['dev']:.3f} {accs['eval']:.3f}")
    
    if results:
        best_feature = max(results.keys(), key=lambda x: results[x]['dev'])
        print(f"\nBest: {best_feature} (dev: {results[best_feature]['dev']:.3f})")
    
    print("Done!")