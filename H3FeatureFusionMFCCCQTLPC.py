import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class FusionDataset(Dataset):
    def __init__(self, mfcc_file, cqt_file, lpc_file, labels_file, max_len=None):
        # load all features
        self.mfcc = np.load(mfcc_file, allow_pickle=True)
        self.cqt = np.load(cqt_file, allow_pickle=True)
        self.lpc = np.load(lpc_file, allow_pickle=True)
        self.labels = np.load(labels_file)
        
        # make sure features are 2D
        self.mfcc = [f[np.newaxis, :] if f.ndim == 1 else f for f in self.mfcc]
        self.cqt = [f[np.newaxis, :] if f.ndim == 1 else f for f in self.cqt]
        self.lpc = [f[np.newaxis, :] if f.ndim == 1 else f for f in self.lpc]
        
        # get max length for padding
        if max_len is None:
            all_lengths = []
            for features in [self.mfcc, self.cqt, self.lpc]:
                lengths = [f.shape[1] for f in features]
                all_lengths.extend(lengths)
            self.max_len = int(np.percentile(all_lengths, 95))
        else:
            self.max_len = max_len
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        mfcc = self.mfcc[idx]
        cqt = self.cqt[idx]
        lpc = self.lpc[idx]
        label = self.labels[idx]
        
        # pad all features to same length
        for feat in [mfcc, cqt, lpc]:
            c, t = feat.shape
            if t > self.max_len:
                feat = feat[:, :self.max_len]
            elif t < self.max_len:
                pad = self.max_len - t
                feat = np.pad(feat, ((0,0), (0,pad)), mode='constant')
        
        # concatenate features
        fused = np.concatenate([mfcc, cqt, lpc], axis=0)
        
        return torch.FloatTensor(fused), torch.LongTensor([label]).squeeze()


class SimpleFusionCNN(nn.Module):
    def __init__(self, input_channels, num_classes=7):
        super(SimpleFusionCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        
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


def train_fusion_model(epochs=25):
    """Train the fusion model"""
    
    print("Training fusion model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load data
    train_mfcc = 'train_features_corrected/mfcc_features.npy'
    train_cqt = 'train_features_corrected/cqt_features.npy'
    train_lpc = 'train_features_corrected/lpc_features.npy'
    train_labels = 'train_features_corrected/labels.npy'
    
    dev_mfcc = 'dev_features_corrected/mfcc_features.npy'
    dev_cqt = 'dev_features_corrected/cqt_features.npy'
    dev_lpc = 'dev_features_corrected/lpc_features.npy'
    dev_labels = 'dev_features_corrected/labels.npy'
    
    # check files exist
    files = [train_mfcc, train_cqt, train_lpc, train_labels, 
             dev_mfcc, dev_cqt, dev_lpc, dev_labels]
    for f in files:
        if not os.path.exists(f):
            print(f"File not found: {f}")
            return None
    
    train_dataset = FusionDataset(train_mfcc, train_cqt, train_lpc, train_labels)
    dev_dataset = FusionDataset(dev_mfcc, dev_cqt, dev_lpc, dev_labels, 
                               max_len=train_dataset.max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
    
    # create model
    sample_feat, _ = train_dataset[0]
    input_channels = sample_feat.shape[0]
    num_classes = len(np.unique(train_dataset.labels))
    
    print(f"Input channels: {input_channels}, Classes: {num_classes}")
    
    model = SimpleFusionCNN(input_channels, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
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
            torch.save(model.state_dict(), 'fusion_model.pth')
    
    print(f"Best accuracy: {best_acc:.3f}")
    
    # plot training
    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, label='Train')
    plt.plot(dev_accs, label='Dev')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Fusion Model Training')
    plt.legend()
    plt.grid(True)
    plt.savefig('fusion_training.png')
    plt.show()
    
    return best_acc


def test_fusion_model():
    """Test fusion model on eval set"""
    
    print("Testing fusion model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load eval data
    eval_mfcc = 'eval_features_corrected/mfcc_features.npy'
    eval_cqt = 'eval_features_corrected/cqt_features.npy'
    eval_lpc = 'eval_features_corrected/lpc_features.npy'
    eval_labels = 'eval_features_corrected/labels.npy'
    
    files = [eval_mfcc, eval_cqt, eval_lpc, eval_labels]
    for f in files:
        if not os.path.exists(f):
            print(f"File not found: {f}")
            return 0
    
    # get max_len from train data
    train_dataset = FusionDataset(
        'train_features_corrected/mfcc_features.npy',
        'train_features_corrected/cqt_features.npy', 
        'train_features_corrected/lpc_features.npy',
        'train_features_corrected/labels.npy'
    )
    
    eval_dataset = FusionDataset(eval_mfcc, eval_cqt, eval_lpc, eval_labels,
                                max_len=train_dataset.max_len)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
    
    # load model
    sample_feat, _ = eval_dataset[0]
    input_channels = sample_feat.shape[0]
    model = SimpleFusionCNN(input_channels).to(device)
    
    if not os.path.exists('fusion_model.pth'):
        print("Model file not found")
        return 0
    
    model.load_state_dict(torch.load('fusion_model.pth', map_location=device))
    model.eval()
    
    # test
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_feat, batch_labels in eval_loader:
            batch_feat = batch_feat.to(device)
            outputs = model(batch_feat)
            preds = outputs.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
    
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"Test accuracy: {test_acc:.3f}")
    
    return test_acc


def compare_with_individual():
    """Compare fusion with individual feature results"""
    
    print("Comparing fusion vs individual features...")
    
    # individual results from previous experiments
    # TODO: get these from actual saved results
    individual_results = {
        'MFCC': 0.965,
        'CQT': 0.972, 
        'LPC': 0.891
    }
    
    # fusion result
    fusion_acc = test_fusion_model()
    
    # comparison
    print("\nResults comparison:")
    print("Method     Accuracy")
    print("-" * 20)
    for method, acc in individual_results.items():
        print(f"{method:<10} {acc:.3f}")
    print(f"{'Fusion':<10} {fusion_acc:.3f}")
    
    best_individual = max(individual_results.values())
    improvement = fusion_acc - best_individual
    
    print(f"\nBest individual: {best_individual:.3f}")
    print(f"Fusion:          {fusion_acc:.3f}")
    print(f"Improvement:     {improvement:+.3f}")
    
    if improvement > 0:
        print("Fusion beats individual features!")
    else:
        print("Individual features still better")
    
    # simple plot
    plt.figure(figsize=(8, 5))
    methods = list(individual_results.keys()) + ['Fusion']
    accuracies = list(individual_results.values()) + [fusion_acc]
    colors = ['blue', 'blue', 'blue', 'red']
    
    bars = plt.bar(methods, accuracies, color=colors, alpha=0.7)
    plt.ylabel('Accuracy')
    plt.title('Individual vs Fusion Performance')
    plt.ylim(0.8, 1.0)
    
    # add values on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.3f}', ha='center', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fusion_comparison.png')
    plt.show()
    
    return improvement


if __name__ == "__main__":
    print("Feature Fusion Experiment")
    print("Combining MFCC + CQT + LPC features")
    
    # train fusion model
    train_acc = train_fusion_model()
    
    if train_acc:
        # compare with individual features
        improvement = compare_with_individual()
        
        print(f"\nFusion experiment complete!")
        print(f"Improvement over best individual: {improvement:+.3f}")
        
        if improvement > 0:
            print("Conclusion: Fusion helps!")
        else:
            print("Conclusion: Individual features sufficient")
    
    print("Done!")