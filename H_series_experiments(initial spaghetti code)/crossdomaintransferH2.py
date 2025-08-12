import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from collections import defaultdict
import scipy.stats as stats

# use the dataset class from before
class CorrectedFeatureDataset:
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
        
        c, t = feat.shape
        if t > self.max_len:
            feat = feat[:, :self.max_len]
        elif t < self.max_len:
            pad_width = self.max_len - t
            feat = np.pad(feat, ((0, 0), (0, pad_width)), mode='constant')
        
        return torch.FloatTensor(feat), torch.LongTensor([label]).squeeze()


def load_models():
    """Load the trained models"""
    models = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for feature_type in ['mfcc', 'cqt', 'lpc']:
        model_path = f'{feature_type}_model_corrected.pth'
        
        if os.path.exists(model_path):
            try:
                from updatedCNNmodel import EnhancedCNN
                checkpoint = torch.load(model_path, map_location=device)
                model_config = checkpoint['model_config']
                
                model = EnhancedCNN(
                    n_channels=model_config['n_channels'],
                    max_len=model_config['max_len'], 
                    num_classes=model_config['num_classes'],
                    dropout=0.4
                ).to(device)
                
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                models[feature_type] = model
                print(f"Loaded {feature_type}")
            except:
                print(f"Failed to load {feature_type}")
    
    return models, device


def test_unknown_attacks(models, device):
    """Test how models respond to unknown attacks A07-A19"""
    
    results = {}
    
    for feature_type, model in models.items():
        print(f"\nTesting {feature_type} on unknown attacks...")
        
        # load eval data
        eval_feature_file = f'eval_features_corrected/{feature_type}_features.npy'
        eval_label_file = f'eval_features_corrected/labels.npy'
        
        if not os.path.exists(eval_feature_file):
            print(f"No eval file for {feature_type}")
            continue
        
        # get max length from training
        train_dataset = CorrectedFeatureDataset(
            f'train_features_corrected/{feature_type}_features.npy',
            f'train_features_corrected/labels.npy'
        )
        
        eval_dataset = CorrectedFeatureDataset(eval_feature_file, eval_label_file, max_len=train_dataset.max_len)
        eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
        
        # get predictions
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_feat, batch_labels in eval_loader:
                batch_feat = batch_feat.to(device)
                outputs = model(batch_feat)
                preds = outputs.argmax(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.numpy())
        
        # analyze confusion for each unknown attack
        confusion_scores = []
        
        for true_label in range(7, 20):  # A07-A19
            mask = np.array(all_labels) == true_label
            if not np.any(mask):
                continue
            
            attack_preds = np.array(all_preds)[mask]
            
            # how confused is the model?
            pred_counts = np.bincount(attack_preds, minlength=7)
            pred_dist = pred_counts / len(attack_preds)
            
            # calculate entropy as confusion measure
            entropy = -np.sum(pred_dist * np.log2(pred_dist + 1e-10))
            max_entropy = np.log2(7)
            normalized_entropy = entropy / max_entropy
            
            confusion_scores.append(normalized_entropy)
            
            most_common = np.argmax(pred_dist)
            print(f"  A{true_label:02d}: mostly predicted as class {most_common} ({pred_dist[most_common]:.2f}), confusion={normalized_entropy:.3f}")
        
        avg_confusion = np.mean(confusion_scores)
        results[feature_type] = {
            'avg_confusion': avg_confusion,
            'generalization': 1 - avg_confusion  # lower confusion = better generalization
        }
        
        print(f"  Average confusion: {avg_confusion:.3f}")
        print(f"  Generalization score: {1 - avg_confusion:.3f}")
    
    return results


def analyze_correlation(results):
    """Check if better known performance correlates with better unknown performance"""
    
    # known attack performance from previous experiments
    # TODO: get these from actual results
    known_performance = {
        'mfcc': 0.975,  # average F1 on A01-A06
        'cqt': 0.988,
        'lpc': 0.894
    }
    
    print("\nCorrelation analysis:")
    print("Hypothesis: Better known attack performance -> better unknown attack generalization")
    
    features = []
    known_scores = []
    unknown_scores = []
    
    for feature in ['mfcc', 'cqt', 'lpc']:
        if feature in results:
            features.append(feature)
            known_scores.append(known_performance[feature])
            unknown_scores.append(results[feature]['generalization'])
    
    if len(features) < 2:
        print("Not enough data for correlation")
        return
    
    # calculate correlation
    correlation = np.corrcoef(known_scores, unknown_scores)[0, 1]
    
    print(f"\nResults:")
    for i, feature in enumerate(features):
        print(f"{feature}: known={known_scores[i]:.3f}, unknown={unknown_scores[i]:.3f}")
    
    print(f"\nCorrelation: {correlation:.3f}")
    
    if correlation > 0.5:
        print("Strong positive correlation - hypothesis supported")
    elif correlation > 0.2:
        print("Moderate correlation - some support")
    elif correlation > -0.2:
        print("Weak correlation - inconclusive")
    else:
        print("Negative correlation - hypothesis not supported")
    
    # simple ranking check
    expected_ranking = ['cqt', 'mfcc', 'lpc']  # based on known performance
    actual_ranking = sorted(features, key=lambda f: results[f]['generalization'], reverse=True)
    
    print(f"\nExpected ranking: {expected_ranking}")
    print(f"Actual ranking: {actual_ranking}")
    print(f"Match: {'Yes' if expected_ranking == actual_ranking else 'No'}")
    
    return correlation, expected_ranking, actual_ranking


def make_simple_plot(results):
    """Make a simple plot"""
    
    known_performance = {'mfcc': 0.975, 'cqt': 0.988, 'lpc': 0.894}
    
    features = list(results.keys())
    known = [known_performance[f] for f in features]
    unknown = [results[f]['generalization'] for f in features]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(known, unknown)
    
    for i, feature in enumerate(features):
        plt.annotate(feature, (known[i], unknown[i]))
    
    plt.xlabel('Known Attack Performance')
    plt.ylabel('Unknown Attack Generalization')
    plt.title('Domain Transfer Analysis')
    plt.grid(True)
    
    # add trend line
    z = np.polyfit(known, unknown, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(known), max(known), 100)
    plt.plot(x_line, p(x_line), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('domain_transfer.png')
    plt.show()
    
    print("Plot saved as domain_transfer.png")


if __name__ == "__main__":
    print("H2 Analysis: Domain Transfer")
    print("Testing if good known attack performance means good unknown attack performance")
    
    # load models
    models, device = load_models()
    
    if not models:
        print("No models loaded")
        exit()
    
    print(f"Loaded {len(models)} models")
    
    # test on unknown attacks
    results = test_unknown_attacks(models, device)
    
    # analyze correlation
    correlation, expected, actual = analyze_correlation(results)
    
    # make plot
    make_simple_plot(results)
    
    print("\nDone!")
    print(f"Correlation: {correlation:.3f}")
    print(f"Ranking match: {'Yes' if expected == actual else 'No'}")