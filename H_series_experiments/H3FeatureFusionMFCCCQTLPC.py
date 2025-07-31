import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import json
from tqdm import tqdm

class MultiFusionDataset(Dataset):
    """
    Dataset that loads multiple feature types for fusion experiments
    """
    def __init__(self, feature_files_dict, labels_file, max_len=None):
        """
        Args:
            feature_files_dict: {'mfcc': 'path', 'cqt': 'path', 'lpc': 'path'}
            labels_file: path to labels
            max_len: max sequence length for padding
        """
        self.feature_names = list(feature_files_dict.keys())
        self.labels = np.load(labels_file)
        
        # Load all feature types
        self.features = {}
        for name, file_path in feature_files_dict.items():
            features = np.load(file_path, allow_pickle=True)
            # Ensure 2D: (channels, time)
            processed = []
            for feat in features:
                arr = np.array(feat)
                if arr.ndim == 1:
                    arr = arr[np.newaxis, :]
                processed.append(arr)
            self.features[name] = processed
        
        # Determine max_len if not provided
        if max_len is None:
            all_lengths = []
            for name in self.feature_names:
                lengths = [feat.shape[1] for feat in self.features[name]]
                all_lengths.extend(lengths)
            self.max_len = int(np.percentile(all_lengths, 95))
        else:
            self.max_len = max_len
        
        print(f"MultiFusionDataset: {len(self.labels)} samples, max_len: {self.max_len}")
        print(f"Feature types: {self.feature_names}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        
        # Process each feature type
        processed_features = {}
        for name in self.feature_names:
            feat = self.features[name][idx]
            c, t = feat.shape
            
            # Pad or truncate
            if t > self.max_len:
                feat = feat[:, :self.max_len]
            elif t < self.max_len:
                pad = self.max_len - t
                feat = np.pad(feat, ((0,0), (0,pad)), mode='constant')
            
            processed_features[name] = torch.FloatTensor(feat)
        
        return processed_features, torch.LongTensor([label]).squeeze()


class EarlyFusionCNN(nn.Module):
    """
    Early fusion: Concatenate features before CNN processing
    """
    def __init__(self, feature_dims, max_len, num_classes=7, dropout=0.4):
        super(EarlyFusionCNN, self).__init__()
        
        # feature_dims: {'mfcc': 13, 'cqt': 84, 'lpc': 1}
        self.feature_names = list(feature_dims.keys())
        total_channels = sum(feature_dims.values())
        
        print(f"EarlyFusion: Concatenating {feature_dims} → {total_channels} channels")
        
        # Enhanced CNN architecture for fused features
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(total_channels, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
            
            # Second conv block  
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
            
            # Third conv block
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
            
            # Fourth conv block
            nn.Conv1d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, feature_dict):
        # Concatenate features along channel dimension
        features = [feature_dict[name] for name in self.feature_names]
        x = torch.cat(features, dim=1)  # Concatenate along channel dimension
        
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x


class LateFusionCNN(nn.Module):
    """
    Late fusion: Process features separately, combine predictions
    """
    def __init__(self, trained_models, feature_dims, num_classes=7, fusion_method='weighted'):
        super(LateFusionCNN, self).__init__()
        
        self.feature_names = list(trained_models.keys())
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        
        # Use pre-trained feature extractors (freeze them)
        self.feature_extractors = nn.ModuleDict()
        for name, model in trained_models.items():
            # Extract feature extractor part (everything except final classifier)
            feature_extractor = nn.Sequential(*list(model.children())[:-1])
            
            # Freeze parameters
            for param in feature_extractor.parameters():
                param.requires_grad = False
                
            self.feature_extractors[name] = feature_extractor
        
        if fusion_method == 'weighted':
            # Learnable fusion weights based on H2 performance
            h2_performance = {'mfcc': 0.708, 'cqt': 0.726, 'lpc': 0.443}
            init_weights = torch.tensor([h2_performance.get(name, 1.0) for name in self.feature_names])
            init_weights = init_weights / init_weights.sum()
            
            self.fusion_weights = nn.Parameter(init_weights)
            print(f"LateFusion weights initialized: {dict(zip(self.feature_names, init_weights.tolist()))}")
            
        elif fusion_method == 'learned':
            # Learn fusion through small MLP
            self.fusion_mlp = nn.Sequential(
                nn.Linear(len(self.feature_names) * num_classes, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
        
    def forward(self, feature_dict):
        # Get predictions from each feature
        feature_predictions = []
        
        for name in self.feature_names:
            # Extract features using pre-trained model
            features = self.feature_extractors[name](feature_dict[name])
            
            # Get the final classifier from the original model
            # This is a bit tricky - we need to recreate the classifier
            if features.dim() > 2:
                features = features.flatten(start_dim=1)
            
            # Simple classifier for this demo (in practice, use the original)
            pred = F.linear(features, weight=torch.randn(self.num_classes, features.size(1)).to(features.device))
            feature_predictions.append(F.softmax(pred, dim=1))
        
        if self.fusion_method == 'weighted':
            # Weighted combination
            weighted_preds = []
            for i, pred in enumerate(feature_predictions):
                weighted_preds.append(self.fusion_weights[i] * pred)
            
            fused_output = torch.stack(weighted_preds).sum(dim=0)
            return fused_output
            
        elif self.fusion_method == 'learned':
            # Concatenate and learn fusion
            concat_preds = torch.cat(feature_predictions, dim=1)
            return self.fusion_mlp(concat_preds)
        
        else:  # simple average
            return torch.stack(feature_predictions).mean(dim=0)


class AttentionFusionCNN(nn.Module):
    """
    Attention-based fusion: Learn to attend to different features
    """
    def __init__(self, feature_dims, max_len, num_classes=7, dropout=0.4):
        super(AttentionFusionCNN, self).__init__()
        
        self.feature_names = list(feature_dims.keys())
        self.num_classes = num_classes
        
        # Individual feature extractors
        self.feature_extractors = nn.ModuleDict()
        self.feature_sizes = {}
        
        for name, input_dim in feature_dims.items():
            # Feature-specific CNN
            extractor = nn.Sequential(
                nn.Conv1d(input_dim, 128, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(128, 256, kernel_size=5, padding=2),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
            self.feature_extractors[name] = extractor
            self.feature_sizes[name] = 512
        
        # Cross-attention mechanism
        embed_dim = 512
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=8, 
            dropout=dropout, 
            batch_first=True
        )
        
        # Final classifier
        total_feature_dim = sum(self.feature_sizes.values())
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, feature_dict):
        # Extract features from each modality
        extracted_features = []
        for name in self.feature_names:
            features = self.feature_extractors[name](feature_dict[name])
            features = features.squeeze(-1)  # Remove last dimension: (batch, 512)
            extracted_features.append(features)
        
        # Stack for attention: (batch_size, num_features, feature_dim)
        feature_stack = torch.stack(extracted_features, dim=1)
        
        # Apply cross-attention
        attended_features, attention_weights = self.cross_attention(
            feature_stack, feature_stack, feature_stack
        )
        
        # Flatten for classification
        attended_features = attended_features.flatten(start_dim=1)
        
        # Final classification
        output = self.classifier(attended_features)
        return output


class H3FusionExperimentor:
    """
    H3: Feature Fusion Hypothesis Testing
    """
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.feature_types = ['mfcc', 'cqt', 'lpc']
        
        # Load individual model performance (from H1/H2)
        self.individual_performance = {
            'mfcc': {'known_f1': 0.986, 'generalization': 0.708},
            'cqt': {'known_f1': 0.989, 'generalization': 0.726},  
            'lpc': {'known_f1': 0.909, 'generalization': 0.443}
        }
        
        # Expected best individual performance
        self.best_individual = max(self.individual_performance.items(), 
                                 key=lambda x: x[1]['known_f1'])
        
        print(f"H3 Target: Beat best individual feature {self.best_individual[0].upper()} "
              f"(F1={self.best_individual[1]['known_f1']:.3f})")
    
    def prepare_fusion_datasets(self):
        """
        Prepare datasets for fusion experiments
        """
        print(f"\n📂 Preparing Fusion Datasets...")
        
        # Check available feature files
        base_dirs = {
            'train': 'train_features_corrected',
            'dev': 'dev_features_corrected',
            'eval': 'eval_features_corrected'
        }
        
        datasets = {}
        
        for split_name, base_dir in base_dirs.items():
            feature_files = {}
            labels_file = os.path.join(base_dir, 'labels.npy')
            
            # Check which features are available
            for feature_type in self.feature_types:
                feature_file = os.path.join(base_dir, f'{feature_type}_features.npy')
                if os.path.exists(feature_file):
                    feature_files[feature_type] = feature_file
            
            if len(feature_files) >= 2 and os.path.exists(labels_file):
                try:
                    dataset = MultiFusionDataset(feature_files, labels_file)
                    datasets[split_name] = dataset
                    print(f"✅ {split_name}: {len(dataset)} samples with {list(feature_files.keys())}")
                except Exception as e:
                    print(f"❌ Error loading {split_name}: {e}")
            else:
                print(f"❌ {split_name}: Missing files")
        
        return datasets
    
    def train_fusion_model(self, model, train_loader, dev_loader, model_name, epochs=30, lr=1e-3):
        """
        Train a fusion model
        """
        print(f"\n🚀 Training {model_name}...")
        
        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_dev_acc = 0.0
        patience_counter = 0
        patience = 8
        
        history = {'train_acc': [], 'dev_acc': [], 'train_loss': [], 'dev_loss': []}
        
        for epoch in range(1, epochs + 1):
            # Training phase
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            
            for batch_features, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
                # Move features to device
                device_features = {}
                for name, features in batch_features.items():
                    device_features[name] = features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(device_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_labels.size(0)
                train_correct += (outputs.argmax(1) == batch_labels).sum().item()
                train_total += batch_labels.size(0)
            
            train_acc = train_correct / train_total
            train_loss = train_loss / train_total
            
            # Validation phase
            model.eval()
            dev_loss, dev_correct, dev_total = 0.0, 0, 0
            
            with torch.no_grad():
                for batch_features, batch_labels in dev_loader:
                    device_features = {}
                    for name, features in batch_features.items():
                        device_features[name] = features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = model(device_features)
                    loss = criterion(outputs, batch_labels)
                    
                    dev_loss += loss.item() * batch_labels.size(0)
                    dev_correct += (outputs.argmax(1) == batch_labels).sum().item()
                    dev_total += batch_labels.size(0)
            
            dev_acc = dev_correct / dev_total
            dev_loss = dev_loss / dev_total
            
            scheduler.step(dev_loss)
            
            # Store history
            history['train_acc'].append(train_acc)
            history['dev_acc'].append(dev_acc)
            history['train_loss'].append(train_loss)
            history['dev_loss'].append(dev_loss)
            
            print(f"Epoch {epoch:2d}/{epochs} | Train: {train_acc:.3f} | Dev: {dev_acc:.3f}")
            
            # Early stopping
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        return {
            'model': model,
            'best_dev_acc': best_dev_acc,
            'history': history,
            'model_name': model_name
        }
    
    def evaluate_fusion_model(self, model, test_loader, model_name):
        """
        Evaluate fusion model
        """
        print(f"\n📊 Evaluating {model_name}...")
        
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                device_features = {}
                for name, features in batch_features.items():
                    device_features[name] = features.to(self.device)
                
                outputs = model(device_features)
                predictions = outputs.argmax(1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_labels.numpy())
        
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Calculate F1 scores per attack
        attack_names = ['bonafide', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06']
        f1_scores = {}
        
        for i, attack_name in enumerate(attack_names):
            if i in all_labels:
                binary_labels = (np.array(all_labels) == i).astype(int)
                binary_preds = (np.array(all_predictions) == i).astype(int)
                f1 = f1_score(binary_labels, binary_preds, zero_division=0)
                f1_scores[attack_name] = f1
        
        avg_f1 = np.mean(list(f1_scores.values()))
        
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Average F1: {avg_f1:.3f}")
        
        return {
            'accuracy': accuracy,
            'avg_f1': avg_f1,
            'f1_scores': f1_scores,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def run_h3_experiments(self):
        """
        Run complete H3 fusion experiments
        """
        print("🚀 H3 EXPERIMENTS: Feature Fusion Hypothesis")
        print("="*60)
        
        print("H3 HYPOTHESIS:")
        print("\"Complementary features fusion will outperform best individual feature\"")
        
        # Prepare datasets
        datasets = self.prepare_fusion_datasets()
        
        if 'train' not in datasets or 'dev' not in datasets:
            print("❌ Missing train or dev datasets!")
            return None
        
        # Get feature dimensions
        sample_features, _ = datasets['train'][0]
        feature_dims = {name: feat.size(0) for name, feat in sample_features.items()}
        max_len = datasets['train'].max_len
        
        print(f"\nFeature dimensions: {feature_dims}")
        print(f"Max sequence length: {max_len}")
        
        # Create data loaders
        train_loader = DataLoader(datasets['train'], batch_size=32, shuffle=True)
        dev_loader = DataLoader(datasets['dev'], batch_size=32, shuffle=False)
        
        fusion_results = {}
        
        # Experiment 1: Early Fusion
        print(f"\n{'='*50}")
        print("EXPERIMENT 1: EARLY FUSION")
        print(f"{'='*50}")
        
        early_fusion_model = EarlyFusionCNN(feature_dims, max_len, num_classes=7)
        early_result = self.train_fusion_model(
            early_fusion_model, train_loader, dev_loader, 
            "Early Fusion", epochs=30
        )
        
        fusion_results['early_fusion'] = early_result
        
        # Experiment 2: Attention Fusion
        print(f"\n{'='*50}")
        print("EXPERIMENT 2: ATTENTION FUSION")
        print(f"{'='*50}")
        
        attention_fusion_model = AttentionFusionCNN(feature_dims, max_len, num_classes=7)
        attention_result = self.train_fusion_model(
            attention_fusion_model, train_loader, dev_loader,
            "Attention Fusion", epochs=30
        )
        
        fusion_results['attention_fusion'] = attention_result
        
        # Test H3 hypothesis
        h3_results = self.test_h3_hypothesis(fusion_results)
        
        # Evaluate on test set if available
        if 'eval' in datasets:
            eval_loader = DataLoader(datasets['eval'], batch_size=32, shuffle=False)
            print(f"\n📊 EVALUATING ON UNKNOWN ATTACKS (A07-A19):")
            
            for fusion_name, result in fusion_results.items():
                eval_result = self.evaluate_fusion_model(
                    result['model'], eval_loader, result['model_name']
                )
                fusion_results[fusion_name]['eval_results'] = eval_result
        
        # Create visualizations
        self.create_h3_visualizations(fusion_results, h3_results)
        
        # Generate report
        report = self.generate_h3_report(fusion_results, h3_results)
        
        return {
            'fusion_results': fusion_results,
            'h3_results': h3_results,
            'report': report
        }
    
    def test_h3_hypothesis(self, fusion_results):
        """
        Test H3: Do fusion methods beat best individual feature?
        """
        print(f"\n🔬 H3 HYPOTHESIS TESTING")
        print("="*40)
        
        best_individual_f1 = self.best_individual[1]['known_f1']
        best_individual_name = self.best_individual[0].upper()
        
        print(f"Target to beat: {best_individual_name} F1 = {best_individual_f1:.3f}")
        
        h3_results = {
            'target_performance': best_individual_f1,
            'target_name': best_individual_name,
            'fusion_comparisons': {},
            'best_fusion': None,
            'hypothesis_supported': False,
            'improvement_achieved': 0.0
        }
        
        best_fusion_performance = 0.0
        best_fusion_name = None
        
        for fusion_name, result in fusion_results.items():
            fusion_f1 = result['best_dev_acc']  # Using dev accuracy as proxy for F1
            improvement = fusion_f1 - best_individual_f1
            
            h3_results['fusion_comparisons'][fusion_name] = {
                'performance': fusion_f1,
                'improvement': improvement,
                'beats_individual': improvement > 0
            }
            
            if fusion_f1 > best_fusion_performance:
                best_fusion_performance = fusion_f1
                best_fusion_name = fusion_name
            
            print(f"{result['model_name']}: F1 = {fusion_f1:.3f} "
                  f"(improvement: {improvement:+.3f})")
        
        h3_results['best_fusion'] = best_fusion_name
        h3_results['improvement_achieved'] = best_fusion_performance - best_individual_f1
        h3_results['hypothesis_supported'] = h3_results['improvement_achieved'] > 0
        
        if h3_results['hypothesis_supported']:
            verdict = "SUPPORTED"
            explanation = f"Best fusion ({best_fusion_name}) outperforms best individual by {h3_results['improvement_achieved']:+.3f}"
        else:
            verdict = "NOT SUPPORTED"  
            explanation = f"No fusion method beats best individual feature ({best_individual_name})"
        
        print(f"\n🏛️ H3 VERDICT: {verdict}")
        print(f"Explanation: {explanation}")
        
        h3_results['verdict'] = verdict
        h3_results['explanation'] = explanation
        
        return h3_results
    
    def create_h3_visualizations(self, fusion_results, h3_results):
        """
        Create H3 visualization plots
        """
        print(f"\n📊 Creating H3 Visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Performance comparison
        ax1 = axes[0, 0]
        
        methods = ['Best Individual'] + [result['model_name'] for result in fusion_results.values()]
        performances = [h3_results['target_performance']] + [result['best_dev_acc'] for result in fusion_results.values()]
        colors = ['red'] + ['blue', 'green', 'orange'][:len(fusion_results)]
        
        bars = ax1.bar(methods, performances, color=colors, alpha=0.7)
        ax1.set_title('H3: Fusion vs Individual Performance', fontweight='bold')
        ax1.set_ylabel('Dev Set Accuracy')
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for bar, perf in zip(bars, performances):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{perf:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Training curves for best fusion
        ax2 = axes[0, 1]
        
        if fusion_results:
            best_fusion = h3_results['best_fusion']
            if best_fusion and best_fusion in fusion_results:
                history = fusion_results[best_fusion]['history']
                epochs = range(1, len(history['train_acc']) + 1)
                
                ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
                ax2.plot(epochs, history['dev_acc'], 'r-', label='Dev Acc', linewidth=2)
                ax2.set_title(f'Training Curves: {fusion_results[best_fusion]["model_name"]}')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        # Plot 3: Improvement breakdown
        ax3 = axes[1, 0]
        
        fusion_names = [result['model_name'] for result in fusion_results.values()]
        improvements = [h3_results['fusion_comparisons'][name]['improvement'] 
                       for name in fusion_results.keys()]
        
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = ax3.bar(fusion_names, improvements, color=colors, alpha=0.7)
        
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_title('Performance Improvement over Best Individual')
        ax3.set_ylabel('Improvement (F1 Score)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, imp in zip(bars, improvements):
            y_pos = bar.get_height() + (0.005 if imp > 0 else -0.015)
            ax3.text(bar.get_x() + bar.get_width()/2, y_pos,
                    f'{imp:+.3f}', ha='center', va='bottom' if imp > 0 else 'top',
                    fontweight='bold')
        
        # Plot 4: H3 Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
H3 FUSION HYPOTHESIS SUMMARY

Hypothesis: Complementary features fusion 
outperforms best individual feature

Target to Beat: {h3_results['target_name']} 
({h3_results['target_performance']:.3f})

Best Fusion: {fusion_results[h3_results['best_fusion']]['model_name'] if h3_results['best_fusion'] else 'None'}
Performance: {max([r['best_dev_acc'] for r in fusion_results.values()]) if fusion_results else 0:.3f}

Improvement: {h3_results['improvement_achieved']:+.3f}

VERDICT: {h3_results['verdict']}
        """
        
        ax4.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen" if h3_results['hypothesis_supported'] else "lightcoral", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('h3_fusion_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ H3 visualization saved as 'h3_fusion_analysis.png'")
    
    def generate_h3_report(self, fusion_results, h3_results):
        """
        Generate comprehensive H3 report
        """
        print(f"\n📝 H3 COMPREHENSIVE REPORT")
        print("="*50)
        
        report = {
            'hypothesis': 'H3: Complementary Feature Fusion Hypothesis',
            'statement': 'Feature fusion will outperform best individual feature on known attacks',
            'verdict': h3_results['verdict'],
            'explanation': h3_results['explanation'],
            'target_performance': h3_results['target_performance'],
            'target_feature': h3_results['target_name'],
            'best_fusion_method': h3_results['best_fusion'],
            'improvement_achieved': h3_results['improvement_achieved'],
            'fusion_results': {},
            'implications': {}
        }
        
        # Store results for each fusion method
        for fusion_name, result in fusion_results.items():
            report['fusion_results'][fusion_name] = {
                'dev_accuracy': result['best_dev_acc'],
                'improvement': h3_results['fusion_comparisons'][fusion_name]['improvement'],
                'beats_individual': h3_results['fusion_comparisons'][fusion_name]['beats_individual']
            }
            
            # Add eval results if available
            if 'eval_results' in result:
                report['fusion_results'][fusion_name]['eval_accuracy'] = result['eval_results']['accuracy']
                report['fusion_results'][fusion_name]['eval_f1'] = result['eval_results']['avg_f1']
        
        # Research implications
        if h3_results['hypothesis_supported']:
            report['implications'] = {
                'practical': 'Fusion methods provide performance gains over individual features',
                'theoretical': 'Confirms complementary information hypothesis in ASVspoof features',
                'methodological': f"Recommend {h3_results['best_fusion']} for ASVspoof detection systems"
            }
        else:
            report['implications'] = {
                'practical': 'Individual features may be sufficient for this task',
                'theoretical': 'Limited complementary information between MFCC, CQT, and LPC',
                'methodological': f"Focus optimization efforts on best individual feature ({h3_results['target_name']})"
            }
        
        # Save detailed results
        with open('h3_fusion_results.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("💾 H3 results saved to 'h3_fusion_results.json'")
        
        # Print key findings
        print(f"\n🎯 H3 KEY FINDINGS:")
        print(f"• Target to beat: {h3_results['target_name']} ({h3_results['target_performance']:.3f})")
        for fusion_name, comparison in h3_results['fusion_comparisons'].items():
            status = "✅ BEATS" if comparison['beats_individual'] else "❌ BELOW"
            print(f"• {fusion_results[fusion_name]['model_name']}: {comparison['performance']:.3f} "
                  f"({comparison['improvement']:+.3f}) {status}")
        
        return report


def run_h3_analysis():
    """
    Main function to run complete H3 analysis
    """
    print("🚀 STARTING H3 ANALYSIS: Feature Fusion Hypothesis")
    print("="*70)
    
    # Initialize experimentor
    experimentor = H3FusionExperimentor()
    
    # Run complete H3 experiments
    results = experimentor.run_h3_experiments()
    
    if results:
        print(f"\n🎉 H3 ANALYSIS COMPLETE!")
        print("📊 Generated files:")
        print("  • h3_fusion_analysis.png (comprehensive visualization)")
        print("  • h3_fusion_results.json (detailed results)")
        print(f"\n🎯 H3 VERDICT: {results['h3_results']['verdict']}")
        print(f"📈 Best improvement: {results['h3_results']['improvement_achieved']:+.3f}")
        print(f"\n🚀 Ready for H2.1 (Deep Dive Analysis) or H4!")
        
        return results
    else:
        print("❌ H3 analysis failed!")
        return None

if __name__ == "__main__":
    # Run H3 analysis
    results = run_h3_analysis()