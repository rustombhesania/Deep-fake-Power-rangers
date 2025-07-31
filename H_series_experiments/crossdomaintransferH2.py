import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import scipy.stats as stats
import json

# Import your dataset class
class CorrectedFeatureDataset:
    def __init__(self, feature_file, label_file, max_len=None):
        self.features = np.load(feature_file, allow_pickle=True)
        self.labels = np.load(label_file)
        
        # Ensure 2D features: (channels, time)
        processed_features = []
        for feat in self.features:
            if feat.ndim == 1:
                feat = feat[np.newaxis, :]
            processed_features.append(feat)
        self.features = processed_features
        
        # Determine max length for padding
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
        
        # Pad or truncate to max_len
        c, t = feat.shape
        if t > self.max_len:
            feat = feat[:, :self.max_len]
        elif t < self.max_len:
            pad_width = self.max_len - t
            feat = np.pad(feat, ((0, 0), (0, pad_width)), mode='constant')
        
        return torch.FloatTensor(feat), torch.LongTensor([label]).squeeze()


class H2DomainTransferAnalyzer:
    """
    H2: Domain Transfer Hypothesis Analyzer
    Tests if features with better known-attack performance generalize better to unknown attacks
    """
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.feature_types = ['mfcc', 'cqt', 'lpc']
        
        # Known attacks (training domain): A01-A06
        self.known_attacks = {
            0: 'bonafide', 1: 'A01', 2: 'A02', 3: 'A03', 4: 'A04', 5: 'A05', 6: 'A06'
        }
        
        # Unknown attacks (test domain): A07-A19  
        self.unknown_attacks = {
            0: 'bonafide', 7: 'A07', 8: 'A08', 9: 'A09', 10: 'A10', 11: 'A11',
            12: 'A12', 13: 'A13', 14: 'A14', 15: 'A15', 16: 'A16', 17: 'A17',
            18: 'A18', 19: 'A19'
        }
        
        self.attack_families = {
            'traditional_tts': ['A01', 'A02', 'A03'],
            'traditional_vc': ['A04', 'A05', 'A06'],
            'neural_vocoder': ['A07', 'A08', 'A09', 'A10', 'A11', 'A12'],
            'neural_e2e': ['A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19']
        }
        
        # Load trained models
        self.models = {}
        self.load_trained_models()
        
        # H1 results (your baseline performance on known attacks)
        self.h1_performance = {
            'mfcc': {'bonafide': 0.952, 'A01': 1.000, 'A02': 0.965, 'A03': 1.000,
                    'A04': 1.000, 'A05': 0.998, 'A06': 0.986},
            'cqt': {'bonafide': 0.960, 'A01': 0.994, 'A02': 0.998, 'A03': 1.000,
                   'A04': 0.996, 'A05': 0.992, 'A06': 0.985},
            'lpc': {'bonafide': 0.904, 'A01': 0.995, 'A02': 0.997, 'A03': 1.000,
                   'A04': 0.561, 'A05': 0.985, 'A06': 0.921}
        }
    
    def load_trained_models(self):
        """Load trained models from H1 experiment"""
        print("📂 Loading trained models for H2 analysis...")
        
        for feature_type in self.feature_types:
            model_path = f'{feature_type}_model_corrected.pth'
            
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                model_config = checkpoint['model_config']
                
                # Import your model class
                try:
                    from updatedCNNmodel import EnhancedCNN
                    model = EnhancedCNN(
                        n_channels=model_config['n_channels'],
                        max_len=model_config['max_len'], 
                        num_classes=model_config['num_classes'],
                        dropout=0.4
                    ).to(self.device)
                    
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.eval()
                    self.models[feature_type] = model
                    print(f"✅ Loaded {feature_type.upper()} model")
                except ImportError:
                    print(f"❌ Could not import model class for {feature_type}")
            else:
                print(f"❌ Model not found: {model_path}")
        
        print(f"📊 Successfully loaded {len(self.models)} models")
    
    def analyze_unknown_attack_responses(self):
        """
        Core H2 analysis: How do models trained on A01-A06 respond to A07-A19?
        """
        print(f"\n🔬 H2 CORE ANALYSIS: Unknown Attack Response Patterns")
        print("="*70)
        
        unknown_responses = defaultdict(dict)
        
        for feature_type in self.feature_types:
            if feature_type not in self.models:
                continue
            
            print(f"\n📊 Analyzing {feature_type.upper()} responses to unknown attacks...")
            
            # Load eval dataset (contains A07-A19)
            eval_feature_file = f'eval_features_corrected/{feature_type}_features.npy'
            eval_label_file = f'eval_features_corrected/labels.npy'
            
            if not os.path.exists(eval_feature_file):
                print(f"❌ Eval file not found: {eval_feature_file}")
                continue
            
            # Get max_len from train
            train_dataset = CorrectedFeatureDataset(
                f'train_features_corrected/{feature_type}_features.npy',
                f'train_features_corrected/labels.npy'
            )
            max_len = train_dataset.max_len
            
            eval_dataset = CorrectedFeatureDataset(eval_feature_file, eval_label_file, max_len=max_len)
            eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
            
            # Get model predictions on unknown attacks
            model = self.models[feature_type]
            model.eval()
            
            all_predictions = []
            all_labels = []
            all_probabilities = []
            
            with torch.no_grad():
                for batch_feat, batch_labels in eval_loader:
                    batch_feat = batch_feat.to(self.device)
                    outputs = model(batch_feat)
                    
                    # Model outputs 7 classes (0-6: bonafide, A01-A06)
                    # But eval has 14 classes (0, 7-19: bonafide, A07-A19)
                    probs = F.softmax(outputs, dim=1)
                    preds = outputs.argmax(1)
                    
                    all_predictions.extend(preds.cpu().numpy())
                    all_labels.extend(batch_labels.numpy())
                    all_probabilities.extend(probs.cpu().numpy())
            
            # Analyze each unknown attack individually
            for true_label in range(7, 20):  # A07-A19
                attack_name = f'A{true_label:02d}'
                attack_mask = np.array(all_labels) == true_label
                
                if not np.any(attack_mask):
                    continue
                
                attack_predictions = np.array(all_predictions)[attack_mask]
                attack_probs = np.array(all_probabilities)[attack_mask]
                
                # Analyze what the model thinks these unknown attacks are
                pred_counts = np.bincount(attack_predictions, minlength=7)
                pred_distribution = pred_counts / len(attack_predictions)
                
                # Key metrics
                most_common_pred = np.argmax(pred_distribution)
                most_common_prob = pred_distribution[most_common_pred]
                avg_confidence = np.mean(np.max(attack_probs, axis=1))
                
                # Entropy (measure of confusion)
                entropy = -np.sum(pred_distribution * np.log2(pred_distribution + 1e-10))
                max_entropy = np.log2(7)  # Maximum possible entropy for 7 classes
                normalized_entropy = entropy / max_entropy
                
                # Store results
                unknown_responses[feature_type][attack_name] = {
                    'sample_count': np.sum(attack_mask),
                    'prediction_distribution': pred_distribution.tolist(),
                    'most_common_prediction': int(most_common_pred),
                    'most_common_prediction_name': list(self.known_attacks.values())[most_common_pred],
                    'confidence_in_most_common': float(most_common_prob),
                    'average_confidence': float(avg_confidence),
                    'entropy': float(entropy),
                    'normalized_entropy': float(normalized_entropy),
                    'confusion_score': float(normalized_entropy)  # Higher = more confused
                }
                
                print(f"  {attack_name}: Mostly predicted as {list(self.known_attacks.values())[most_common_pred]} "
                      f"({most_common_prob:.1%}), Confusion={normalized_entropy:.3f}")
        
        return unknown_responses
    
    def calculate_domain_transfer_metrics(self, unknown_responses):
        """
        Calculate metrics to test H2: correlation between known and unknown performance
        """
        print(f"\n📈 CALCULATING DOMAIN TRANSFER METRICS")
        print("="*50)
        
        transfer_metrics = {}
        
        for feature_type in self.feature_types:
            if feature_type not in unknown_responses:
                continue
            
            # Known domain performance (from H1)
            known_attacks_only = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06']
            known_f1_scores = [self.h1_performance[feature_type][attack] for attack in known_attacks_only]
            avg_known_performance = np.mean(known_f1_scores)
            
            # Unknown domain "performance" (inverse of confusion)
            unknown_attack_names = [f'A{i:02d}' for i in range(7, 20)]
            confusion_scores = []
            
            for attack_name in unknown_attack_names:
                if attack_name in unknown_responses[feature_type]:
                    confusion_scores.append(unknown_responses[feature_type][attack_name]['confusion_score'])
            
            avg_confusion = np.mean(confusion_scores) if confusion_scores else 1.0
            generalization_ability = 1 - avg_confusion  # Higher = better generalization
            
            transfer_metrics[feature_type] = {
                'known_performance': avg_known_performance,
                'unknown_confusion': avg_confusion,
                'generalization_ability': generalization_ability,
                'known_scores': known_f1_scores,
                'confusion_scores': confusion_scores
            }
            
            print(f"{feature_type.upper()}:")
            print(f"  Known domain avg F1: {avg_known_performance:.3f}")
            print(f"  Unknown domain confusion: {avg_confusion:.3f}")
            print(f"  Generalization ability: {generalization_ability:.3f}")
        
        return transfer_metrics
    
    def test_h2_correlation(self, transfer_metrics):
        """
        Test H2: Correlation between known and unknown performance
        """
        print(f"\n🔬 H2 HYPOTHESIS TESTING")
        print("="*40)
        
        print("H2 HYPOTHESIS:")
        print("\"Features with higher intra-domain performance will show better cross-domain generalization\"")
        
        # Extract data for correlation analysis
        features = []
        known_perfs = []
        generalization_abilities = []
        
        for feature_type, metrics in transfer_metrics.items():
            features.append(feature_type)
            known_perfs.append(metrics['known_performance'])
            generalization_abilities.append(metrics['generalization_ability'])
        
        if len(features) < 2:
            print("❌ Not enough features for correlation analysis")
            return None
        
        # Calculate correlation
        correlation = np.corrcoef(known_perfs, generalization_abilities)[0, 1]
        
        # Rank correlation (more robust)
        rank_correlation, rank_p_value = stats.spearmanr(known_perfs, generalization_abilities)
        
        print(f"\n📊 CORRELATION ANALYSIS:")
        print(f"Pearson correlation: {correlation:.3f}")
        print(f"Spearman rank correlation: {rank_correlation:.3f} (p={rank_p_value:.3f})")
        
        print(f"\n📋 DETAILED BREAKDOWN:")
        for i, feature in enumerate(features):
            print(f"{feature.upper()}: Known={known_perfs[i]:.3f}, Generalization={generalization_abilities[i]:.3f}")
        
        # Expected ranking based on H1
        expected_ranking = ['mfcc', 'cqt', 'lpc']  # Best to worst on known attacks
        actual_generalization_ranking = sorted(features, key=lambda f: transfer_metrics[f]['generalization_ability'], reverse=True)
        
        print(f"\nExpected ranking (based on H1): {[f.upper() for f in expected_ranking]}")
        print(f"Actual generalization ranking: {[f.upper() for f in actual_generalization_ranking]}")
        
        # Test H2 verdict
        if correlation > 0.5:
            h2_verdict = "STRONGLY SUPPORTED"
            explanation = f"Strong positive correlation ({correlation:.3f}): better known performance → better generalization"
        elif correlation > 0.2:
            h2_verdict = "SUPPORTED"
            explanation = f"Moderate positive correlation ({correlation:.3f}): some evidence for transfer"
        elif correlation > -0.2:
            h2_verdict = "INCONCLUSIVE"
            explanation = f"Weak correlation ({correlation:.3f}): no clear relationship"
        else:
            h2_verdict = "NOT SUPPORTED"
            explanation = f"Negative correlation ({correlation:.3f}): contradicts hypothesis"
        
        print(f"\n🏛️ H2 VERDICT: {h2_verdict}")
        print(f"Explanation: {explanation}")
        
        # Check if ranking matches expectation
        ranking_match = expected_ranking == actual_generalization_ranking
        print(f"Ranking consistency: {'✅ MATCHES' if ranking_match else '❌ DIFFERS'}")
        
        return {
            'verdict': h2_verdict,
            'explanation': explanation,
            'correlation': correlation,
            'rank_correlation': rank_correlation,
            'rank_p_value': rank_p_value,
            'expected_ranking': expected_ranking,
            'actual_ranking': actual_generalization_ranking,
            'ranking_match': ranking_match,
            'transfer_metrics': transfer_metrics
        }
    
    def create_h2_visualizations(self, unknown_responses, transfer_metrics, h2_results):
        """
        Create comprehensive visualizations for H2 analysis
        """
        print(f"\n📊 Creating H2 Visualizations...")
        
        fig = plt.figure(figsize=(20, 12))
        
        # Plot 1: Unknown attack confusion heatmap
        plt.subplot(2, 3, 1)
        
        # Create confusion matrix
        features = list(unknown_responses.keys())
        attacks = [f'A{i:02d}' for i in range(7, 20)]
        
        confusion_matrix_data = []
        for feature in features:
            confusion_row = []
            for attack in attacks:
                if attack in unknown_responses[feature]:
                    confusion = unknown_responses[feature][attack]['confusion_score']
                    confusion_row.append(confusion)
                else:
                    confusion_row.append(0)
            confusion_matrix_data.append(confusion_row)
        
        sns.heatmap(confusion_matrix_data, 
                   xticklabels=attacks, 
                   yticklabels=[f.upper() for f in features],
                   annot=True, fmt='.2f', cmap='Reds',
                   cbar_kws={'label': 'Confusion Score'})
        plt.title('Unknown Attack Confusion Matrix\n(Higher = More Confused)')
        plt.xticks(rotation=45)
        
        # Plot 2: Known vs Unknown performance scatter
        plt.subplot(2, 3, 2)
        
        known_perfs = [transfer_metrics[f]['known_performance'] for f in features]
        gen_abilities = [transfer_metrics[f]['generalization_ability'] for f in features]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, feature in enumerate(features):
            plt.scatter(known_perfs[i], gen_abilities[i], 
                       s=100, c=colors[i], label=feature.upper(), alpha=0.8)
            plt.annotate(feature.upper(), 
                        (known_perfs[i], gen_abilities[i]),
                        xytext=(5, 5), textcoords='offset points')
        
        # Add trend line
        z = np.polyfit(known_perfs, gen_abilities, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(known_perfs), max(known_perfs), 100)
        plt.plot(x_trend, p(x_trend), "r--", alpha=0.8)
        
        plt.xlabel('Known Attack Performance (F1)')
        plt.ylabel('Unknown Attack Generalization')
        plt.title(f'H2: Domain Transfer Correlation\nr = {h2_results["correlation"]:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Attack family analysis
        plt.subplot(2, 3, 3)
        
        # Group unknown attacks by family
        neural_vocoder_attacks = ['A07', 'A08', 'A09', 'A10', 'A11', 'A12']
        neural_e2e_attacks = ['A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19']
        
        family_confusion = {}
        for feature in features:
            vocoder_confusions = []
            e2e_confusions = []
            
            for attack in neural_vocoder_attacks:
                if attack in unknown_responses[feature]:
                    vocoder_confusions.append(unknown_responses[feature][attack]['confusion_score'])
            
            for attack in neural_e2e_attacks:
                if attack in unknown_responses[feature]:
                    e2e_confusions.append(unknown_responses[feature][attack]['confusion_score'])
            
            family_confusion[feature] = {
                'Neural Vocoder': np.mean(vocoder_confusions) if vocoder_confusions else 0,
                'Neural E2E': np.mean(e2e_confusions) if e2e_confusions else 0
            }
        
        x = np.arange(len(features))
        width = 0.35
        
        vocoder_scores = [family_confusion[f]['Neural Vocoder'] for f in features]
        e2e_scores = [family_confusion[f]['Neural E2E'] for f in features]
        
        plt.bar(x - width/2, vocoder_scores, width, label='Neural Vocoder (A07-A12)', alpha=0.8)
        plt.bar(x + width/2, e2e_scores, width, label='Neural E2E (A13-A19)', alpha=0.8)
        
        plt.xlabel('Feature Type')
        plt.ylabel('Average Confusion Score')
        plt.title('Attack Family Confusion Analysis')
        plt.xticks(x, [f.upper() for f in features])
        plt.legend()
        
        # Plot 4: Prediction distribution for worst case
        plt.subplot(2, 3, 4)
        
        # Find the most confused attack
        max_confusion = 0
        worst_case = None
        worst_feature = None
        
        for feature in features:
            for attack in unknown_responses[feature]:
                confusion = unknown_responses[feature][attack]['confusion_score']
                if confusion > max_confusion:
                    max_confusion = confusion
                    worst_case = attack
                    worst_feature = feature
        
        if worst_case:
            pred_dist = unknown_responses[worst_feature][worst_case]['prediction_distribution']
            known_attack_names = list(self.known_attacks.values())
            
            plt.bar(known_attack_names, pred_dist, alpha=0.8)
            plt.title(f'Most Confused Case: {worst_feature.upper()} vs {worst_case}\nConfusion = {max_confusion:.3f}')
            plt.xlabel('Predicted as Known Attack')
            plt.ylabel('Probability')
            plt.xticks(rotation=45)
        
        # Plot 5: Performance ranking comparison
        plt.subplot(2, 3, 5)
        
        expected = h2_results['expected_ranking']
        actual = h2_results['actual_ranking']
        
        y_pos = np.arange(len(expected))
        
        plt.barh(y_pos - 0.2, [3, 2, 1], 0.4, label='Expected (H1)', alpha=0.7)
        
        actual_scores = []
        for exp_feat in expected:
            if exp_feat in actual:
                actual_scores.append(len(actual) - actual.index(exp_feat))
            else:
                actual_scores.append(0)
        
        plt.barh(y_pos + 0.2, actual_scores, 0.4, label='Actual (H2)', alpha=0.7)
        
        plt.yticks(y_pos, [f.upper() for f in expected])
        plt.xlabel('Ranking (Higher = Better)')
        plt.title('Expected vs Actual Performance Ranking')
        plt.legend()
        
        # Plot 6: Summary verdict
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        verdict_text = f"""
H2 ANALYSIS SUMMARY

Hypothesis: Features with higher intra-domain 
performance show better cross-domain generalization

Verdict: {h2_results['verdict']}

Correlation: {h2_results['correlation']:.3f}
Rank Correlation: {h2_results['rank_correlation']:.3f}

Expected Ranking: {' > '.join([f.upper() for f in expected])}
Actual Ranking: {' > '.join([f.upper() for f in actual])}

Ranking Match: {'Yes' if h2_results['ranking_match'] else 'No'}
        """
        
        plt.text(0.1, 0.5, verdict_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('h2_domain_transfer_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ H2 visualization saved as 'h2_domain_transfer_analysis.png'")
    
    def generate_h2_report(self, unknown_responses, transfer_metrics, h2_results):
        """
        Generate comprehensive H2 report
        """
        print(f"\n📝 H2 COMPREHENSIVE REPORT")
        print("="*50)
        
        report = {
            'hypothesis': 'H2: Domain Transfer Hypothesis',
            'statement': 'Features with higher intra-domain performance will show better cross-domain generalization',
            'verdict': h2_results['verdict'],
            'explanation': h2_results['explanation'],
            'correlation': h2_results['correlation'],
            'rank_correlation': h2_results['rank_correlation'],
            'statistical_significance': h2_results['rank_p_value'] < 0.05,
            'ranking_consistency': h2_results['ranking_match'],
            'detailed_findings': {},
            'implications': {}
        }
        
        # Detailed findings per feature
        for feature in transfer_metrics:
            report['detailed_findings'][feature] = {
                'known_performance': transfer_metrics[feature]['known_performance'],
                'generalization_ability': transfer_metrics[feature]['generalization_ability'],
                'unknown_confusion': transfer_metrics[feature]['unknown_confusion']
            }
        
        # Research implications
        if h2_results['verdict'] in ['STRONGLY SUPPORTED', 'SUPPORTED']:
            report['implications'] = {
                'practical': 'Features performing well on known attacks can be trusted for unknown attacks',
                'theoretical': 'Validates traditional ML assumption about feature generalization',
                'methodological': 'Justifies using known-attack performance for feature selection'
            }
        else:
            report['implications'] = {
                'practical': 'Need specialized approaches for unknown attack detection',
                'theoretical': 'Challenges assumptions about cross-domain generalization',
                'methodological': 'Requires domain adaptation techniques'
            }
        
        # Save detailed results
        with open('h2_analysis_results.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("💾 H2 results saved to 'h2_analysis_results.json'")
        
        return report

def run_h2_analysis():
    """
    Main function to run complete H2 analysis
    """
    print("🚀 STARTING H2 ANALYSIS: Domain Transfer Hypothesis")
    print("="*70)
    
    # Initialize analyzer
    analyzer = H2DomainTransferAnalyzer()
    
    if not analyzer.models:
        print("❌ No trained models found!")
        return None
    
    print(f"✅ H2 Analysis Setup Complete")
    print(f"📊 Testing {len(analyzer.models)} features on unknown attacks A07-A19")
    
    # Step 1: Analyze unknown attack responses
    unknown_responses = analyzer.analyze_unknown_attack_responses()
    
    # Step 2: Calculate transfer metrics
    transfer_metrics = analyzer.calculate_domain_transfer_metrics(unknown_responses)
    
    # Step 3: Test H2 correlation
    h2_results = analyzer.test_h2_correlation(transfer_metrics)
    
    # Step 4: Create visualizations
    analyzer.create_h2_visualizations(unknown_responses, transfer_metrics, h2_results)
    
    # Step 5: Generate comprehensive report
    report = analyzer.generate_h2_report(unknown_responses, transfer_metrics, h2_results)
    
    print(f"\n🎉 H2 ANALYSIS COMPLETE!")
    print("📊 Generated files:")
    print("  • h2_domain_transfer_analysis.png (comprehensive visualization)")
    print("  • h2_analysis_results.json (detailed results)")
    print(f"\n🎯 H2 VERDICT: {h2_results['verdict']}")
    print(f"🔬 Ready for next hypothesis or feature fusion experiments!")
    
    return {
        'unknown_responses': unknown_responses,
        'transfer_metrics': transfer_metrics,
        'h2_results': h2_results,
        'report': report
    }

if __name__ == "__main__":
    # Run H2 analysis
    results = run_h2_analysis()