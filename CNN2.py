import os
import numpy as np
import librosa
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class ASVspoofDataset(Dataset):
    """Custom Dataset for ASVspoof2019"""
    def __init__(self, features, labels):
        # Features for CNN should be (batch_size, n_mfcc, max_pad_len)
        # So we permute them here from (num_samples, max_pad_len, n_mfcc) to (num_samples, n_mfcc, max_pad_len)
        self.features = torch.FloatTensor(features).permute(0, 2, 1)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class CNNBaseline(nn.Module):
    """Convolutional Neural Network baseline for ASVspoof detection"""
    def __init__(self, n_mfcc, max_pad_len, num_classes=2, dropout=0.3):
        super(CNNBaseline, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_mfcc, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Calculate the output size of the convolutional layers
        # Input shape: (batch_size, n_mfcc, max_pad_len)
        # After conv1: max_pad_len / 2
        # After conv2: max_pad_len / 4
        # After conv3: max_pad_len / 8
        self._output_size = self._get_conv_output_size(n_mfcc, max_pad_len)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self._output_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def _get_conv_output_size(self, n_mfcc, max_pad_len):
        # Create a dummy tensor to pass through the convolutional layers
        # Input: (batch_size, channels, length)
        dummy_input = torch.zeros(1, n_mfcc, max_pad_len)
        output = self.conv1(dummy_input)
        output = self.conv2(output)
        output = self.conv3(output)
        return output.flatten().shape[0]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Flatten the output for the fully connected layers
        x = x.flatten(start_dim=1)
        
        x = self.fc_layers(x)
        return x

class ASVspoof2019Detector:
    def __init__(self, data_path, n_mfcc=13, n_fft=2048, hop_length=512, max_pad_len=None):
        """
        Initialize the ASVspoof2019 detector
        
        Args:
            data_path: Path to ASVspoof2019 dataset
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for STFT
            max_pad_len: Maximum length for padding (if None, will be calculated)
        """
        self.data_path = data_path
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_pad_len = max_pad_len
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def load_protocol_file(self, protocol_file):
        """Load ASVspoof2019 protocol file"""
        protocols = []
        with open(protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) >= 5:
                    speaker_id = parts[0]
                    filename = parts[1]
                    system_id = parts[2] if parts[2] != '-' else 'bonafide'
                    label = parts[4]  # 'bonafide' or 'spoof'
                    protocols.append({
                        'speaker_id': speaker_id,
                        'filename': filename,
                        'system_id': system_id,
                        'label': label
                    })
        return pd.DataFrame(protocols)
    
    def extract_mfcc_features(self, audio_file):
        """Extract MFCC features from audio file"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_file, sr=None)
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=y, 
                sr=sr, 
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Transpose to have time frames as rows (max_pad_len, n_mfcc)
            mfcc = mfcc.T
            
            return mfcc
            
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
            return None
    
    def check_and_pad_features(self, features_list):
        """Check lengths and pad features to same length"""
        lengths = [feat.shape[0] for feat in features_list if feat is not None]
        
        print(f"Feature lengths statistics:")
        print(f"Min length: {min(lengths)}")
        print(f"Max length: {max(lengths)}")
        print(f"Mean length: {np.mean(lengths):.2f}")
        print(f"Std length: {np.std(lengths):.2f}")
        
        # Set max padding length
        if self.max_pad_len is None:
            # Find the 95th percentile length to avoid padding to extremely long samples
            self.max_pad_len = int(np.percentile(lengths, 95))
            print(f"Setting max_pad_len to 95th percentile length: {self.max_pad_len}")
            
        print(f"Padding/Truncating all features to length: {self.max_pad_len}")
        
        padded_features = []
        for feat in features_list:
            if feat is not None:
                if feat.shape[0] > self.max_pad_len:
                    # Truncate if longer
                    feat = feat[:self.max_pad_len, :]
                elif feat.shape[0] < self.max_pad_len:
                    # Pad if shorter
                    pad_width = self.max_pad_len - feat.shape[0]
                    feat = np.pad(feat, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)
                
                padded_features.append(feat)
            else:
                # Create zero array for failed extractions
                padded_features.append(np.zeros((self.max_pad_len, self.n_mfcc)))
        
        return np.array(padded_features) # Return as numpy array for easier scaling
    
    def prepare_data(self, protocol_df, audio_dir):
        """Prepare data for training/testing"""
        print("Extracting MFCC features...")
        
        features = []
        labels = []
        
        for idx, row in tqdm(protocol_df.iterrows(), total=len(protocol_df)):
            audio_path = os.path.join(audio_dir, f"{row['filename']}.flac")
            
            if not os.path.exists(audio_path):
                # print(f"Audio file not found: {audio_path}") # This can be very verbose
                continue
            
            # Extract features
            mfcc_feat = self.extract_mfcc_features(audio_path)
            if mfcc_feat is not None:
                features.append(mfcc_feat)
                # Convert label to binary (0: bonafide, 1: spoof)
                labels.append(0 if row['label'] == 'bonafide' else 1)
        
        print(f"Successfully processed {len(features)} audio files")
        
        # Check lengths and pad
        padded_features = self.check_and_pad_features(features)
        
        # Reshape for StandardScaler: (num_samples * max_pad_len, n_mfcc)
        original_shape = padded_features.shape
        reshaped_features = padded_features.reshape(-1, self.n_mfcc)
        
        print("Scaling features...")
        scaled_features = self.scaler.fit_transform(reshaped_features)
        
        # Reshape back to (num_samples, max_pad_len, n_mfcc)
        scaled_features = scaled_features.reshape(original_shape)
        
        return scaled_features, np.array(labels)
    
    def build_model(self, n_mfcc, max_pad_len, num_classes=2, dropout=0.3):
        """Build CNN model"""
        self.model = CNNBaseline(
            n_mfcc=n_mfcc,
            max_pad_len=max_pad_len,
            num_classes=num_classes,
            dropout=dropout
        ).to(self.device)
        
        return self.model
    
    def train_model(self, X_train, y_train, X_val, y_val, 
                   batch_size=32, epochs=100, lr=0.001, patience=10):
        """Train the CNN model"""
        
        # Normalize features - Already done in prepare_data now
        # However, for validation and test set, we need to transform using the scaler fitted on training data.
        # This is handled in evaluate function. Here for train/val, the features are already scaled.
        
        # Create datasets and dataloaders
        train_dataset = ASVspoofDataset(X_train, y_train)
        val_dataset = ASVspoofDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training history
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        best_val_acc = 0
        patience_counter = 0
        
        print("Starting training...")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()
            
            # Calculate metrics
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss / len(val_loader))
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            print(f'Epoch [{epoch+1}/{epochs}] - '
                  f'Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss/len(val_loader):.4f}, '
                  f'Val Acc: {val_acc:.2f}%')
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }
    
    def calculate_eer(self, y_true, y_scores):
        """Calculate Equal Error Rate (EER)"""
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
        fnr = 1 - tpr
        
        # Find the threshold where FPR and FNR are closest
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        
        return eer * 100, eer_threshold  # Return EER as percentage
        
    def evaluate(self, X_test, y_test, batch_size=32):
        """Evaluate the model"""
        # Reshape for StandardScaler: (num_samples * max_pad_len, n_mfcc)
        original_shape = X_test.shape
        reshaped_X_test = X_test.reshape(-1, self.n_mfcc)
        X_test_scaled = self.scaler.transform(reshaped_X_test)
        X_test_scaled = X_test_scaled.reshape(original_shape)

        test_dataset = ASVspoofDataset(X_test_scaled, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        predictions = []
        true_labels = []
        scores = []  # For EER calculation
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(self.device)
                outputs = self.model(batch_features)
                
                # Get probabilities for EER calculation
                probs = F.softmax(outputs, dim=1)
                spoof_scores = probs[:, 1].cpu().numpy()  # Probability of spoof class
                
                _, predicted = torch.max(outputs.data, 1)
                
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(batch_labels.numpy())
                scores.extend(spoof_scores)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions, 
                                     target_names=['bonafide', 'spoof'])
        cm = confusion_matrix(true_labels, predictions)
        
        # Calculate EER
        eer, eer_threshold = self.calculate_eer(true_labels, scores)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Equal Error Rate (EER): {eer:.4f}%")
        print(f"EER Threshold: {eer_threshold:.4f}")
        print("\nClassification Report:")
        print(report)
        
        return {
            'accuracy': accuracy,
            'eer': eer,
            'eer_threshold': eer_threshold,
            'predictions': predictions,
            'true_labels': true_labels,
            'scores': scores,
            'confusion_matrix': cm
        }
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(history['train_losses'], label='Training Loss')
        ax1.plot(history['val_losses'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy plot
        ax2.plot(history['train_accs'], label='Training Accuracy')
        ax2.plot(history['val_accs'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, y_true, y_scores, eer):
        """Plot ROC curve with EER point"""
        fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
        
        # Mark EER point
        eer_point = eer / 100
        plt.plot(eer_point, 1 - eer_point, 'ro', markersize=8, 
                label=f'EER = {eer:.2f}%')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve with Equal Error Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['bonafide', 'spoof'],
                   yticklabels=['bonafide', 'spoof'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = ASVspoof2019Detector(
        data_path="/home/rben10/team_labs/ASVSpoof19/LA",
        n_mfcc=13,
        n_fft=2048,
        hop_length=512
    )
    
    # Base path for dataset
    base_path = "/home/rben10/team_labs/ASVSpoof19/LA"
    
    # Load protocol files with full paths
    train_protocol = detector.load_protocol_file(os.path.join(base_path, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt"))
    dev_protocol = detector.load_protocol_file(os.path.join(base_path, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.dev.trl.txt"))
    eval_protocol = detector.load_protocol_file(os.path.join(base_path, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.eval.trl.txt"))
    
    # Prepare training data
    # prepare_data now returns scaled features, ready for ASVspoofDataset
    X_train, y_train = detector.prepare_data(train_protocol, os.path.join(base_path, "ASVspoof2019_LA_train", "flac"))
    X_dev, y_dev = detector.prepare_data(dev_protocol, os.path.join(base_path, "ASVspoof2019_LA_dev", "flac"))
    X_eval, y_eval = detector.prepare_data(eval_protocol, os.path.join(base_path, "ASVspoof2019_LA_eval", "flac"))
    
    print(f"Training set: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Development set: {X_dev.shape}, Labels: {y_dev.shape}")
    print(f"Evaluation set: {X_eval.shape}, Labels: {y_eval.shape}")
    
    # Build model (now a CNN)
    n_mfcc = detector.n_mfcc
    max_pad_len = detector.max_pad_len
    model = detector.build_model(n_mfcc=n_mfcc, max_pad_len=max_pad_len)
    print(model)
    
    # Train model
    history = detector.train_model(
        X_train, y_train, X_dev, y_dev,
        batch_size=32, epochs=100, lr=0.001, patience=10
    )
    
    # Evaluate on test set
    results = detector.evaluate(X_eval, y_eval)
    
    # Plot training history
    detector.plot_training_history(history)
    
    # Plot ROC curve with EER
    detector.plot_roc_curve(results['true_labels'], results['scores'], results['eer'])
    
    # Plot confusion matrix
    detector.plot_confusion_matrix(results['confusion_matrix'])