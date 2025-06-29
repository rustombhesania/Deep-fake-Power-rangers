import os
import numpy as np
import librosa
from tqdm import tqdm

class FeatureExtractorCorrected:
    """
    Feature extractor using corrected labels
    """
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
    def extract_mfcc(self, y, n_mfcc=13):
        """Extract MFCC features"""
        return librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=n_mfcc)
    
    def extract_cqt(self, y):
        """Extract Constant-Q Transform features"""
        return np.abs(librosa.cqt(y=y, sr=self.sample_rate))
    
    def extract_lpc(self, y, order=16):
        """Extract Linear Predictive Coding coefficients"""
        frame_length = int(0.025 * self.sample_rate)
        if len(y) < frame_length:
            frame = y
        else:
            frame = y[:frame_length]
        return librosa.lpc(y=frame, order=order)
    
    def process_audio_file(self, audio_path):
        """Process single audio file and extract all features"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Normalize
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))
            
            # Extract features
            features = {
                'mfcc': self.extract_mfcc(y),
                'cqt': self.extract_cqt(y),
                'lpc': self.extract_lpc(y)
            }
            
            return features
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def extract_features_for_split_corrected(self, split_name):
        """Extract features using corrected labels"""
        print(f"\n{'='*60}")
        print(f"EXTRACTING FEATURES: {split_name.upper()} (CORRECTED LABELS)")
        print(f"{'='*60}")
        
        # Load corrected data
        corrected_dir = f"{split_name}_data_corrected"
        
        if not os.path.exists(corrected_dir):
            print(f"❌ Corrected data directory not found: {corrected_dir}")
            print("Please run the corrected parser first!")
            return None
        
        file_paths = np.load(os.path.join(corrected_dir, 'file_paths.npy'), allow_pickle=True)
        labels = np.load(os.path.join(corrected_dir, 'labels.npy'))
        
        print(f"📊 Processing {len(file_paths)} files...")
        print(f"📊 Label distribution: {np.bincount(labels)}")
        
        # Extract features
        all_features = []
        valid_labels = []
        processed_count = 0
        
        for i, (file_path, label) in enumerate(tqdm(zip(file_paths, labels), 
                                                   total=len(file_paths),
                                                   desc=f"Extracting {split_name}")):
            features = self.process_audio_file(file_path)
            
            if features is not None:
                all_features.append(features)
                valid_labels.append(label)
                processed_count += 1
            
            # Progress update every 1000 files
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(file_paths)} files...")
        
        print(f"✅ Successfully processed {processed_count}/{len(file_paths)} files")
        
        # Separate features by type
        mfcc_features = [f['mfcc'] for f in all_features]
        cqt_features = [f['cqt'] for f in all_features]
        lpc_features = [f['lpc'] for f in all_features]
        
        # Convert to numpy arrays (object dtype for variable length)
        mfcc_arr = np.empty(len(mfcc_features), dtype=object)
        cqt_arr = np.empty(len(cqt_features), dtype=object)
        lpc_arr = np.empty(len(lpc_features), dtype=object)
        
        for i in range(len(mfcc_features)):
            mfcc_arr[i] = mfcc_features[i]
            cqt_arr[i] = cqt_features[i]
            lpc_arr[i] = lpc_features[i]
        
        labels_arr = np.array(valid_labels)
        
        # Save features with corrected labels
        output_dir = f"{split_name}_features_corrected"
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, 'mfcc_features.npy'), mfcc_arr, allow_pickle=True)
        np.save(os.path.join(output_dir, 'cqt_features.npy'), cqt_arr, allow_pickle=True)
        np.save(os.path.join(output_dir, 'lpc_features.npy'), lpc_arr, allow_pickle=True)
        np.save(os.path.join(output_dir, 'labels.npy'), labels_arr)
        
        print(f"✅ Corrected features saved to {output_dir}/")
        print(f"  - mfcc_features.npy: {len(mfcc_features)} samples")
        print(f"  - cqt_features.npy: {len(cqt_features)} samples")
        print(f"  - lpc_features.npy: {len(lpc_features)} samples")
        print(f"  - labels.npy: {len(valid_labels)} labels")
        
        # Show corrected label distribution
        unique_labels, counts = np.unique(valid_labels, return_counts=True)
        print(f"\n📊 CORRECTED Label Distribution:")
        class_names = ['bonafide'] + [f'A{i:02d}' for i in range(1, 20)]
        
        for label, count in zip(unique_labels, counts):
            class_name = class_names[label] if label < len(class_names) else f"Class_{label}"
            print(f"  {class_name}: {count:,}")
        
        # Print feature dimensions
        if len(mfcc_features) > 0:
            print(f"\n📊 Feature Dimensions:")
            print(f"  MFCC: {mfcc_features[0].shape}")
            print(f"  CQT:  {cqt_features[0].shape}")
            print(f"  LPC:  {lpc_features[0].shape}")
        
        return {
            'output_dir': output_dir,
            'num_samples': len(valid_labels),
            'label_distribution': dict(zip(unique_labels, counts)),
            'feature_shapes': {
                'mfcc': mfcc_features[0].shape if mfcc_features else None,
                'cqt': cqt_features[0].shape if cqt_features else None,
                'lpc': lpc_features[0].shape if lpc_features else None
            }
        }
    
    def extract_all_features_corrected(self):
        """Extract features for all splits using corrected labels"""
        print("="*80)
        print("EXTRACTING FEATURES WITH CORRECTED LABELS")
        print("="*80)
        
        splits = ['train', 'dev', 'eval']
        results = {}
        
        for split_name in splits:
            result = self.extract_features_for_split_corrected(split_name)
            if result:
                results[split_name] = result
            else:
                print(f"❌ Failed to extract features for {split_name}")
        
        print(f"\n{'='*80}")
        print("CORRECTED FEATURE EXTRACTION SUMMARY")
        print(f"{'='*80}")
        
        total_samples = 0
        for split_name, result in results.items():
            num_samples = result['num_samples']
            total_samples += num_samples
            
            print(f"{split_name.upper():>5}: {num_samples:>6,} samples")
            
            # Show label breakdown
            label_dist = result['label_distribution']
            bonafide_count = label_dist.get(0, 0)
            attack_count = sum(count for label, count in label_dist.items() if label > 0)
            unique_classes = len(label_dist)
            
            print(f"       {bonafide_count:>6,} bonafide + {attack_count:>6,} attacks = {unique_classes} classes")
        
        print(f"{'TOTAL':>5}: {total_samples:>6,} samples")
        
        print(f"\n✅ Corrected feature directories created:")
        for split_name in results.keys():
            print(f"  - {split_name}_features_corrected/")
        
        print(f"\n🎯 Ready for training with REALISTIC results!")
        
        return results


def cleanup_old_files():
    """Clean up old incorrect feature files"""
    print("🧹 CLEANING UP OLD INCORRECT FILES")
    print("="*50)
    
    old_dirs = [
        'train_features', 'dev_features', 'eval_features',
        'train_data', 'dev_data', 'eval_data'
    ]
    
    import shutil
    
    for dir_name in old_dirs:
        if os.path.exists(dir_name):
            try:
                shutil.rmtree(dir_name)
                print(f"✅ Deleted: {dir_name}/")
            except Exception as e:
                print(f"⚠️  Could not delete {dir_name}: {e}")
        else:
            print(f"  {dir_name}/ (not found)")
    
    print("\n🧹 Cleanup complete!")


def verify_corrected_labels():
    """Quick verification that corrected labels look right"""
    print("\n🔍 VERIFYING CORRECTED LABELS")
    print("="*50)
    
    for split_name in ['train', 'dev', 'eval']:
        corrected_dir = f"{split_name}_data_corrected"
        
        if os.path.exists(corrected_dir):
            labels = np.load(os.path.join(corrected_dir, 'labels.npy'))
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            print(f"\n{split_name.upper()}:")
            print(f"  Total samples: {len(labels):,}")
            print(f"  Unique classes: {len(unique_labels)}")
            print(f"  Bonafide: {counts[unique_labels == 0][0] if 0 in unique_labels else 0:,}")
            print(f"  Attacks: {len(labels) - (counts[unique_labels == 0][0] if 0 in unique_labels else 0):,}")
        else:
            print(f"\n{split_name.upper()}: ❌ Corrected data not found!")


if __name__ == "__main__":
    print("🔧 REGENERATING FEATURES WITH CORRECTED LABELS")
    print("="*60)
    
    # First verify we have corrected labels
    verify_corrected_labels()
    
    # Ask user before cleaning up
    print(f"\n⚠️  About to delete old incorrect feature files...")
    response = input("Continue? (y/n): ").lower().strip()
    
    if response == 'y':
        # Clean up old files
        cleanup_old_files()
        
        # Extract features with corrected labels
        extractor = FeatureExtractorCorrected()
        results = extractor.extract_all_features_corrected()
        
        if results:
            print(f"\n🎉 SUCCESS!")
            print(f"✅ Features regenerated with corrected labels")
            print(f"🎯 Now train models and expect realistic ~80-95% accuracy")
            print(f"📁 Use these directories for training:")
            for split_name in results.keys():
                print(f"   - {split_name}_features_corrected/")
        else:
            print(f"\n❌ Feature extraction failed!")
    else:
        print("❌ Cancelled. Run again when ready.")