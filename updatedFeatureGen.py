import os
import numpy as np
import librosa
from tqdm import tqdm

class FeatureExtractorCorrected:
    """
    extract features with corrected labels
    """
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
    def extract_mfcc(self, y, n_mfcc=13):
        """get MFCC features"""
        return librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=n_mfcc)
    
    def extract_cqt(self, y):
        """get CQT features"""
        return np.abs(librosa.cqt(y=y, sr=self.sample_rate))
    
    def extract_lpc(self, y, order=16):
        """get LPC coefficients"""
        frame_length = int(0.025 * self.sample_rate)
        if len(y) < frame_length:
            frame = y
        else:
            frame = y[:frame_length]
        return librosa.lpc(y=frame, order=order)
    
    def process_audio_file(self, audio_path):
        """process one audio file"""
        try:
            # load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # normalize
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))
            
            # extract all features
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
        """extract features with corrected labels"""
        print(f"\n{'='*60}")
        print(f"EXTRACTING FEATURES: {split_name.upper()} (CORRECTED LABELS)")
        print(f"{'='*60}")
        
        # load corrected data
        corrected_dir = f"{split_name}_data_corrected"
        
        if not os.path.exists(corrected_dir):
            print(f"Corrected data directory not found: {corrected_dir}")
            print("Please run the corrected parser first!")
            return None
        
        file_paths = np.load(os.path.join(corrected_dir, 'file_paths.npy'), allow_pickle=True)
        labels = np.load(os.path.join(corrected_dir, 'labels.npy'))
        
        print(f"Processing {len(file_paths)} files...")
        print(f"Label distribution: {np.bincount(labels)}")
        
        # extract features
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
            
            # progress every 1000 files
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(file_paths)} files...")
        
        print(f"Successfully processed {processed_count}/{len(file_paths)} files")
        
        # separate by feature type
        mfcc_features = [f['mfcc'] for f in all_features]
        cqt_features = [f['cqt'] for f in all_features]
        lpc_features = [f['lpc'] for f in all_features]
        
        # convert to numpy arrays
        mfcc_arr = np.empty(len(mfcc_features), dtype=object)
        cqt_arr = np.empty(len(cqt_features), dtype=object)
        lpc_arr = np.empty(len(lpc_features), dtype=object)
        
        for i in range(len(mfcc_features)):
            mfcc_arr[i] = mfcc_features[i]
            cqt_arr[i] = cqt_features[i]
            lpc_arr[i] = lpc_features[i]
        
        labels_arr = np.array(valid_labels)
        
        # save features
        output_dir = f"{split_name}_features_corrected"
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, 'mfcc_features.npy'), mfcc_arr, allow_pickle=True)
        np.save(os.path.join(output_dir, 'cqt_features.npy'), cqt_arr, allow_pickle=True)
        np.save(os.path.join(output_dir, 'lpc_features.npy'), lpc_arr, allow_pickle=True)
        np.save(os.path.join(output_dir, 'labels.npy'), labels_arr)
        
        print(f"Corrected features saved to {output_dir}/")
        print(f"  - mfcc_features.npy: {len(mfcc_features)} samples")
        print(f"  - cqt_features.npy: {len(cqt_features)} samples")
        print(f"  - lpc_features.npy: {len(lpc_features)} samples")
        print(f"  - labels.npy: {len(valid_labels)} labels")
        
        # show label distribution
        unique_labels, counts = np.unique(valid_labels, return_counts=True)
        print(f"\nCORRECTED Label Distribution:")
        class_names = ['bonafide'] + [f'A{i:02d}' for i in range(1, 20)]
        
        for label, count in zip(unique_labels, counts):
            class_name = class_names[label] if label < len(class_names) else f"Class_{label}"
            print(f"  {class_name}: {count:,}")
        
        # feature dimensions
        if len(mfcc_features) > 0:
            print(f"\nFeature Dimensions:")
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
        """extract features for all splits"""
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
                print(f"Failed to extract features for {split_name}")
        
        print(f"\n{'='*80}")
        print("CORRECTED FEATURE EXTRACTION SUMMARY")
        print(f"{'='*80}")
        
        total_samples = 0
        for split_name, result in results.items():
            num_samples = result['num_samples']
            total_samples += num_samples
            
            print(f"{split_name.upper():>5}: {num_samples:>6,} samples")
            
            # label breakdown
            label_dist = result['label_distribution']
            bonafide_count = label_dist.get(0, 0)
            attack_count = sum(count for label, count in label_dist.items() if label > 0)
            unique_classes = len(label_dist)
            
            print(f"       {bonafide_count:>6,} bonafide + {attack_count:>6,} attacks = {unique_classes} classes")
        
        print(f"{'TOTAL':>5}: {total_samples:>6,} samples")
        
        print(f"\nCorrected feature directories created:")
        for split_name in results.keys():
            print(f"  - {split_name}_features_corrected/")
        
        print(f"\nReady for training with realistic results!")
        
        return results


if __name__ == "__main__":
    print("extracting audio features")
    print("="*40)
    
    extractor = FeatureExtractorCorrected()
    results = extractor.extract_all_features_corrected()
    
    if results:
        print("\ndone")
        print("extracted features for training")
    else:
        print("\nsomething went wrong")