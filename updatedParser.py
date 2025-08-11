import os
import numpy as np
import pandas as pd
from collections import Counter

def correct_parse_protocol_file(protocol_file):
    """
    CORRECT parser for ASVspoof2019 protocol files
    
    Format: SPEAKER_ID FILE_ID SYSTEM_ID_1 SYSTEM_ID_2 LABEL
    Examples:
    - Bonafide: LA_0079 LA_T_1138215 - - bonafide
    - Attack:   LA_0039 LA_E_2834763 - A11 spoof
    """
    print(f"🔧 CORRECT PARSING: {protocol_file}")
    
    if not os.path.exists(protocol_file):
        print(f"❌ File not found: {protocol_file}")
        return None
    
    data = []
    with open(protocol_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) < 5:
                print(f"⚠️  Line {line_num} has incorrect format: {line.strip()}")
                continue
            
            speaker_id = parts[0]      # LA_0079
            file_id = parts[1]         # LA_T_1138215
            system_id_1 = parts[2]     # Always '-'
            system_id_2 = parts[3]     # '-' for bonafide, 'A01'-'A19' for attacks
            label_text = parts[4]      # 'bonafide' or 'spoof'
            
            # THE KEY FIX: Use system_id_2 (column 4) for attack detection
            if system_id_2 == '-':
                # Bonafide samples have system_id_2 = '-'
                numeric_label = 0
                class_name = 'bonafide'
            else:
                # Attack samples have system_id_2 like 'A01', 'A02', etc.
                try:
                    # Extract number from 'A01' -> 1, 'A02' -> 2, etc.
                    attack_num = int(system_id_2[1:])  # Remove 'A' and convert to int
                    numeric_label = attack_num  # A01=1, A02=2, ..., A19=19
                    class_name = system_id_2
                except (ValueError, IndexError):
                    print(f"⚠️  Unexpected system_id_2 format: {system_id_2} on line {line_num}")
                    continue
            
            # Debug: Print first few entries
            if line_num <= 10:
                print(f"  Line {line_num}: {system_id_2} -> label {numeric_label} ({class_name})")
            
            data.append({
                'speaker_id': speaker_id,
                'file_id': file_id,
                'audio_file': f"{file_id}.flac",
                'system_id_1': system_id_1,
                'system_id_2': system_id_2,  # This is the important one!
                'label_text': label_text,
                'numeric_label': numeric_label,
                'class_name': class_name
            })
    
    df = pd.DataFrame(data)
    print(f"✅ Parsed {len(df)} entries")
    
    # Show CORRECTED label distribution
    print(f"\n📊 CORRECTED Label Distribution:")
    label_counts = df['class_name'].value_counts().sort_index()
    for class_name, count in label_counts.items():
        print(f"  {class_name}: {count}")
    
    # Verify we have attacks
    attack_count = len(df[df['numeric_label'] > 0])
    bonafide_count = len(df[df['numeric_label'] == 0])
    unique_classes = df['numeric_label'].nunique()
    
    print(f"\n✅ VERIFICATION:")
    print(f"  Bonafide samples: {bonafide_count:,}")
    print(f"  Attack samples: {attack_count:,}")
    print(f"  Total classes: {unique_classes}")
    print(f"  Attack systems: {sorted(df[df['numeric_label'] > 0]['class_name'].unique())}")
    
    if attack_count == 0:
        print("🚨 STILL NO ATTACKS FOUND!")
    else:
        print("✅ SUCCESS: Attack samples found!")
    
    return df


def test_corrected_parser(base_path):
    """
    Test the corrected parser on all protocol files
    """
    print("🧪 TESTING CORRECTED PARSER")
    print("="*50)
    
    protocol_files = {
        'train': os.path.join(base_path, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt"),
        'dev': os.path.join(base_path, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.dev.trl.txt"),
        'eval': os.path.join(base_path, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.eval.trl.txt")
    }
    
    all_results = {}
    
    for split_name, protocol_file in protocol_files.items():
        print(f"\n{'='*40}")
        print(f"TESTING {split_name.upper()} WITH CORRECTED PARSER")
        print(f"{'='*40}")
        
        df = correct_parse_protocol_file(protocol_file)
        
        if df is not None:
            all_results[split_name] = df
            
            # Show detailed breakdown
            print(f"\n📋 DETAILED BREAKDOWN FOR {split_name.upper()}:")
            attack_systems = df[df['numeric_label'] > 0]['class_name'].value_counts().sort_index()
            bonafide_count = len(df[df['numeric_label'] == 0])
            
            print(f"  bonafide: {bonafide_count:,}")
            for system, count in attack_systems.items():
                print(f"  {system}: {count:,}")
            
            # Show sample entries
            print(f"\n📋 SAMPLE ENTRIES:")
            sample_df = df.head(10)[['file_id', 'system_id_2', 'label_text', 'numeric_label', 'class_name']]
            print(sample_df.to_string(index=False))
    
    return all_results


def generate_corrected_labels(base_path):
    """
    Generate corrected labels using the fixed parser
    """
    print("\n🔧 GENERATING CORRECTED LABELS")
    print("="*50)
    
    protocol_files = {
        'train': os.path.join(base_path, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt"),
        'dev': os.path.join(base_path, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.dev.trl.txt"),
        'eval': os.path.join(base_path, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.eval.trl.txt")
    }
    
    audio_dirs = {
        'train': os.path.join(base_path, "ASVspoof2019_LA_train", "flac"),
        'dev': os.path.join(base_path, "ASVspoof2019_LA_dev", "flac"),
        'eval': os.path.join(base_path, "ASVspoof2019_LA_eval", "flac")
    }
    
    for split_name, protocol_file in protocol_files.items():
        print(f"\n🔧 Processing {split_name}...")
        
        # Parse with corrected function
        df = correct_parse_protocol_file(protocol_file)
        
        if df is not None:
            # Filter to only include existing audio files
            audio_dir = audio_dirs[split_name]
            existing_files = []
            existing_labels = []
            existing_paths = []
            
            for _, row in df.iterrows():
                audio_path = os.path.join(audio_dir, row['audio_file'])
                if os.path.exists(audio_path):
                    existing_files.append(row['audio_file'])
                    existing_labels.append(row['numeric_label'])
                    existing_paths.append(audio_path)
            
            # Save corrected labels
            output_dir = f"{split_name}_data_corrected"
            os.makedirs(output_dir, exist_ok=True)
            
            np.save(os.path.join(output_dir, 'file_list.npy'), existing_files)
            np.save(os.path.join(output_dir, 'labels.npy'), existing_labels)
            np.save(os.path.join(output_dir, 'file_paths.npy'), existing_paths)
            
            # Save metadata
            existing_df = df[df['audio_file'].isin(existing_files)].copy()
            existing_df.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)
            
            print(f"✅ Corrected {split_name} labels saved to {output_dir}/")
            
            # Show final statistics
            label_counts = Counter(existing_labels)
            bonafide_count = label_counts[0]
            attack_count = sum(count for label, count in label_counts.items() if label > 0)
            unique_classes = len(set(existing_labels))
            
            print(f"  📊 {len(existing_files):,} total files")
            print(f"  📊 {bonafide_count:,} bonafide samples")
            print(f"  📊 {attack_count:,} attack samples") 
            print(f"  📊 {unique_classes} unique classes")


if __name__ == "__main__":
    # UPDATE THIS PATH TO YOUR ASVspoof2019 DIRECTORY
    base_path = r"C:\ASVSpoof19\LA"
    
    print("🔧 CORRECT ASVSPOOF PROTOCOL PARSER")
    print("="*50)
    
    # Test corrected parser
    results = test_corrected_parser(base_path)
    
    print(f"\n{'='*70}")
    print("📊 OVERALL SUMMARY")
    print(f"{'='*70}")
    
    total_bonafide = 0
    total_attacks = 0
    
    for split_name, df in results.items():
        bonafide = len(df[df['numeric_label'] == 0])
        attacks = len(df[df['numeric_label'] > 0])
        total_bonafide += bonafide
        total_attacks += attacks
        
        print(f"{split_name.upper():>5}: {bonafide:>6,} bonafide + {attacks:>6,} attacks = {len(df):>6,} total")
    
    print(f"{'TOTAL':>5}: {total_bonafide:>6,} bonafide + {total_attacks:>6,} attacks = {total_bonafide + total_attacks:>6,} total")
    
    # Generate corrected labels
    print(f"\n🔧 Generating corrected label files...")
    generate_corrected_labels(base_path)
    
    print(f"\n✅ CORRECTED LABELS GENERATED!")
    print(f"📁 Use these directories for training:")
    print(f"  - train_data_corrected/")
    print(f"  - dev_data_corrected/")
    print(f"  - eval_data_corrected/")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. Extract features using corrected labels")
    print(f"2. Train models - expect ~80-95% accuracy")
    print(f"3. Celebrate realistic results! 🎉")