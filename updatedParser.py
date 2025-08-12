import os
import numpy as np
import pandas as pd
from collections import Counter

def parse_protocol_file(protocol_file):
    """
    Parse ASVspoof2019 protocol files
    Format: SPEAKER_ID FILE_ID SYSTEM_ID_1 SYSTEM_ID_2 LABEL
    """
    print(f"Reading: {protocol_file}")
    
    if not os.path.exists(protocol_file):
        print("File not found")
        return None
    
    data = []
    f = open(protocol_file, 'r')
    lines = f.readlines()
    f.close()
    
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        speaker_id = parts[0]
        file_id = parts[1]
        system_id_1 = parts[2]
        system_id_2 = parts[3]  # this should be the attack type
        label_text = parts[4]
        
        # figure out the label from system_id_2
        if system_id_2 == '-':
            numeric_label = 0
            class_name = 'bonafide'
        else:
            # should be like A01, A02, etc
            attack_num = int(system_id_2[1:])
            numeric_label = attack_num
            class_name = system_id_2
        
        data.append({
            'speaker_id': speaker_id,
            'file_id': file_id,
            'audio_file': f"{file_id}.flac",
            'system_id_1': system_id_1,
            'system_id_2': system_id_2,
            'label_text': label_text,
            'numeric_label': numeric_label,
            'class_name': class_name
        })
    
    df = pd.DataFrame(data)
    print(f"Got {len(df)} entries")
    
    # check what we have
    label_counts = df['class_name'].value_counts()
    for class_name, count in label_counts.items():
        print(f"{class_name}: {count}")
    
    return df


def test_parser(base_path):
    """
    Test parser on train/dev/eval files
    """
    print("Testing parser...")
    
    # TODO: make this more flexible for different paths
    protocol_files = {
        'train': os.path.join(base_path, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt"),
        'dev': os.path.join(base_path, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.dev.trl.txt"),
        'eval': os.path.join(base_path, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.eval.trl.txt")
    }
    
    results = {}
    
    for split_name, protocol_file in protocol_files.items():
        print(f"\n--- {split_name} ---")
        df = parse_protocol_file(protocol_file)
        
        if df is not None:
            results[split_name] = df
            
            # show some stats
            bonafide_count = len(df[df['numeric_label'] == 0])
            attack_count = len(df[df['numeric_label'] > 0])
            
            print(f"bonafide: {bonafide_count}")
            print(f"attacks: {attack_count}")
    
    return results


def save_labels(base_path):
    """
    Save the parsed labels to files
    """
    print("Saving labels...")
    
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
    
    for split_name in protocol_files.keys():
        print(f"Processing {split_name}...")
        
        df = parse_protocol_file(protocol_files[split_name])
        
        if df is not None:
            # check which files actually exist
            audio_dir = audio_dirs[split_name]
            existing_files = []
            existing_labels = []
            
            for idx, row in df.iterrows():
                audio_path = os.path.join(audio_dir, row['audio_file'])
                if os.path.exists(audio_path):
                    existing_files.append(row['audio_file'])
                    existing_labels.append(row['numeric_label'])
            
            # save to numpy files
            output_dir = f"{split_name}_data"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            np.save(os.path.join(output_dir, 'file_list.npy'), existing_files)
            np.save(os.path.join(output_dir, 'labels.npy'), existing_labels)
            
            print(f"Saved {len(existing_files)} files to {output_dir}")


if __name__ == "__main__":
    # change this to your data path
    base_path = r"C:\ASVSpoof19\LA"
    
    print("ASVspoof parser")
    
    # test the parser
    results = test_parser(base_path)
    
    print("\nSummary:")
    total_bonafide = 0
    total_attacks = 0
    
    for split_name, df in results.items():
        bonafide = len(df[df['numeric_label'] == 0])
        attacks = len(df[df['numeric_label'] > 0])
        total_bonafide += bonafide
        total_attacks += attacks
        print(f"{split_name}: {bonafide} bonafide, {attacks} attacks")
    
    print(f"Total: {total_bonafide} bonafide, {total_attacks} attacks")
    
    # save the labels
    save_labels(base_path)
    print("Done")