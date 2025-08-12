import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import pandas as pd
from tqdm import tqdm

def load_attack_samples(attack_id, num_samples=20):
    """Load audio samples for specific attack"""
    
    print(f"Loading {attack_id} samples...")
    
    # paths for different datasets
    if attack_id == 'A04':
        # A04 is in dev set
        metadata_path = "./dev_data_corrected_with_a04/metadata.csv"
        audio_dir = "C:/ASVSpoof19/LA/ASVspoof2019_LA_dev/flac"
    else:
        # A12, A16 are in eval set
        metadata_path = "./eval_data_corrected/metadata.csv"
        audio_dir = "C:/ASVSpoof19/LA/ASVspoof2019_LA_eval/flac"
    
    if not os.path.exists(metadata_path):
        print(f"Metadata not found: {metadata_path}")
        return []
    
    # load metadata
    metadata = pd.read_csv(metadata_path)
    attack_samples = metadata[metadata['class_name'] == attack_id].head(num_samples)
    
    samples = []
    for _, row in tqdm(attack_samples.iterrows(), desc=f"Loading {attack_id}"):
        audio_path = os.path.join(audio_dir, row['audio_file'])
        if os.path.exists(audio_path):
            try:
                y, sr = librosa.load(audio_path, sr=16000)
                # normalize
                if np.max(np.abs(y)) > 0:
                    y = y / np.max(np.abs(y))
                samples.append(y)
                
                if len(samples) >= num_samples:
                    break
            except:
                continue
    
    print(f"Loaded {len(samples)} {attack_id} samples")
    return samples


def extract_mfcc_features(audio_samples, attack_id):
    """Extract MFCC features from audio samples"""
    
    print(f"Extracting MFCC features for {attack_id}...")
    
    mfcc_features = []
    
    for audio in tqdm(audio_samples, desc=f"MFCC {attack_id}"):
        try:
            # extract 13 MFCC coefficients
            mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)
            # average across time to get one vector per sample
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_features.append(mfcc_mean)
        except:
            continue
    
    print(f"Extracted MFCC for {len(mfcc_features)} {attack_id} samples")
    return np.array(mfcc_features)


def calculate_similarity(features1, features2):
    """Calculate average cosine similarity between two feature sets"""
    
    # get average feature vector for each attack
    avg_feat1 = np.mean(features1, axis=0)
    avg_feat2 = np.mean(features2, axis=0)
    
    # calculate cosine similarity
    similarity = 1 - cosine(avg_feat1, avg_feat2)
    
    return similarity


def check_attack_similarities():
    """Check similarities between A04, A12, and A16"""
    
    print("Checking similarities between A04, A12, and A16")
    print("="*50)
    
    # load samples for each attack
    attacks = ['A04', 'A12', 'A16']
    samples = {}
    features = {}
    
    for attack in attacks:
        samples[attack] = load_attack_samples(attack)
        if len(samples[attack]) > 0:
            features[attack] = extract_mfcc_features(samples[attack], attack)
        else:
            print(f"No samples loaded for {attack}")
            return
    
    print("\nCalculating similarities...")
    
    # calculate pairwise similarities
    similarities = {}
    
    for i, attack1 in enumerate(attacks):
        for j, attack2 in enumerate(attacks):
            if i <= j:  # avoid duplicates
                if attack1 in features and attack2 in features:
                    sim = calculate_similarity(features[attack1], features[attack2])
                    similarities[f"{attack1}-{attack2}"] = sim
    
    # print results
    print("\nSimilarity Results:")
    print("-" * 30)
    
    for pair, similarity in similarities.items():
        print(f"{pair}: {similarity:.3f}")
    
    # focus on the key comparisons
    a04_a12 = similarities.get('A04-A12', 0)
    a04_a16 = similarities.get('A04-A16', 0)
    a12_a16 = similarities.get('A12-A16', 0)
    
    print(f"\nKey Findings:")
    print(f"A04 (Voice Conv) - A12 (MelGAN):     {a04_a12:.3f}")
    print(f"A04 (Voice Conv) - A16 (FastSpeech): {a04_a16:.3f}")
    print(f"A12 (MelGAN) - A16 (FastSpeech):     {a12_a16:.3f}")
    
    print(f"\nResults:")
    if a04_a12 > 0.8:
        print(f"A04-A12: {a04_a12:.3f} (high similarity)")
    else:
        print(f"A04-A12: {a04_a12:.3f} (low similarity)")
    
    if a04_a16 > 0.8:
        print(f"A04-A16: {a04_a16:.3f} (high similarity)")
    else:
        print(f"A04-A16: {a04_a16:.3f} (low similarity)")
    
    if a12_a16 > 0.8:
        print(f"A12-A16: {a12_a16:.3f} (high similarity)")
    else:
        print(f"A12-A16: {a12_a16:.3f} (moderate similarity)")
    
    # create simple plot
    plt.figure(figsize=(8, 6))
    
    pairs = ['A04-A12', 'A04-A16', 'A12-A16']
    values = [a04_a12, a04_a16, a12_a16]
    colors = ['blue', 'red', 'green']
    
    bars = plt.bar(pairs, values, color=colors, alpha=0.7)
    plt.axhline(y=threshold, color='black', linestyle='--', alpha=0.7, label=f'Threshold ({threshold})')
    
    plt.ylabel('MFCC Similarity')
    plt.title('Attack Similarity Comparison')
    plt.ylim(0, 1)
    plt.legend()
    
    # add values on bars
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('attack_similarity.png', dpi=150)
    plt.show()
    
    print(f"\nPlot saved as 'attack_similarity.png'")
    
    return similarities


if __name__ == "__main__":
    print("Attack Similarity Analysis")
    print("Comparing A04 (Voice Conversion), A12 (MelGAN), A16 (FastSpeech2)")
    
    similarities = check_attack_similarities()
    
    if similarities:
        print("\nAnalysis complete!")
        
        # simple conclusion
        a04_a12 = similarities.get('A04-A12', 0)
        a04_a16 = similarities.get('A04-A16', 0)
        avg_cross = (a04_a12 + a04_a16) / 2
        
        print(f"\nAverage A04-neural similarity: {avg_cross:.3f}")
        
        if avg_cross > 0.8:
            print("Neural attacks are similar to voice conversion")
        else:
            print("Neural attacks are different from voice conversion")
    
    print("Done!")