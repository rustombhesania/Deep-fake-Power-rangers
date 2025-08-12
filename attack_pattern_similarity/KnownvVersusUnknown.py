
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
import pandas as pd
from tqdm import tqdm

def load_attack_samples(attack_id, num_samples=15):
    """Load samples for specific attack"""
    
    if attack_id in ['A01', 'A02', 'A03', 'A04', 'A05', 'A06']:
        # classical attacks from dev set
        metadata_path = "./dev_data_corrected_with_a04/metadata.csv"
        audio_dir = "C:/ASVSpoof19/LA/ASVspoof2019_LA_dev/flac"
    else:
        # neural attacks from eval set
        metadata_path = "./eval_data_corrected/metadata.csv"
        audio_dir = "C:/ASVSpoof19/LA/ASVspoof2019_LA_eval/flac"
    
    if not os.path.exists(metadata_path):
        print(f"No metadata for {attack_id}")
        return []
    
    metadata = pd.read_csv(metadata_path)
    attack_samples = metadata[metadata['class_name'] == attack_id].head(num_samples)
    
    samples = []
    for _, row in attack_samples.iterrows():
        audio_path = os.path.join(audio_dir, row['audio_file'])
        if os.path.exists(audio_path):
            try:
                y, sr = librosa.load(audio_path, sr=16000)
                if np.max(np.abs(y)) > 0:
                    y = y / np.max(np.abs(y))
                samples.append(y)
                
                if len(samples) >= num_samples:
                    break
            except:
                continue
    
    return samples


def extract_mfcc(audio_samples):
    """Extract MFCC features from audio samples"""
    
    mfcc_features = []
    
    for audio in audio_samples:
        try:
            mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)
            # average across time
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_features.append(mfcc_mean)
        except:
            continue
    
    return np.array(mfcc_features)


def create_similarity_matrix():
    """Create similarity matrix for available attacks"""
    
    print("Creating attack similarity matrix...")
    
    # attacks to analyze
    attacks = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06',  # classical
               'A07', 'A08', 'A09', 'A10', 'A11', 'A12',  # neural vocoder/tts
               'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19']  # neural e2e/hybrid
    
    # load samples and extract features
    attack_features = {}
    
    for attack in attacks:
        print(f"Loading {attack}...")
        samples = load_attack_samples(attack)
        
        if len(samples) > 0:
            features = extract_mfcc(samples)
            if len(features) > 0:
                # average feature vector for this attack
                attack_features[attack] = np.mean(features, axis=0)
                print(f"  Got {len(samples)} samples")
            else:
                print(f"  No features extracted")
        else:
            print(f"  No samples found")
    
    print(f"\nLoaded features for {len(attack_features)} attacks")
    
    # create similarity matrix
    attack_list = sorted(attack_features.keys())
    n_attacks = len(attack_list)
    similarity_matrix = np.zeros((n_attacks, n_attacks))
    
    print("Computing similarities...")
    
    for i, attack1 in enumerate(attack_list):
        for j, attack2 in enumerate(attack_list):
            if i == j:
                similarity_matrix[i, j] = 1.0  # self similarity
            else:
                sim = 1 - cosine(attack_features[attack1], attack_features[attack2])
                similarity_matrix[i, j] = sim
    
    return similarity_matrix, attack_list


def plot_similarity_matrix(similarity_matrix, attack_list):
    """Plot the similarity matrix"""
    
    plt.figure(figsize=(12, 10))
    
    # create heatmap
    sns.heatmap(similarity_matrix, 
                xticklabels=attack_list, 
                yticklabels=attack_list,
                annot=True, 
                fmt='.3f',
                cmap='viridis',
                cbar_kws={'label': 'Similarity'})
    
    plt.title('Attack Similarity Matrix (MFCC-based)')
    plt.xlabel('Attack ID')
    plt.ylabel('Attack ID')
    plt.tight_layout()
    plt.savefig('attack_similarity_matrix.png', dpi=150)
    plt.show()
    
    print("Similarity matrix plot saved as 'attack_similarity_matrix.png'")


def analyze_similarities(similarity_matrix, attack_list):
    """Analyze the similarity results"""
    
    print("\nSimilarity Analysis:")
    print("=" * 30)
    
    # find highest similarities (excluding diagonal)
    high_similarities = []
    
    for i in range(len(attack_list)):
        for j in range(i+1, len(attack_list)):  # upper triangle only
            sim = similarity_matrix[i, j]
            high_similarities.append((attack_list[i], attack_list[j], sim))
    
    # sort by similarity
    high_similarities.sort(key=lambda x: x[2], reverse=True)
    
    print("Top 10 most similar attack pairs:")
    for i, (attack1, attack2, sim) in enumerate(high_similarities[:10]):
        print(f"{i+1:2d}. {attack1}-{attack2}: {sim:.3f}")
    
    # check specific pairs of interest
    print(f"\nKey pairs of interest:")
    
    key_pairs = [('A04', 'A12'), ('A04', 'A16'), ('A12', 'A16')]
    
    for attack1, attack2 in key_pairs:
        if attack1 in attack_list and attack2 in attack_list:
            i = attack_list.index(attack1)
            j = attack_list.index(attack2)
            sim = similarity_matrix[i, j]
            print(f"{attack1}-{attack2}: {sim:.3f}")
        else:
            print(f"{attack1}-{attack2}: not available")
    
    # classical vs neural analysis
    classical = [a for a in attack_list if a in ['A01', 'A02', 'A03', 'A04', 'A05', 'A06']]
    neural = [a for a in attack_list if a not in classical]
    
    print(f"\nClassical attacks found: {classical}")
    print(f"Neural attacks found: {neural}")
    
    # average similarities within groups
    if len(classical) >= 2:
        classical_sims = []
        for i, a1 in enumerate(classical):
            for j, a2 in enumerate(classical):
                if i < j:
                    idx1 = attack_list.index(a1)
                    idx2 = attack_list.index(a2)
                    classical_sims.append(similarity_matrix[idx1, idx2])
        
        if classical_sims:
            print(f"Average classical-classical similarity: {np.mean(classical_sims):.3f}")
    
    if len(neural) >= 2:
        neural_sims = []
        for i, a1 in enumerate(neural):
            for j, a2 in enumerate(neural):
                if i < j:
                    idx1 = attack_list.index(a1)
                    idx2 = attack_list.index(a2)
                    neural_sims.append(similarity_matrix[idx1, idx2])
        
        if neural_sims:
            print(f"Average neural-neural similarity: {np.mean(neural_sims):.3f}")
    
    # cross-group similarity
    if classical and neural:
        cross_sims = []
        for a1 in classical:
            for a2 in neural:
                idx1 = attack_list.index(a1)
                idx2 = attack_list.index(a2)
                cross_sims.append(similarity_matrix[idx1, idx2])
        
        if cross_sims:
            print(f"Average classical-neural similarity: {np.mean(cross_sims):.3f}")


if __name__ == "__main__":
    print("Attack Similarity Matrix Analysis")
    print("Analyzing MFCC similarity between all available attacks")
    
    # create similarity matrix
    similarity_matrix, attack_list = create_similarity_matrix()
    
    if len(attack_list) >= 2:
        print(f"\nAnalyzing {len(attack_list)} attacks: {attack_list}")
        
        # plot matrix
        plot_similarity_matrix(similarity_matrix, attack_list)
        
        # analyze results
        analyze_similarities(similarity_matrix, attack_list)
        
        print("\nAnalysis complete!")
    else:
        print("Not enough attacks loaded for analysis")
    
    print("Done!")