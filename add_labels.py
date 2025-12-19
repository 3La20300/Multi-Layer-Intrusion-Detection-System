"""Add labels to aggregated_features.csv"""
import pandas as pd

# Load the aggregated features
df = pd.read_csv('data/processed/aggregated_features.csv')

# Apply labeling rules
syn_threshold = 5
ports_threshold = 3
syn_ack_ratio_threshold = 3.0

# Rule 1: High SYN count AND many unique ports (port scanning pattern)
condition1 = (df['syn_count'] > syn_threshold) & (df['unique_dst_ports'] > ports_threshold)

# Rule 2: Very high SYN/ACK ratio (incomplete connections)
condition2 = df['syn_ack_ratio'] > syn_ack_ratio_threshold

# Combine: Either pattern indicates attack
df['label'] = ((condition1) | (condition2)).astype(int)

# Save the labeled CSV
df.to_csv('data/processed/aggregated_features_labeled.csv', index=False)

# Print summary
print('Labels added successfully!')
print(f'\nLabel distribution:')
normal = (df['label'] == 0).sum()
attack = (df['label'] == 1).sum()
print(f'  Normal (0): {normal} ({100*normal/len(df):.1f}%)')
print(f'  Attack (1): {attack} ({100*attack/len(df):.1f}%)')
print(f'\nTotal rows: {len(df)}')
