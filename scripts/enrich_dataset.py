# scripts/enrich_dataset.py
import pandas as pd
import numpy as np

def enrich_dataset(df):
    """Preenche: auth_status, attempt_count, rule_applied, ground_truth"""
    
    # 1. AUTH_STATUS (90% success, 10% failed)
    df['auth_status'] = np.random.choice(
        ['success', 'failed'], 
        size=len(df), 
        p=[0.90, 0.10]
    )
    
    # 2. ATTEMPT_COUNT (1-5, mais peso em 1)
    df['attempt_count'] = np.random.choice(
        [1, 2, 3, 4, 5], 
        size=len(df), 
        p=[0.70, 0.15, 0.10, 0.03, 0.02]
    )
    # Se auth falhou, aumentar attempts
    df.loc[df['auth_status'] == 'failed', 'attempt_count'] = np.random.choice(
        [2, 3, 4, 5], size=(df['auth_status'] == 'failed').sum()
    )
    
    # 3. RULE_APPLIED (baseado em contexto)
    def assign_rule(row):
        if row['emergency_flag']:
            return 'emergency_override'
        if row['human_present']:
            return 'human_supervision'
        if row['agent_type'] == 'robot':
            return f"robot_{row['action']}_policy"
        return 'default_policy'
    
    df['rule_applied'] = df.apply(assign_rule, axis=1)
    
    # 4. GROUND_TRUTH
    def label_gt(row):
    """Lógica mais realista baseada no fluxograma"""
    
    # Deny imediato
    if row['attempt_count'] >= 5:
        return 'deny'
    if row['emergency_flag'] and not row['human_present'] and row['action'] == 'execute':
        return 'deny'
    
    # MFA requerido
    if row['attempt_count'] in [3, 4]:
        return 'mfa'
    if row['auth_status'] == 'failed' and row['attempt_count'] <= 2:
        return 'mfa'
    
    # Review manual
    if row['emergency_flag'] and row['action'] == 'write':
        return 'review'
    
    # Allow
    return 'allow'
    
    df['ground_truth'] = df.apply(label_gt, axis=1)
    df['stat_score'] = np.random.uniform(0.1, 0.9, len(df))
    df['ml_score'] = np.random.uniform(0.1, 0.9, len(df))
    df['policy_score'] = np.random.uniform(0.1, 0.9, len(df))
    # Ajustar scores baseado em ground_truth para consistência
    df.loc[df['ground_truth'] == 'deny', 'stat_score'] *= 0.3  # Scores baixos
    df.loc[df['ground_truth'] == 'allow', 'stat_score'] *= 1.5  # Scores altos
    # Repetir para ml_score e policy_score
    
    return df

# Processar splits
for split in ['train', 'validation', 'test']:
    df = pd.read_csv(f'data/100k/{split}.csv')
    df = enrich_dataset(df)
    df.to_csv(f'data/100k/{split}.csv', index=False)
    
    # Mostrar distribuição
    print(f"\n{split.upper()}:")
    print(f"  Ground truth: {df['ground_truth'].value_counts().to_dict()}")
    print(f"  Auth status: {df['auth_status'].value_counts().to_dict()}")
