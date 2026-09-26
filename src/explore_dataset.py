import pandas as pd
import os

def explore_dataset():
    os.makedirs('experiments', exist_ok=True)
    report_path = 'experiments/phase1_dataset_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        def log(msg):
            print(msg)
            f.write(str(msg) + '\n')
            
        log("=== PHASE 1: DATASET EXPLORATION ===\n")
        
        data_dir = 'data'
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'test')
        
        train_sources = ['train_source1.tsv', 'train_source2.tsv', 'train_source3.tsv']
        test_sources = ['test_source1.tsv', 'test_source2.tsv', 'test_source3.tsv']
        
        def process_source_file(filepath, is_train=True):
            log(f"--- Analyzing {os.path.basename(filepath)} ---")
            df = pd.read_csv(filepath, sep='\t', dtype=str)
            
            log(f"File name: {os.path.basename(filepath)}")
            log(f"Number of rows: {len(df)}")
            log(f"Column names: {list(df.columns)}")
            log(f"Data types:\n{df.dtypes}")
            log(f"Number of missing values per column:\n{df.isna().sum()}")
            if 'entity_id' in df.columns:
                log(f"Number of unique entity_id values: {df['entity_id'].nunique()}")
            if 'country' in df.columns:
                log(f"Country value counts:\n{df['country'].value_counts()}")
            log(f"Sample of 5 records:\n{df.head(5)}\n")
            
            if is_train:
                if 'name' in df.columns:
                    log(f"Number of unique business names: {df['name'].nunique()}")
                    log(f"Top 10 most frequent business names:\n{df['name'].value_counts().head(10)}")
                if 'address' in df.columns:
                    log(f"Number of unique business addresses: {df['address'].nunique()}")
                if 'country' in df.columns:
                    log(f"Number of countries: {df['country'].nunique()}")
                    log(f"Top 10 most frequent countries:\n{df['country'].value_counts().head(10)}")
            
            log("-" * 40 + "\n")
        
        for src in train_sources:
            path = os.path.join(train_dir, src)
            if os.path.exists(path):
                process_source_file(path, is_train=True)
            else:
                log(f"File not found: {path}")
                
        for src in test_sources:
            path = os.path.join(test_dir, src)
            if os.path.exists(path):
                process_source_file(path, is_train=False)
            else:
                log(f"File not found: {path}")
                
        # Process ground truth
        gt_path = os.path.join(train_dir, 'train_ground_truth.tsv')
        if os.path.exists(gt_path):
            log("--- Analyzing train_ground_truth.tsv ---")
            gt_df = pd.read_csv(gt_path, sep='\t', dtype=str)
            log(f"Number of rows: {len(gt_df)}")
            log(f"Column names: {list(gt_df.columns)}")
            log(f"Number of missing values per column:\n{gt_df.isna().sum()}")
            
            id_col = 'entity_id' if 'entity_id' in gt_df.columns else gt_df.columns[0]
            match_col = 'matched_entity_ids' if 'matched_entity_ids' in gt_df.columns else gt_df.columns[1] if len(gt_df.columns) > 1 else None
            
            log(f"Sample of 10 ground-truth rows:\n{gt_df.head(10)}\n")
            
            if id_col and match_col:
                log(f"Number of Source 1 entities represented: {gt_df[id_col].nunique()}")
                
                empty_mask = gt_df[match_col].isna() | (gt_df[match_col] == '') | (gt_df[match_col] == '[]')
                log(f"Rows with empty matched_entity_ids: {empty_mask.sum()}")
                
                def count_matches(val):
                    if pd.isna(val) or str(val).strip() == '' or str(val).strip() == '[]':
                        return 0
                    val_str = str(val).strip()
                    if val_str.startswith('[') and val_str.endswith(']'):
                        import ast
                        try:
                            l = ast.literal_eval(val_str)
                            return len(l)
                        except:
                            pass
                    if ',' in val_str:
                        return len(val_str.split(','))
                    if ' ' in val_str:
                        return len(val_str.split(' '))
                    return 1
                    
                match_counts = gt_df[match_col].apply(count_matches)
                
                log("Distribution of number of matched IDs per Source 1 entity:")
                counts_dist = match_counts.value_counts()
                log(f"0 matches: {counts_dist.get(0, 0)}")
                log(f"1 match: {counts_dist.get(1, 0)}")
                log(f"2 matches: {counts_dist.get(2, 0)}")
                log(f"3 matches: {counts_dist.get(3, 0)}")
                log(f"4+ matches: {match_counts[match_counts >= 4].count()}")
                
                total_entities = len(gt_df)
                log(f"Total number of Source 1 training entities: {total_entities}")
                if total_entities > 0:
                    log(f"Percentage of Source 1 entities with zero matches: {counts_dist.get(0, 0) / total_entities * 100:.2f}%")
                    log(f"Percentage with one match: {counts_dist.get(1, 0) / total_entities * 100:.2f}%")
                    log(f"Percentage with multiple matches (2+): {match_counts[match_counts >= 2].count() / total_entities * 100:.2f}%")
            
        else:
            log(f"File not found: {gt_path}")
            
    print(f"\nReport generated at: {report_path}")

if __name__ == '__main__':
    explore_dataset()
