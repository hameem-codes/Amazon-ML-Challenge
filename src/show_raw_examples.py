import pandas as pd
import os

def show_raw_examples():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    
    data_dir = os.path.join('data', 'train')
    sources = {
        'SOURCE 1': 'train_source1.tsv',
        'SOURCE 2': 'train_source2.tsv',
        'SOURCE 3': 'train_source3.tsv'
    }
    
    with open('experiments/raw_examples_output.txt', 'w', encoding='utf-8') as f:
        for source_name, filename in sources.items():
            filepath = os.path.join(data_dir, filename)
            if not os.path.exists(filepath):
                f.write(f"File not found: {filepath}\n")
                continue
                
            f.write(f"\n{'='*80}\n")
            f.write(f"{source_name}\n")
            f.write(f"{'='*80}\n")
            
            df = pd.read_csv(filepath, sep='\t', dtype=str, nrows=20, na_filter=False)
            
            f.write("20 example records:\n\n")
            f.write(df.to_string(index=False) + "\n")
            
            if source_name in ['SOURCE 2', 'SOURCE 3']:
                f.write(f"\n--- 10 examples with missing business_address in {source_name} ---\n")
                
                missing_examples = pd.DataFrame()
                chunksize = 10000
                
                for chunk in pd.read_csv(filepath, sep='\t', dtype=str, chunksize=chunksize):
                    missing_in_chunk = chunk[chunk['business_address'].isna()]
                    missing_examples = pd.concat([missing_examples, missing_in_chunk])
                    if len(missing_examples) >= 10:
                        break
                        
                if not missing_examples.empty:
                    f.write(missing_examples.head(10).to_string(index=False) + "\n")
                else:
                    f.write("No missing business_address found in the scanned chunks.\n")

if __name__ == '__main__':
    show_raw_examples()
