import pandas as pd
from pathlib import Path
from omegaconf import DictConfig

def load_data(config: DictConfig) -> pd.DataFrame:
    data_frames = []
    
    for file_name in config.data.files:
        file_path = Path(config.paths.raw_data) / file_name
        print(f"Loading {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            data_frames.append(df)
            print(f"  Loaded {len(df)} rows")
        except Exception as e:
            print(f"  Error loading {file_name}: {e}")
    
    combined_df = pd.concat(data_frames, axis=0, ignore_index=True)
    print(f"\nTotal rows loaded: {len(combined_df)}")
    
    return combined_df