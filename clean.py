import argparse
import shutil
import os
from config import SPLITS, split_dir

def main():
    parser = argparse.ArgumentParser(description="Clean up alphas or weights for a given split.")
    parser.add_argument("--split", choices=list(SPLITS.keys()) + ["all"], required=True, help="Which split to clean (or 'all').")
    parser.add_argument("--target", choices=["alphas", "weights", "both"], required=True, help="What to clean.")
    
    args = parser.parse_args()
    
    splits_to_clean = list(SPLITS.keys()) if args.split == "all" else [args.split]
    
    for split in splits_to_clean:
        base_dir = split_dir(split)
        
        targets = ["alphas", "weights"] if args.target == "both" else [args.target]
        
        for target in targets:
            target_dir = os.path.join(base_dir, target)
            if os.path.exists(target_dir):
                print(f"Removing {target_dir}...")
                shutil.rmtree(target_dir)
            else:
                print(f"Directory {target_dir} does not exist. Skipping.")

if __name__ == "__main__":
    main()
