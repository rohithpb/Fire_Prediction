"""
End-to-End Orchestrator for NASA FIRMS Fire Prediction Pipeline.
Executes data preparation, feature engineering, model training, and evaluation sequentially.
"""

import sys
import time

from src.data_prep import run_data_preparation
from src.features import run_feature_engineering
from src.train import run_training_pipeline
from src.evaluate import run_evaluation_pipeline


def main():
    start_time = time.time()
    print("=" * 80)
    print("      NASA FIRMS VIIRS ACTIVE FIRE MACHINE LEARNING PIPELINE")
    print("                South India / Kerala Region")
    print("=" * 80)
    
    # Step 1: Data Preparation
    run_data_preparation()
    
    # Step 2: Feature Engineering
    run_feature_engineering()
    
    # Step 3: Model Training
    run_training_pipeline()
    
    # Step 4: Evaluation & Visualization
    run_evaluation_pipeline()
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.1f} SECONDS ({elapsed/60:.2f} MINUTES)!")
    print("All models saved to:  models/")
    print("All figures saved to: outputs/")
    print("=" * 80)


if __name__ == "__main__":
    main()
