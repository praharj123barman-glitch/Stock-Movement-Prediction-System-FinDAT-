# run_pipeline.py
import os
import subprocess
import time

def run_command(command):
    """Run a command and print output"""
    print(f"Running: {command}")
    start_time = time.time()
    subprocess.run(command, shell=True)
    elapsed_time = time.time() - start_time
    print(f"Completed in {elapsed_time:.2f} seconds\n")

def main():
    """Run the complete pipeline from preprocessing to evaluation"""
    print("=== Starting FinGAT with Dynamic Transformer Pipeline ===")
    
    # Step 1: Preprocess data
    print("\n=== Step 1: Data Preprocessing ===")
    run_command("python preprocess.py")
    
    # Step 2: Train model
    print("\n=== Step 2: Model Training ===")
    run_command("python train.py")
    
    # Step 3: Evaluate model
    print("\n=== Step 3: Model Evaluation ===")
    run_command("python evaluate.py")
    
    print("\n=== Pipeline Complete ===")
    print("Check the 'results' directory for prediction outputs")

if __name__ == "__main__":
    main()
