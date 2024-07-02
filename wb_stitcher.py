import pickle
from matplotlib import pyplot as plt
import wandb
import pandas as pd
import re
import os
import nan_cleaner

# Authenticate with W&B
wandb.login()

# Function to extract the numeric suffix from run names
def extract_number(run_name):
    matches = re.findall(r'\d+', run_name)
    return int(matches[0]) if matches else None

# Function to get runs data from W&B project
def get_runs_data(entity, project, filters={"config.training_type": "single_trajectory"}):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters=filters)

    # Sort runs by the number contained in their name
    runs = sorted(runs, key=lambda run: extract_number(run.name))

    # Dictionary to hold the stitched data
    stitched_data = {}
    skipped = []
    # Process each run
    for i, run in enumerate(runs):
        config = run.config
        run_name = run.name
        start_epoch = config.get('start_epoch', 0)
        
        # Extract the base name and run number
        base_name = re.sub(r'\d+$', '', run_name)
        run_number = extract_number(run_name)

        if start_epoch == 0 or run_name == 'absurd-sun-148':
            # Try to get the run data
            try:
                run_df = run.scan_history()
                run_df = pd.DataFrame(run_df)
                # Filter the DataFrame to include only the specified keys
                run_df = run_df[['train_l1', 'val_l1', 'train_loss', 'val_loss']]
                stitched_data[run_name] = (run_number, run_df)
            except Exception as e:
                print(f"Error processing run {run.name}: {e}")

            # Iterate through the following runs for continuations
            for j in range(i + 1, len(runs)):
                next_run = runs[j]
                next_run_name = next_run.name
                next_start_epoch = next_run.config.get('start_epoch', 0)
                next_run_number = extract_number(next_run_name)

                if next_start_epoch == 0:
                    break

                if next_run_number == run_number + 1 or (next_run.config.get('continue_training') is not None and run_name in next_run.config.get('continue_training')) or run_name in next_run.config.get('ckpt_dir'):
                    # Try to get the continuation run data
                    try:
                        next_run_df = next_run.scan_history()
                        next_run_df = pd.DataFrame(next_run_df)   
                        # Filter the DataFrame to include only the specified keys
                        next_run_df = next_run_df[['train_l1', 'val_l1', 'train_loss', 'val_loss']]
                        stitched_data[run_name] = (stitched_data[run_name][0], pd.concat([stitched_data[run_name][1], next_run_df], axis=0))
                        run_number += 1
                    except Exception as e:
                        print(f"Error processing run {next_run.name}: {e}")
                else:
                    print("Skipped " + next_run_name)
                    skipped.append(run_name + "->" +  next_run_name)
                    break     
    return stitched_data, skipped


# Function to run the data fetching and processing
def fetch(filters, recalc = False):
    entity = 'paul_hallmann'  # Replace with your W&B entity
    project = 'masterthesis'  # Replace with your W&B project name

    # Download and stitch data
    filters = dict(sorted(filters.items()))
    foldername = os.path.join('run_data', ''.join([k.split('config.')[1]+"="+ str(v)+";"  for k,v in filters.items()]))
    foldername_2 = os.path.join('cleaned_run_data', ''.join([k.split('config.')[1]+"="+ str(v)+";"  for k,v in filters.items()]))
    if not os.path.exists(foldername) or recalc:
        os.makedirs(foldername, exist_ok=True)
        stitched_data, skipped = get_runs_data(entity, project, filters=filters)
        if len(skipped)>0:
            filename = os.path.join(foldername, "skipped.txt")
            with open(filename, 'w') as f:
                for runname in skipped:
                    f.write(f"{runname}\n")

        for k,v in stitched_data.items():
            filename = os.path.join(foldername, k+".pickle")
            v[1].to_pickle(filename)
    # clean data        
    if not os.path.exists(foldername_2) or recalc:
        os.makedirs(foldername_2, exist_ok=True)
        loaded = {}
        for filename in os.listdir(foldername):
            if "pickle" in filename:
                loaded[filename.split('.')[0]] = (extract_number(filename.lstrip('\\/')), pd.read_pickle(os.path.join(foldername, filename)))

        for filename, (id,run_table) in loaded.items():
            cleaned_table = nan_cleaner.logger_nan_cleaner(run_table)
            filepath = os.path.join(foldername_2, filename)
            cleaned_table.to_pickle(filepath)

    loaded = {}
    for filename in os.listdir(foldername_2):
        loaded[filename.split('.')[0]] = (extract_number(filename.lstrip('\\/')), pd.read_pickle(os.path.join(foldername_2, filename)))
    return loaded

def plot_train_val_comp_ema(run1,label1, run2,label2, param_name, span=10):
    # Ensure the input dataframes have the necessary keys
    required_keys = ["train_l1", "val_l1", "train_loss", "val_loss"]
    for key in required_keys:
        if key not in run1.columns or key not in run2.columns:
            raise ValueError(f"Both dataframes must contain the '{key}' column.")
    
    # Calculate the EMA for each column
    run1_ema = run1.ewm(span=span, adjust=False).mean()
    run2_ema = run2.ewm(span=span, adjust=False).mean()
    
    # Create the plots folder if it doesn't exist
    os.makedirs('plots', exist_ok=True)

    # Plot "train_l1" and "val_l1" comparison
    plt.figure(figsize=(7, 6))
    plt.plot(run1_ema['train_l1'], label=label1 + ' Train L1 EMA', color='blue')
    plt.plot(run1_ema['val_l1'], label=label1 + ' Val L1 EMA', color='blue', linestyle='--')
    plt.plot(run2_ema['train_l1'], label=label2 + ' Train L1 EMA', color='green')
    plt.plot(run2_ema['val_l1'], label=label2 + ' Val L1 EMA', color='green', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('L1 Loss')
    plt.title('Train and Validation L1 Loss Comparison (EMA)')
    plt.legend()
    l1_filename = f"plots/{param_name}_l1_comparison.svg"
    plt.savefig(l1_filename, format='svg')
    plt.close()  # Close the plot to prevent overlapping
    
    # Plot "train_loss" and "val_loss" comparison
    plt.figure(figsize=(7, 6))
    plt.plot(run1_ema['train_loss'], label='Run 1 Train Loss EMA', color='blue')
    plt.plot(run1_ema['val_loss'], label='Run 1 Val Loss EMA', color='blue', linestyle='--')
    plt.plot(run2_ema['train_loss'], label='Run 2 Train Loss EMA', color='green')
    plt.plot(run2_ema['val_loss'], label='Run 2 Val Loss EMA', color='green', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss Comparison (EMA)')
    plt.legend()
    loss_filename = f"plots/{param_name}_loss_comparison.svg"
    plt.savefig(loss_filename, format='svg')
    plt.close()  # Close the plot to prevent overlapping

    print(f"Plots saved as {l1_filename} and {loss_filename}")

    

if __name__ == "__main__":
    
    # relevant single trajectory runs
    # filters = {"config.training_type": "single_trajectory", "config.use_qvel": False}
    
    #TODO plot comparison between smooth_data_generated act_old and single_trajectory

    # # relevant align_push_mono runs
    filters = {"config.training_type": "multi_trajectory", "config.task_name": "align_push_mono"}
    loaded = fetch(filters,False)
    #plot comparison between act_old and act_new
    plot_train_val_comp_ema(loaded['warm-lake-120'][1], "New Transformer", loaded['absurd-sun-148'][1], "Old Transformer", "act_old_vs_new_")
    #TODO plot comparison between act_old and act_new