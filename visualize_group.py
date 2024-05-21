# python file which takes as input a group path (folder) which can contain files like val_{mode}.csv.
# use matplotlib to visualize the data in the files and save the plot in that folder.
import ipdb
st = ipdb.set_trace
import os
import sys
import matplotlib.font_manager


import pandas as pd
import matplotlib as mpl

import matplotlib.pyplot as plt
import matplotlib.font_manager
import numpy as np
import argparse

MODE2NAME = {
    'f': 'Forward',
    'ff': 'Forward-Forward',
    'r': 'Reverse',
    'smr': 'Semi-Mid-Reverse',
    'sr': 'Semi-Reverse',
}

MODE2COLOR = {
    'f': 'C1',
    'ff': 'C2',
    'r': 'Reverse',
    'smr': 'Semi-Mid-Reverse',
    'sr': 'Semi-Reverse',
}
def visualize_operation(operation_path, operation_name, metric, split):
    # get all the files in the group path
    files = os.listdir(operation_path)
    files = [f for f in files if f.endswith('.csv')]
    # plot the data in the files
    plt.figure(dpi=250, figsize=(4, 3))
    for idx, f in enumerate(files):
        print(operation_path, f)
        name = f[:-4]
        # get the mode from the file name
        if split:
            splits = name.split('_')
            st()
            mode = splits[0]
            mode = splits[1]
            color = MODE2COLOR[mode]
            # st()
            # read the file
            df = pd.read_csv(os.path.join(operation_path, f))
            # plot the data
            plt.plot(df['steps'], df[metric], label=MODE2NAME[mode], c= color)            
        else:
            mode = f.split('_')[1].split('.')[0]
            color = MODE2COLOR[mode]
            # st()
            # read the file
            df = pd.read_csv(os.path.join(operation_path, f))
            # plot the data
            plt.plot(df['steps'], df[metric], label=MODE2NAME[mode], c= color)

    plt.xscale('log')  # Set x-axis to logarithmic scale
    plt.xlabel('Steps (Log-scale)')
    plt.ylabel(metric)
    plt.title(f'Operator: {operation_name} for {metric}')
    plt.legend()
    plt.grid()
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(operation_path, 'comparision.png'))
    print(f'Saved the plot in {operation_path}')
    # plt.show()




def visualize_group(group_name, split, metric='val_accuracy'):
    # get all the operations in the group path
    # st()
    group_path = f"exps/logs/{group_name}"
    
    operations = os.listdir(group_path)
    operations = [o for o in operations if os.path.isdir(os.path.join(group_path, o))]
    

    for o in operations:
        # get name of operation stored like op_{operation_name}
        operation_name = o.split('_')[1]
        # get the path of the operation
        operation_path = os.path.join(group_path, o)
        # visualize the operation
        visualize_operation(operation_path, operation_name, metric, split)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize the data in the group folder')
    parser.add_argument('--group_name', '-g', type=str, help='Path to the group folder')
    parser.add_argument('--metric', '-m', choices=['val_accuracy', 'val_loss', 'val_perplexity'], default='val_accuracy', help='Metric to visualize')
    parser.add_argument('--split','-s', action='store_true', help='Split the data')
    args = parser.parse_args()
    plt.rcParams.update({'font.size': 12})
    # st()
    # plt.rcParams["font.family"] = "Times New Roman"
    # matplotlib.rcParams['font.family'] = "sans-serif"
    # matplotlib.rcParams['font.sans-serif'] = "Comic Sans MS"    
    visualize_group(args.group_name, args.split)