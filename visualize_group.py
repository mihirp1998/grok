# python file which takes as input a group path (folder) which can contain files like val_{mode}.csv.
# use matplotlib to visualize the data in the files and save the plot in that folder.

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

MODE2NAME = {
    'f': 'Forward',
    'ff': 'Forward-Forward',
    'r': 'Reverse',
    'smr': 'Semi-Mid-Reverse',
    'sr': 'Semi-Reverse',
}

def visualize_operation(operation_path, operation_name, metric):
    # get all the files in the group path
    files = os.listdir(operation_path)
    files = [f for f in files if f.endswith('.csv')]

    # plot the data in the files
    plt.figure()

    for f in files:
        print(operation_path, f)
        # get the mode from the file name
        mode = f.split('_')[1].split('.')[0]
        # read the file
        df = pd.read_csv(os.path.join(operation_path, f))
        # plot the data
        plt.plot(df['steps'], df[metric], label=MODE2NAME[mode])

    plt.xscale('log')  # Set x-axis to logarithmic scale
    plt.xlabel('Steps (Log-scale)')
    plt.ylabel(metric)
    plt.title(f'Operator: {operation_name} for {metric}')
    plt.legend()
    plt.savefig(os.path.join(operation_path, 'comparision.png'))
    print(f'Saved the plot in {operation_path}')
    # plt.show()

def visualize_group(group_path, metric='val_accuracy'):
    # get all the operations in the group path
    operations = os.listdir(group_path)
    operations = [o for o in operations if os.path.isdir(os.path.join(group_path, o))]

    for o in operations:
        # get name of operation stored like op_{operation_name}
        operation_name = o.split('_')[1]
        # get the path of the operation
        operation_path = os.path.join(group_path, o)
        # visualize the operation
        visualize_operation(operation_path, operation_name, metric)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize the data in the group folder')
    parser.add_argument('--group_path', '-g', type=str, help='Path to the group folder')
    parser.add_argument('--metric', '-m', choices=['val_accuracy', 'val_loss', 'val_perplexity'], default='val_accuracy', help='Metric to visualize')
    args = parser.parse_args()
    visualize_group(args.group_path)