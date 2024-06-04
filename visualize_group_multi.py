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
    'f': 'Multi-Task',
    'ff': 'Ours',
    'r': 'Reverse',
    'smr': 'Semi-Mid-Reverse',
    'sr': 'Semi-Reverse',
}

MODE2COLOR = {
    'f': 'C0',
    'ff': 'C1',
    'r': 'Reverse',
    'smr': 'Semi-Mid-Reverse',
    'sr': 'Semi-Reverse',
}
def visualize_operation(operation_path, operation_name, metric, split):
    # get all the files in the group path
    # st()
    files = [os.path.join(operation_path, file) for file in os.listdir(operation_path)] + [os.path.join(operation_path.replace('m17','m18'), file) for file in os.listdir(operation_path.replace('m17','m18'))]
    files = [f for f in files if f.endswith('.csv')]
    plt.figure(dpi=250, figsize=(4, 3))
    for idx, f in enumerate(files):
        print(operation_path, f)
        name = f.split("/")[-1][:-4]
        # st()
        # get the mode from the file name
        if split:
            splits = name.split('_')
            st()
            mode = splits[0]
            mode = splits[1]
            color = MODE2COLOR[mode]
            # st()
            # read the file
            df = pd.read_csv(f)
            # plot the data
            plt.plot(df['steps'], df[metric], label=MODE2NAME[mode], c= color)            
        else:
            mode = name.split('_')[1]
            color = MODE2COLOR[mode]
            # st()
            # read the file
            df = pd.read_csv( f)
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
    index_val = 0
    rows,cols = (3,3)
    fig,ax=plt.subplots(rows,cols,figsize=(rows*4,cols*3), dpi=250)
    for i in range(rows):
        for j in range(cols):
            if index_val >= len(operations):
                break
            else:
                o = operations[index_val]
                operation_name = o.split('_')[1]
                if operation_name == '':
                    operation_name = "/"
                # st()
                operation_path = os.path.join(group_path, o)
                files = [os.path.join(operation_path, file) for file in os.listdir(operation_path)] + [os.path.join(operation_path.replace('m17','m18'), file) for file in os.listdir(operation_path.replace('m17','m18'))]
                files = [f for f in files if f.endswith('.csv')]
                # st()
                for idx, f in enumerate(files[:2]):
                    print(operation_path, f)
                    name = f.split("/")[-1][:-4]
                    mode = name.split('_')[1]
                    color = MODE2COLOR[mode]
                    df = pd.read_csv( f)
                    # st()
                    ax[i,j].plot(df['steps'], df[metric], label=MODE2NAME[mode], c= color)
                ax[i,j].set_xscale('log')  # Set x-axis to logarithmic scale
                # ax[i,j].set_xlabel('Steps (Log-scale)')
                # ax[i,j].set_ylabel(metric)
                ax[i,j].set_title(f'Operation: {operation_name}')
                ax[i,j].legend()
                ax[i,j].grid()
                index_val+=1
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.savefig('multitask.pdf', bbox_inches='tight')
    st()
    
    # for o in operations:
    #     # get name of operation stored like op_{operation_name}
    #     operation_name = o.split('_')[1]
    #     # get the path of the operation
    #     operation_path = os.path.join(group_path, o)
    #     # visualize the operation
    #     visualize_operation(operation_path, operation_name, metric, split)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize the data in the group folder')
    parser.add_argument('--group_name', '-g', type=str, help='Path to the group folder')
    parser.add_argument('--metric', '-m', choices=['val_accuracy', 'val_loss', 'val_perplexity'], default='val_accuracy', help='Metric to visualize')
    parser.add_argument('--split','-s', action='store_true', help='Split the data')
    args = parser.parse_args()
    plt.rcParams.update({'font.size': 12})
    # st()
    plt.rcParams["font.family"] = "Times New Roman"
    # matplotlib.rcParams['font.family'] = "sans-serif"
    # matplotlib.rcParams['font.sans-serif'] = "Comic Sans MS"    
    visualize_group(args.group_name, args.split)