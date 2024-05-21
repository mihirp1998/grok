#!/usr/bin/env python
import hydra
from omegaconf import DictConfig
import wandb
import grok
import os
import multiprocessing
wandb.login(key='5508720f47b02cabd61bb6acd61dc553d313b062', relogin=False)

@hydra.main(config_path="config", config_name="config")
def main(args: DictConfig):
    print(args)
    args.logdir = os.path.abspath(args.logdir)
    args.datadir = os.path.abspath(args.datadir)
    grok.training.train(args)

def run_experiment(val):

    # Override the learning rate in the config
    with hydra.initialize(config_path="config"):
        cfg = hydra.compose(config_name="config")
        cfg.train_data_pct = val * 100
        main(cfg)

if __name__ == "__main__":
    lst = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]

    processes = []
    for val in lst:
        p = multiprocessing.Process(target=run_experiment, args=(val,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    print("All processes are done")
