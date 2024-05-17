#!/usr/bin/env python
import hydra
from omegaconf import DictConfig
import ipdb
import wandb
st = ipdb.set_trace
import grok
import os
# st()
# parser = grok.training.add_args()
# parser.set_defaults(logdir=os.environ.get("GROK_LOGDIR", "."))
# hparams = parser.parse_args()
# hparams.datadir = os.path.abspath(hparams.datadir)
# hparams.logdir = os.path.abspath(hparams.logdir)


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(args: DictConfig):
    # st()
    # wandb.login(key='899662853ead8246d39f962194401e222ad8517a', relogin=False)
    wandb.login(key='ad54c625d116da251040a8a315cb1b1933b181cf', relogin=False)
    print(args)
    args.logdir = os.path.abspath(args.logdir)
    args.datadir = os.path.abspath(args.datadir)
    grok.training.train(args)


if __name__ == "__main__":
    main()
