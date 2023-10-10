"""Generates the results over the four images for two clicks taken from SAM IIS for our method and the SAM IIS method."""

from pathlib import Path

from dataloaders.ndd20 import load_sample
from baselines.sam_iis import run_experiment as run_sam_iis_experiment
from IISS.run_experiment import run_experiment as run_iiss_experiment
from config import device, dev

def main():
    datapath = Path('main_idea/images') 
    def load_sample_fn(index):
        return load_sample(datapath, index)

    # run sam iis first
    run_sam_iis_experiment(load_sample_fn, 4, dev, 2, 'main_idea_sam_iis', device)

    # run IISS second
    run_iiss_experiment(load_sample_fn, 4, max_total_clicks=8, runname='main_idea_iiss')



if __name__ == '__main__':
    main()