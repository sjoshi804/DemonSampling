# Standard Library Imports
import os
import json
from datetime import datetime

# Third-party Imports
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import fire
import numpy as np

# Local Application/Library Specific Imports
from reward_models.AestheticScorer import AestheticScorer
from generate_abstract import DemonGenerater

# Instantiate the aesthetic scorer (can be used globally or within the subclass)
aesthetic_scorer = AestheticScorer()

def read_animals(file_path):
    """
    Read a file containing a list of animals.
    """
    with open(file_path, 'r') as f:
        animals = f.read().splitlines()
    return animals


# Define a subclass that implements the abstract rewards method.
class AestheticAnimalEvaluator(DemonGenerater):
    def rewards(self, pils):
        return aesthetic_scorer(pils).cpu().numpy().tolist()

def aesthetic_animal_eval(
    beta=0.1,
    tau='adaptive',
    K=16,             # Number of candidate samples (was action_num)
    T=64,             # Number of time steps (was sample_step)
    demon_type="tanh",# Weighting method (was weighting)
    r_of_c="baseline",
    c_steps=20,
    ode_after=0.11,
    cfg=2,
    seed=42,
    experiment_directory="experiments/aesthetic_animal_eval",
):
    """
    Evaluate the aesthetic score of animals using latent space optimization.
    """
    # Create a unique logging directory.
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(experiment_directory, datetime_str)
    os.makedirs(log_dir, exist_ok=False)

    # Instantiate our evaluator subclass.
    evaluator = AestheticAnimalEvaluator(
        cfg=cfg,
        beta=beta,
        tau=tau,
        K=K,
        T=T,
        demon_type=demon_type,
        r_of_c=r_of_c,
        c_steps=c_steps,
        ode_after=ode_after,
        seed=seed,
        save_pils=False,
        experiment_directory=log_dir,
        ylabel="Aesthetic Score"
    )

    for prompt in tqdm(read_animals('assets/common_animals.txt')):
        evaluator.generate(prompt=prompt)

if __name__ == '__main__':
    fire.Fire(aesthetic_animal_eval)

# python3 pipelines/aesthetic_animal_eval.py  --r_of_c "consistency"