import os
from datetime import datetime

import fire
from transformers import AutoModel, AutoProcessor
from generate_abstract import DemonGenerater
from llm import ask_gemini, ask_gpt

# Read scenario and prompt files.
with open('assets/scenarios.txt', 'r') as f:
    scenarios = f.readlines()

with open('assets/scenarios_prompt.txt', 'r') as f:
    scenarios_prompt = f.readlines()


def modified_partition(arr, cmp_fn):
    def quicksort_partition(arr, idx):
        """Perform a quicksort partitioning based on the first element."""
        pivot = idx[0]
        left = []
        right = []
        for i in idx[1:]:
            print(f"Comparing index {i} with pivot {pivot}")
            if cmp_fn(arr[i], arr[pivot]):
                left.append(i)
            else:
                right.append(i)
        return left, [pivot], right

    # Apply the modified quicksort partitioning:
    # 1. Partition the array on the first element.
    # 2. Repartition the larger of the two resulting partitions.
    idx = list(range(len(arr)))
    left, pivot, right = quicksort_partition(arr, idx)
    if len(left) < len(right):
        # Repartition right.
        sub_left, sub_pivot, sub_right = quicksort_partition(arr, right)
        sorted_idx = left + pivot + sub_left + sub_pivot + sub_right
    else:
        # Repartition left.
        sub_left, sub_pivot, sub_right = quicksort_partition(arr, left)
        sorted_idx = sub_left + sub_pivot + sub_right + pivot + right

    return [1 if i in sorted_idx[-8:] else 0 for i in idx]


class VLLMGenerater(DemonGenerater):
    def __init__(self, senario, prompt, model='gemini', *args, **kwargs):
        """
        Initialize with the scenario and prompt along with the sampling parameters.
        """
        super().__init__(*args, **kwargs)
        self.senario = senario
        self.prompt = prompt
        self.choices = []
        if model == 'gemini':
            self.ask = ask_gemini
        elif model == 'gpt':
            self.ask = ask_gpt
        else:
            raise ValueError("Unknown model.")

    def rewards(self, pils):
        """
        Compute rewards for each PIL image by partitioning the list.
        """
        def cmp(x, y):
            # Compare two images using the provided LLM.
            return self.ask(self.senario, self.prompt, [x, y])[1]
        selects = modified_partition(pils, cmp)
        print(f"Selects: {selects}")
        self.choices.append(selects)
        return selects


def vllm_generate(
    beta=0.2,
    tau=0.0001,
    K=16,
    T=128,
    demon_type="tanh",
    r_of_c="baseline",
    c_steps=20,
    ode_after=8,
    cfg=2,
    seed=66,
    model='gemini',
    experiment_directory="experiments/vllm_as_demon",
):
    """
    Generate images using VLLM-based demon sampling.
    
    The order of parameters is aligned with demon_sampling:
      - beta, tau, K, T, demon_type, r_of_c, c_steps, ode_after, cfg, seed, etc.
    """
    for text, scenario in zip(scenarios_prompt, scenarios):
        # Use a portion of the scenario string as a folder name.
        exp_dir = os.path.join(experiment_directory, model, scenario.split(' ')[3])
        if os.path.exists(exp_dir):
            continue
        os.makedirs(exp_dir, exist_ok=True)
        
        generator = VLLMGenerater(
            senario=scenario,
            prompt=text,
            model=model,
            beta=beta,
            tau=tau,
            K=K,
            T=T,
            demon_type=demon_type,
            r_of_c=r_of_c,
            c_steps=c_steps,
            ode_after=ode_after,
            cfg=cfg,
            seed=seed,
            save_pils=True,
            ylabel="Decision",
            experiment_directory=exp_dir
        )
        generator.generate(prompt=text)
        
        with open(os.path.join(exp_dir, 'choices.txt'), 'a') as f:
            f.write(str(generator.choices))


if __name__ == '__main__':
    fire.Fire(vllm_generate)

# Example usage:
# python pipelines/vllm_generate.py 