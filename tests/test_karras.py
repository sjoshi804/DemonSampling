# tests/test_karras.py

from api import odeint, sdeint, get_init_latent
from utils import get_condition, from_latent_to_pil
import torch
import os
import numpy as np
condition = {
    "prompts": ["An astronaut riding a horse on Mars."],
    "cfgs": [2]
}
if not os.path.exists('./tmp'):
    os.mkdir('./tmp')



def test_ode():
    x = get_init_latent()
    sample_step = 20
    x = odeint(x, condition, sample_step)
    pil = from_latent_to_pil(x)
    pil.save('./tmp/odeint.png')
    # test if the file exists
    assert os.path.exists('./tmp/odeint.png')

def test_sde():
    x = get_init_latent()
    sample_step = 20
    x = sdeint(x, condition, 0.01, sample_step)
    pil = from_latent_to_pil(x)
    pil.save('./tmp/sdeint.png')
    # test if the file exists
    assert os.path.exists('./tmp/sdeint.png')

# run the tests
# time pytest tests/test_karras.py

