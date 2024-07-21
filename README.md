# Synthetic Functions
This repo contains implementations of useful synthetic functions that is not currently in BoTorch.

## How to use
First, within your environment, install the package.
```bash
pip install git+https://github.com/sangttruong/synthetic_functions.git
```

In your script, include the package:
```bash
from synthetic_functions.alpine import AlpineN1
from synthetic_functions.syngp import SynGP
```

Then you can initialize these functions as below.
```bash
f_ = AlpineN1(dim=2, noise_std=0)
f_ = SynGP(dim=2, noise_std=0)
```
