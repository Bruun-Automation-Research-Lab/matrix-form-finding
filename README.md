# Matrix Form Finding

## Environment Setup

```zsh
conda env create -f environment.yml
conda activate CEE6501-student
```

```zsh
conda env remove --name CEE6501-student
```

```zsh
conda env create -f environment_jupyter.yml
conda activate CEE6501-student
```

```zsh
pre-commit install
```

To profile the full script and save the profiling output to the `output` folder:

```bash
python -m cProfile -o ./output/profile.out ./FormFinder/main.py
```

To read the saved profile and print the total recorded execution time:

```bash
python -c "import pstats; p = pstats.Stats('./output/profile.out'); print(f'Total time: {p.total_tt:.6f} s')"
```
