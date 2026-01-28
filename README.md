# TPGDiff

TPGDiff is a project for modeling degradation and performing image restoration via a two-stage training pipeline. This README describes environment setup, dependency installation, data generation, training stages, and testing commands.

## Prerequisites

- Python 3.8
- Conda (recommended) or compatible virtual environment manager
- CUDA-enabled GPU and matching PyTorch build (if using GPU)
- Bash (for some provided shell scripts). On Windows, use WSL or adapt commands for PowerShell.

## Create environment

Create and activate the Conda environment named `tpgd` with Python 3.8:

```bash
conda create -n tpgd python=3.8 -y
conda activate tpgd
```

## Install dependencies

Install Python dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```


## Usage

### 1) Generate data

Generate the dataset text files used by the first-stage training loader. From the project root:

```bash
cd scripts
python generate_data.py
```

The script produces `.txt` lists or dataset artifacts in the configured output folder. Inspect `scripts/generate_data.py` to point outputs to your dataset locations or to change split ratios.

## Training

### Stage 1

### 2) training

Run the first-stage training. From the project root:

```bash
cd tpgd/src
bash single_train.sh
```

`single_train.sh` should call the appropriate Python training entrypoint with relevant config. Edit script or configuration files to change hyperparameters or dataset paths.


Run the second-stage restoration training using PyTorch distributed launch (example uses 2 GPUs):

```bash
cd universal-restoration/config/tpgd-sde/options
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 train.py -opt=options/train.yml --launcher pytorch
```

Notes:
- Adjust `--nproc_per_node` to match the number of GPUs available.
- Change `--master_port` if the chosen port is in use.
- `-opt=options/train.yml` points to the YAML config for training; edit it to set dataset paths, model checkpoints, and training hyperparameters.
- Newer PyTorch versions recommend `torchrun` as an alternative to `torch.distributed.launch`.

### 4) Testing

After training, run evaluation or inference with the test configuration:

```bash
python test.py -opt=options/test.yml
```

The test configuration specifies dataset paths, model checkpoint locations, output directories, and evaluation metrics.

## Troubleshooting

- CUDA / PyTorch mismatch: Verify installed `torch` wheel matches your CUDA toolkit version. Reinstall `torch` if necessary.
- Distributed errors: Ensure network ports are free and environment variables (`MASTER_ADDR`, `MASTER_PORT`) are set correctly if using multi-node setups.
- Missing dependencies: Inspect top-level and component `requirements` or `setup` files and install any additional packages required by specific modules.
- Windows users: Some shell scripts use `bash`; run them under WSL or convert commands for PowerShell.


## License & Citation


