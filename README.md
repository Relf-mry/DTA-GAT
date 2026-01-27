# DTA-GAT: Dynamic Time-Aware Graph Attention Networks 

A PyTorch implementation of the DTA-GAT model for derived topic propagation prediction on social media.

## Project Structure

```
DTA-GAT/
├── src/                       # Source code
│   ├── models/                # Model implementations
│   │   ├── dta_gat.py        # Main DTA-GAT model
│   │   ├── node2vec.py       # Node2vec embedding
│   │   └── attractiveness.py # Topic attractiveness
│   ├── data/                  # Data loading
│   │   └── loader.py         # Dataset loader
│   └── utils/                 # Utilities
│       └── trainer.py        # Training utilities
├── scripts/                   # Experiment scripts
│   └── run_experiments.py    # Main experiments
├── data/                      # Data files
│   ├── pheme/                # PHEME dataset
│   └── twitter_covid19/      # Twitter COVID-19 dataset
├── results/                   # Experiment results
└── configs/                   # Configuration files
```

## Data Format

Each dataset should contain a `propagation.txt` file with the format:
```
派生话题ID,父话题ID,传播规模,时间戳
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run Experiments

```bash
cd scripts
python run_experiments.py
```

This will run experiments 5.2.1 (Topic Structure Feature Representation) and 5.2.2 (Topic Attractiveness Effectiveness) on both datasets.

### Results

Results will be saved in the `results/` directory:
- `figure_5_2_1.png` - Topic structure feature comparison
- `figure_5_2_2.png` - Topic attractiveness comparison  
- `all_results.json` - Detailed numerical results

## Model Components

### 1. Topic Structure Feature Representation
- Normalized co-occurrence strength (Equation 7)
- Node2vec embedding (Equations 8-12)

### 2. Topic Attractiveness Quantification
- Topic influence (Equation 14)
- Topic relatedness (Equation 15)
- Multi-linear regression fusion (Equation 16)

### 3. Time-Aware GAT Layer
- Weighted average co-occurrence time (Equation 17)
- Time-sensitive attention (Equations 18-21)

## Citation

If you use this code, please cite our paper.

## License

This project is for research purposes only.
