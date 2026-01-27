# DTA-GAT: Dynamic Topic Attractiveness Graph Attention Network 

This repository provides the official implementation of the DTA-GAT (Dynamic Topic Attractiveness Graph Attention Network) proposed in our paper for modeling and predicting the dynamic propagation patterns of derivative topics in social media.
The model integrates topic attractiveness modeling and time-aware attention mechanisms to capture the temporal evolution of topic relationships and user engagement in large-scale social networks.

The released code supports the complete experimental pipeline, including data preprocessing, topic co-occurrence network construction, attractiveness modeling, dynamic graph representation learning, and propagation trend prediction.

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
## Requirements

numpy==2.1.2
pandas==2.2.3
scipy==1.14.1

torch>=2.6.0
torch_geometric==2.6.1
tqdm==4.66.5

## Dataset

The datasets used in this paper are not directly included in this repository due to licensing restrictions, but they are publicly available and can be obtained from the following sources:

| Dataset Name            | Description                                                                 | Access Link |
|-------------------------|-----------------------------------------------------------------------------|-------------|
| COVID-19-rumor dataset  | A large-scale Twitter dataset for COVID-19 rumor detection and propagation | https://github.com/MickeysClubhouse/COVID-19-rumor-dataset |
| PHEME dataset           | A benchmark dataset for rumor detection and veracity classification        | https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078 |



## Install Dependencies

```bash
#Clone the project:

git clone https://github.com/Relf-mry/DTA-GAT.git

#Install dependencies
pip install -r requirements.txt
```



## Citation

If you use this code, please cite our paper.

## License

This project is for research purposes only.
