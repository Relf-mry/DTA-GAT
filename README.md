# DTA-GAT: Dynamic Topic Attractiveness Graph Attention Network 

This repository provides the official implementation of the DTA-GAT (Dynamic Topic Attractiveness Graph Attention Network) proposed in our paper for modeling and predicting the dynamic propagation patterns of derivative topics in social media.
The model integrates topic attractiveness modeling and time-aware attention mechanisms to capture the temporal evolution of topic relationships and user engagement in large-scale social networks.


## Project Structure

### Requirements

```text
numpy==2.1.2
pandas==2.2.3
scipy==1.14.1

torch>=2.6.0
torch_geometric==2.6.1
tqdm==4.66.5


### Install Dependencies

```bash
#Clone the project:

git clone https://github.com/Relf-mry/DTA-GAT.git

#Install dependencies
pip install -r requirements.txt
```

## Dataset

The datasets used in this paper are not directly included in this repository due to licensing restrictions, but they are publicly available and can be obtained from the following sources:

| Dataset Name           | Description                                                              | Access Link |
|------------------------|---------------------------------------------------------------------------|-------------|
| COVID-19-rumor dataset | A large-scale Twitter dataset for COVID-19 rumor detection and propagation | https://github.com/MickeysClubhouse/COVID-19-rumor-dataset |
| PHEME dataset          | A benchmark dataset for rumor detection and veracity classification       | https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078 |
| Weibo                  | Chinese social media dataset for rumor detection                          | https://github.com/thunlp/CED |
 


## Citation

If you use this code, please cite our paper.

## License

This project is for research purposes only.
