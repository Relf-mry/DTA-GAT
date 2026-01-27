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
```

### Install Dependencies

```bash
#Clone the project:

git clone https://github.com/Relf-mry/DTA-GAT.git

#Install dependencies
pip install -r requirements.txt
```

## Dataset

The datasets used in this paper are not directly included in this repository due to licensing restrictions, but they are publicly available and can be obtained from the following sources:

<table>
  <thead>
    <tr>
      <th style="width:20%">Dataset Name</th>
      <th style="width:50%">Description</th>
      <th style="width:30%">Access Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>COVID-19-rumor dataset</td>
      <td>A large-scale Twitter dataset for COVID-19 rumor detection and propagation</td>
      <td><a href="https://github.com/MickeysClubhouse/COVID-19-rumor-dataset">https://github.com/MickeysClubhouse/COVID-19-rumor-dataset</a></td>
    </tr>
    <tr>
      <td>PHEME dataset</td>
      <td>A benchmark dataset for rumor detection and veracity classification</td>
      <td><a href="https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078">https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078</a></td>
    </tr>
    <tr>
      <td>Weibo</td>
      <td>Chinese social media dataset for rumor detection</td>
      <td><a href="https://github.com/thunlp/CED">https://github.com/thunlp/CED</a></td>
    </tr>
  </tbody>
</table>


## Citation

If you use this code, please cite our paper.

## License

This project is for research purposes only.
