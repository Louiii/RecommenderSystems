# Recommender Systems Coursework
## Context-aware song recommender systems

## Hierarchy
```
.
├── data
│   ├── Context_POP_RND // data from #NowPlaying
│   ├── Context_POP_USER // data from #NowPlaying
│   ├── rdn_train_final.json // my extracted data for training
│   ├── rdn_test_final.json // my extracted data for testing
│   ├── usr_train_final.json // additional dataset to train
│   └── usr_test_final.json // additional dataset to test
├── GUI_database
│   └── users.json // the likes/dislikes for users that use the GUI
├── SVD-model
│   └── // User profiles/hidden state of the model
├── TF-model
│   └── // User profiles/hidden state of the model
├── Technique1_SVD.py // the first model I use
├── Technique2_TF.py // the second model I use
├── GUI.py // the interface for the user
├── EvaluationMetrics.py // the functions for computing performance
├── RunMetrics.py
└── WriteFinalDataSet.py // the file I used to create the final dataset
```

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the following packages.
- numpy
- matplotlib
- tqdm
- scipy
- PySimpleGUIQt

Replace PACKAGE_NAME with the names in the list.
```bash
pip install PACKAGE_NAME
```

## Running
Replace FILE_NAME with the name of the file you would like to run.
```bash
python FILE_NAME.py
```
To use the recommender systems, run 'GUI.py'. To test them run 'RunMetrics.py'. Unfortunately #NowPlaying didn't provide the names of the tracks from Spotify, to get them back one must use the Spotify API (requires an account) to retrieve them from the track ids.
