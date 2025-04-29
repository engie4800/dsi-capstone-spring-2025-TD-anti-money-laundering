### Directory Explanation

1. `ibm_eda.ipynb` contains preliminary EDA and initial temporal feature engineering
2. `practice_gnn_colab.ipynb` uses a simplified version of GINe model from IBM/Mulit-GNN published repo -- working/messy
3. `temporal_transformations.ipynb` picks out code from `ibm_eda.ipynb` used to transform temporal features for use in model pipeline
4. `example_gnn.ipynb` provides an example for how to use the model pipeline for GNN, done on CPU within vscode
5. `example_gnn_colab.ipynb` provides an example for how to use the model pipeline for GNN, done in Google Colab with GPU or CPU
6. `create_subgraph.ipynb` is the code used to sample 25% of the HI-Small data for our practice usage