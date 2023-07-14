# **Review Helpfulness Prediction**

In the current experiment we aim to evaluate the performance of RSQTOA framework
with a real world dataset. More specifically, we used Yelp Academic Dataset to
predict the helpfulness of the reviews posted by users.

## Usage

The present directory consist of two folders i.e. `data` and `models`. Where the
`data` folder consists of all the necessary information and procedure followed
build the final features that we end up using to build our regressors. And `model`
folder consists of all the models trained during the experiment with the results
that were obtained.

### Data

The folder structure of `data` folder is as follows:

```
data
│───raw
│   │
│   │───csv
│   │   │
│   │   │───yelp_academic_dataset_business.csv
│   │   │───yelp_academic_dataset_review.csv
│   │   └───yelp_academic_dataset_user.csv
│   │
│   └───json
│       │
│       │───yelp_academic_dataset_business.json
│       │───yelp_academic_dataset_review.json
│       └───yelp_academic_dataset_user.json
│
│───intermediate
│   │ 
│   │───final_data_useful_gte_10.csv
│   │───processed_review_data.csv
│   └───yelp_db.sqlite 
│
│───01_data_modelling.ipynb
│───02_feature_extraction.ipynb
│───03_feature_selection.ipynb
└───data.csv
```

The above diagram gives a rough overview of the `data` folder's structure. The
important thing to notice here are three notebooks. By using these notebooks one
can regenerate the data used in this experiment. Below is a brief overview of each
notebook. For more detailed information, please visit the specific notebook.

- **[01_data_modelling.ipynb]**: This notebook is concerned with generating `csv`
  files from the `json` files. As the size of the dataset is quite
  large. Hence, we created a SQL database based on the csv files that would be quite
  helpful as we might need to perform join and search operation in feature extraction
  stage. **Note:** we cannot publish the Yelp Data. Please follow [raw json files instructions] 
  to download the raw `json` files and place them at respective directory. 
- **[02_feature_extraction.ipynb]**: This notebook is concerned with generating
  the additional features that might be helpful in building a good regressor. Please
  refer to the file for more detailed description of feature generation process.
  The output of this notebook is [final_data_useful_gte_10.csv]. 
- **[03_feature_selection.ipynb]**: This notebook is concerned with feature
  transformation and more importantly selecting important features out of all the
  features. Please refer to the notebook for more detailed documentation and step
  involved.
- **[data.csv]**: Final curated data that we intend to use for our experiments.

### Models

The folder structure of `models` folder is as follows:

```
models
│   
│───ann
│   │ 
│   │───logs
│   │───results.csv
│   └───runner.py 
│
│───elasticnet
│   │ 
│   │───hyperparameter_search.ipynb 
│   │───logs
│   │───results.csv
│   └───runner.py 
│
│───grid
│   │ 
│   │───config.yml
│   │───logs
│   │───loss_hist_*.csv
│   │───results.csv
│   └───runner.py 
│
│───interpolator
│   │ 
│   │───config.yml
│   │───logs
│   │───loss_hist_*.csv
│   │───results.csv
│   └───runner.py 
│
│───linear_regression
│   │ 
│   │───logs
│   │───results.csv
│   └───runner.py
│ 
│───plots.ipynb
└───... generated images from plots.ipynb

```

The above diagram gives a rough overview of the `models` folder structure. The directory consists of all the different 
approaches we tried to build a regressor to predict reviews helpfulness. The subdirectories `grid` and `interpolator` 
consists of RSQTOA backed approaches. For each model 20-Fold cross validation was performed. Hence, the `runner.py` 
is designed in such a way. Furthermore, these `runner.py` can also be served as a usage examples/demonstration of how to 
use the framework with proposed `Grid Sampling` and `Data Approximation` approaches such as `interpolation`.

Each model folder consists of some additional files the purpose of each of them are described below:

- **[ann]**: The folder demonstrate the use of Artificial Neural Network (ANN) for building a regressor to predict
review helpfulness. The `logs` and `results.csv` file consists of logs and performance metrics obtained during 20-Fold
cross validation that was performed.
- **[elasticnet]**: The folder demonstrate the use of ElasticNet for building a regressor to predict review helpfulness. 
The `logs` and `results.csv` file consists of logs and performance metrics obtained during 20-Fold cross validation 
that was performed. Lastly, `hyperparameter_search.ipynb` consists of steps that were takes to come up with the 
optimal hyperparameters for ElasticNet to solve the problem at hand.
- **[grid]**: The folder demonstrate the use of RSQTOA Framework with the proposed Grid Sampling algorithm for building 
a regressor to predict review helpfulness. The `logs` and `results.csv` file consists of logs and performance metrics 
obtained during 20-Fold cross validation that was performed. Lastly, `config.yml` consists of the hyperparameters that
were used for the RSQTOA framework and `loss_hist_*.csv` consists of the loss history that was observed for each fold 
along 100 epochs.
- **[interpolator]**: The folder demonstrate the use of RSQTOA Framework with the Nearest Neighbour Interpolator for 
building a regressor to predict review helpfulness. The `logs` and `results.csv` file consists of logs and performance 
metrics obtained during 20-Fold cross validation that was performed. Lastly, `config.yml` consists of the hyperparameters 
that were used for the RSQTOA framework and `loss_hist_*.csv` consists of the loss history that was observed for each 
fold along 100 epochs.
- **[linear_regression]**: The folder demonstrate the use of Ordinary Least Square for building a regressor to predict
review helpfulness. The `logs` and `results.csv` file consists of logs and performance metrics obtained during 20-Fold 
cross validation that was performed.


**[plots.ipynb]** Is a notebook which accumulates the results obtained from building different models and plots the 
comparative performance scores. 


[01_data_modelling.ipynb]: data/01_data_modelling.ipynb
[02_feature_extraction.ipynb]: data/02_feature_extraction.ipynb
[03_feature_selection.ipynb]: data/03_feature_selection.ipynb
[data.csv]: data/data.csv
[ann]: models/ann
[elasticnet]: models/elasticnet
[grid]: models/grid
[interpolator]: models/interpolator
[linear_regression]: models/linear_regression
[plots.ipynb]: models/plots.ipynb
[raw json files instructions]: data/raw/json
[raw csv files instructions]: data/raw/csv
