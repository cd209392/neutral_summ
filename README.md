# neutral_summ
## Description
To complete

## Installation
* Python version: ^3.6
* Run `conda env create --name <env name> -f environment.yml` to install all necessary libraries using [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html).
* Create the following directories: `./outputs/models/lm/`, `./outputs/models/summ/`, `./outputs/summaries/`, and `./data/processed/`

## Usage
#### Define parameters:
* All of the model parameters are defined in `./configs/config.py`. 
* Some parameters as well as training config can also be defined directly on the commandline. Run `python model_script.py --help` for details.

#### How to run:
1. [Optional] Preprocess data using `python preprocess_data.py --source <Path to unprocessed dataset>`
2. Train the language model using `python model_script.py --mode lm [ --epochs <number of epochs> --model-name <name of the model> ]`
3. Use the trained language model (or pretrained model) to finetune the summarizer. Run `python model_script.py --mode summ --lm-path <path to the language model> [ --epochs <number of epcochs> --model-name <name of the model> ]`
4. Generate summaries with the command `python model_script.py --mode eval --model-name <name of the summarizer model> [ --output-dir <directory where to save generated summaries> ]`

#### Metrics
Refer to the notebooks in `./eval/` to compute the different evaluation metrics.

## Dataset
The dataset used in our experiment was built using the [Amazon Reviews Dataset](https://jmcauley.ucsd.edu/data/amazon/). The raw data comes in JSON files. 
Run the following command to build a formatted dataset in CSV: `python build_dataset.py --source <path to downloaded dataset> [ --category <product category> --out_dir <directory where to save built dataset> ]`. 
The constructed datasets (train/eval/test) should have the following columns: 
* `category`: product category
* `prod_id`: product ID
* `rating`: review rating
* `polarity`: overall review sentiment (postive, negative, neutral)
* `review`: review text
* `review_id`: unique ID of the review for a given product (e.g. 0, 1, 2, 3, 4, etc.)

## Project Organization
To complete