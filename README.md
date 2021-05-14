# BERT Fine-Tuning for Congressional Bill Subject Classification

## Environment Setup

This project requires Python 3.8.1. Once installed, you can install the dependencies by running the following command inside the project folder:

```
python -m pip install -r requirements.txt
```

If you want to generate the dataset yourself, you need to create accounts at both [LegiScan](https://legiscan.com/legiscan) and [OpenSecrets](https://www.opensecrets.org/open-data/api) to get API keys for each.

Place each API key in environment variables called `LEGISCAN_KEY` and `OPENSECRETS_KEY`.

## Running the Code

> Please note that this code was designed with the assumption that I would be the only one running it -- some files still have debug printing, some are being changed often, and some might not work at all for you, since I can only test on my own hardware.
> None of the files have argparse, any changes will have to be done in code for now.
> Proceed with caution.

Each file has different functionality baked in. You will need to run the code in this order:

 1. Run `src/data_handlers/data_getter.py`. This will generate the dataset and place it in `raw_data/dataset.p`.
 2. Run `src/data_handlers/data_prepro.py`. This will prepare the dataset for BERT and place the result in `clean_data/transformer_ready_data_1229.p`.
 3. Finally, run `src/main.py`. This will begin fine tuning BERT.
 
