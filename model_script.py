import pandas as pd
import time

#Personnalized libraries
from configs.config import DatasetConfig, HP
from data.DataLoader import build_dataloader
from utils.Errors import loss_estimation
from Procedures import Procedure
from model.lm import LanguageModel
from torch.utils.tensorboard import SummaryWriter
from configs.config import follow

import argparse
import re


def build_dataloaders(ds_config, dataset="train", vocab=None):
    """
    Instanciates dataloader for train, valid and test
    
    Input:
        ds_config: Dataset configurations
    Return:
        train_iter, valid_iter, test_iter: Dataloaders
        test_references: List of reference summaries
    """
    if dataset == "train":
        train_iter, vocab, _ = build_dataloader(
            file_path=ds_config.train_data, 
            vocab_size=ds_config.vocab_size,
            vocab_min_freq=ds_config.min_freq,
            vocab=None,
            is_train=True,
            shuffle_batch=False,
            max_num_reviews=ds_config.max_num_reviews,
            refs_path=None,
            max_len_rev=ds_config.max_len_rev,
            pin_memory=ds_config.pin_memory,
            num_workers=ds_config.workers,
            batch_size=ds_config.batch_size,
            device=ds_config.device
        )
        return train_iter, vocab

    elif dataset == "valid":
        valid_iter, _, valid_references = build_dataloader(
            file_path=ds_config.valid_data, 
            vocab_size=ds_config.vocab_size,
            vocab_min_freq=ds_config.min_freq,
            vocab=vocab,
            is_train=False,
            shuffle_batch=False,
            max_num_reviews=ds_config.max_num_reviews,
            refs_path=None,
            max_len_rev=ds_config.max_len_rev,
            pin_memory=ds_config.pin_memory,
            num_workers=ds_config.workers,
            batch_size=ds_config.batch_size,
            device=ds_config.device
        )
        return valid_iter

    elif dataset == "test":
        test_iter, _, test_references = build_dataloader(
            file_path=ds_config.test_data, 
            vocab_size=ds_config.vocab_size,
            vocab_min_freq=ds_config.min_freq,
            vocab=vocab,
            is_train=False,
            shuffle_batch=False,
            max_num_reviews=15,#ds_config.max_num_reviews,
            refs_path=None,
            max_len_rev=ds_config.max_len_rev,
            pin_memory=ds_config.pin_memory,
            num_workers=ds_config.workers,
            batch_size=ds_config.batch_size,
            device=ds_config.device
        )
        return test_iter, test_references
    else:
        raise Exception(f"Dataset '{dataset}' was not found.")


def load_parameters(agrs):
    """
    Load and update model and dataset parameters following specified parameters
    
    Input:
        args: user arguments
    Return:
        ds_config: Dataset configuration
        hp: Model hyperparameters
    """
    ds_config = DatasetConfig()
    hp = HP()
    
    # Update dataset parameters
    ds_config.batch_size = args.batch_size if args.batch_size else ds_config.batch_size
    
    # Update model parameters
    hp.use_rec = args.use_rec if args.use_rec else hp.use_rec
    hp.use_cls = args.use_cls if args.use_cls else hp.use_cls
    hp.use_grl = args.use_grl if args.use_grl else hp.use_grl
    hp.use_gat = args.use_gat if args.use_gat else hp.use_gat
    hp.use_proj = args.use_proj if args.use_proj else hp.use_proj
    hp.dec_hidden_type = args.dec_hidden_type if args.dec_hidden_type else hp.dec_hidden_type
    hp.mean_hidden_type = args.mean_hidden_type if args.mean_hidden_type else hp.mean_hidden_type
    hp.ref_hidden_type = args.ref_hidden_type if args.ref_hidden_type else hp.ref_hidden_type
    hp.gen_hidden_type = args.gen_hidden_type if args.gen_hidden_type else hp.gen_hidden_type
    hp.beam_decode = args.beam_decode if args.beam_decode else hp.beam_decode
    
    num_epochs = args.epochs
    if args.mode == "lm":
        hp.lm_epochs = num_epochs if num_epochs else hp.lm_epochs
    elif args.mode == "summ" or args.mode == "eval":
        hp.summarizer_epochs = num_epochs if num_epochs else hp.summarizer_epochs
    elif args.mode == "eval":
        pass
    else:
        raise Exception(f"mode '{args.mode}' is not supported. Try 'train', 'finetune', or 'eval'.")
    
    return ds_config, hp
    

def save_summaries(summaries, output_path, model_path):
    """
    Save generated summaries to file
    """
    df = []
    for e in summaries:
        prod_id = e[0]
        summary = e[1][0]
        df.append({
            "model_path": model_path,
            "prod_id": prod_id, 
            "summary": summary
        })
    df = pd.DataFrame(df)
    df.to_csv(output_path, index=False)


def run(args):
    # Load dataset config and model hypermarameters
    ds_config, hp = load_parameters(args)
    
    # Enables Tensorborad monitoring
    if follow["writer"]:
        comment = follow["Name"]
        writer = SummaryWriter(comment=comment)
    else:
        writer = None
    
    if args.mode == "lm" or args.mode == "summ":
        # Load datasets
        train_iter, vocab = build_dataloaders(ds_config, dataset="train")
        valid_iter = build_dataloaders(ds_config, dataset="valid", vocab=vocab)
        procedure = Procedure(hp, ds_config, vocab, writer=writer, train_ter=train_iter, valid_iter=valid_iter)
        
        if args.mode == "lm":
            # Train language model
            procedure.train_lm(model_name=args.model_name, tolerance=args.tolerance, check_every=args.check_every)
        elif args.mode == "summ":
            # Train summarizer
            procedure.train_summarizer(model_name=args.model_name, lm_path=args.lm_path, tolerance=args.tolerance, check_every=args.check_every)
    
    elif args.mode == "eval":
        # Load datasets
        _, vocab = build_dataloaders(ds_config, dataset="train")
        test_iter, _ = build_dataloaders(ds_config, dataset="test", vocab=vocab)
        procedure = Procedure(hp, ds_config, vocab)
        
        # Generate and save summaries
        summaries = procedure.generate_summaries(itr=test_iter, model_name=args.model_name)
        save_summaries(summaries, out_dir=args.output_dir, model_name=args.model_name)
    else:
        raise Exception(f"mode '{args.mode}' is not supported. Try 'train', 'finetune', or 'eval'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument("--mode", type=str, required=True, help="train or test.")
    parser.add_argument("-b", "--batch_size", type=int, default=None, help="Number of reviews per batch.")
    parser.add_argument("-e", "--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--use-rec", type=bool, dest="use_rec", default=None, help="Enable or disable reviews reconstruction. If false, the language model only uses the classifier.")
    parser.add_argument("--use-cls", type=bool, dest="use_cls", default=None, help="Enable or disable classifier.")
    parser.add_argument("--use-grl", type=bool, dest="use_grl", default=None, help="Enable or disable GRL.")
    parser.add_argument("--use-gat", type=bool, dest="use_gat", default=None, help="Enable or disable GAT.")
    parser.add_argument("--use-proj", type=bool, dest="use_proj", default=None, help="Enable or disable projection mechanism.")
    parser.add_argument("--dec-hidden", type=int, dest="dec_hidden_type", default=None, help="Hidden representations used when decoding the input reviews.")
    parser.add_argument("--mean-hidden", type=int, dest="mean_hidden_type", default=None, help="Hidden representations used to create the mean representations of the input reviews.")
    parser.add_argument("--ref-hidden", type=int, dest="ref_hidden_type", default=None, help="Hidden representations used to compute the cosine similarity.")
    parser.add_argument("--gen-hidden", type=int, dest="gen_hidden_type", default=None, help="Hidden representations used to generate summaries at test time.")
    parser.add_argument("--beam-decode", type=bool, dest="beam_decode", default=None, help="Enable or disable beam decoding.")
    
    parser.add_argument("--tolerance", type=int, default=3, help="Defines how many times to check the loss value before stopping the training early if the loss doesn't improve.")
    parser.add_argument("--check-every", dest="check_every", type=int, default=5, help="Number of epochs after which check the loss value in order to stop early.")
    
    parser.add_argument("--lm-path", type=str, dest="lm_path", default=None, help="Path where to save the trained language model or path to a pretrained language model to load.")
    parser.add_argument("--model-name", type=str, dest="model_name", default=None, help="Name of the model. If None, a name will be automatically generated.")
    parser.add_argument("--output-path", type=str, dest="output_path", default="./outputs/summaries/generated_summaries.csv", help="Path where to save generated summaries as CSV files.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run program
    run(args)