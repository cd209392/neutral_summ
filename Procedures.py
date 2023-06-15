import math

#torch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.lm import LanguageModel
from model.summarizer import Summarizer

#Getting metrics on tensorBoard
from torch.utils.tensorboard import SummaryWriter

#Personalized libraries
from utils.model_utils import freeze
from utils.Errors import loss_estimation
#from utils.ROUGE_eval import Rouge_eval

import time
import copy
from tqdm import tqdm
import re


class Procedure():
    def __init__(self, hp, ds_config, vocab, writer=None, train_ter=None, valid_iter=None):
        self.train_iter = train_ter
        self.valid_iter = valid_iter
        
        self.hp = hp
        self.ds = ds_config
        self.device = self.hp.device
        self.vocab = vocab
        
        self.lm = None
        self.summarizer = None
        
        self.acc_size = self.hp.acc_size
        
        # Learning parameters
        self.clip = self.hp.clip
        
        # Folowing results through tensorboard
        self.writer = writer

    def train_lm(self, model_name=None, tolerance=3, check_every=5):
        """
        Train the language model module
        
        Input:
            path (str): Path where to save the trained language model. If None, save to a default directory
            tolerance (int): Defines when to stop training if there is no improvement with the loss
            check_every (int): Check loss value every certain number of epochs
        """
        self.lm = LanguageModel(self.vocab, hp=self.hp).to(self.device)
        self.lm.apply(self.hp.weights_init)
        
        if not model_name:
            model_name = self.gen_model_name(model="lm")
        path = f"./outputs/models/lm/{model_name}"
        
        self.optimizer = self.hp.optimizer(
            params=self.lm.parameters(),
            lr=self.hp.lm_lr,
            weight_decay=self.hp.weight_decay
        )
        
        print(f"Training Language Model for {self.hp.lm_epochs} epochs...")
        n_epochs = self.hp.lm_epochs
        best_epoch = 0
        best_loss = float("inf")
        counter = tolerance
        
        cls_weight = self.hp.cls_weight
        rec_weight = 0.0  # For a certain number of epochs, ignore the reconstruction loss
        
        self.acc_size = self.hp.acc_size
        
        for epoch in range(n_epochs):
            # After a certain number of epochs, consider both the classifier and reconstruction losses
            if epoch == self.hp.cls_num_epochs:
                self.acc_size = 0
                rec_weight = self.hp.rec_weight
            
            epoch_st = time.perf_counter()
            self.lm.train()
            train_iterator_ = iter(self.train_iter)
            # Update gamma for GRL
            alpha = 2.0 / (1.0 + math.exp(-self.hp.grl_gamma * (epoch/n_epochs))) - 1.0
            train_loss = self.run_epochs(epoch, train_iterator_, cls_weight, rec_weight, alpha, model="lm", mode="Train")
            
            # Validation
            self.lm.eval()
            with torch.no_grad():
                valid_iterator_ = iter(self.valid_iter)
                valid_loss = self.run_epochs(epoch, valid_iterator_, cls_weight, rec_weight, alpha, model="lm", mode="Eval")
                print(f"| Epoch: {epoch+1:03} | Train loss: {train_loss:.3f} | Valid loss: {valid_loss:.3f} | time: {(time.perf_counter() - epoch_st):.2f}s'")
                
                if epoch >= self.hp.save_epoch:
                    # Save best model
                    if valid_loss <= best_loss:
                        best_loss = valid_loss
                        best_epoch = epoch
                        counter = tolerance
                        
                        # Save best model
                        #save_path = re.sub("\.pt$", f".{epoch}_epochs.pt", path)
                        save_path = path
                        torch.save(self.lm.state_dict(), save_path)
                    
                    elif epoch%check_every == 0:
                        counter -= 1
            
            if counter == 0:
                print(f"Ending training early after {epoch+1} epochs. best epoch: {best_epoch+1}")
                break
                
        print(f"Saved model in {path}.")
                    
    def train_summarizer(self, model_name, lm_path, tolerance=3, check_every=5):
        """
        Train the summarizer module
        
        Input:
            path (str): Path where to save the trained language model. If None, save to a default directory
            tolerance (int): Defines when to stop training if there is no improvement with the loss
            check_every (int): Check loss value every certain number of epochs
        """
        if not model_name:
            model_name = self.gen_model_name(model="summarizer")
        path = f"./outputs/models/summ/{model_name}"
        
        print(f"Loading Language Model from {lm_path}")
        
        self.lm = LanguageModel(self.vocab, hp=self.hp).to(self.device)
        self.lm.load_state_dict(torch.load(lm_path))
        
        enc_rec = self.lm.encoder_rec
        enc_cls = self.lm.encoder_cls
        dec = self.lm.decoder
        cls_ = self.lm.classifier
        emb_rec = self.lm.embedding_rec
        emb_cls = self.lm.embedding_cls
        grl = self.lm.grl
        
        # Freeze the weights of certain modules
        freeze(enc_rec)
        freeze(enc_cls)
        #freeze(dec)
        freeze(emb_rec)
        freeze(emb_cls)
        
        self.summarizer = Summarizer(enc_rec, enc_cls, dec, cls_, emb_rec, emb_cls, grl, self.vocab, self.hp).to(self.device)
        self.optimizer = self.hp.optimizer(
            params=self.summarizer.parameters(),
            lr=self.hp.summ_lr,
            weight_decay=self.hp.weight_decay
        )
        
        model_name = self.gen_model_name(model="summarizer")
        n_epochs = self.hp.summarizer_epochs
        
        print(f"Training Summarizer for {n_epochs} epochs...")
        best_loss = float("inf")
        best_epoch = 0
        counter = tolerance
        cls_weight = self.hp.cls_weight
        rec_weight = 1.0
        
        for epoch in range(n_epochs):
            epoch_st = time.perf_counter()
            self.summarizer.train()
            train_iterator_ = iter(self.train_iter)
            # Update gamma for GRL
            alpha = 2.0 / (1.0 + math.exp(-self.hp.grl_gamma * (epoch/n_epochs))) - 1.0
            train_loss = self.run_epochs(epoch, train_iterator_, cls_weight, rec_weight, alpha, model="summarizer", mode="Train")
            
            # Validation
            self.lm.eval()
            with torch.no_grad():
                valid_iterator_ = iter(self.valid_iter)
                valid_loss = self.run_epochs(epoch, valid_iterator_, cls_weight, rec_weight, alpha, model="summarizer", mode="Eval")
                print(f"| Epoch: {epoch+1:03} | Train loss: {train_loss:.3f} | Valid loss: {valid_loss:.3f} | time: {(time.perf_counter() - epoch_st):.2f}s'")
            
                if valid_loss <= best_loss:
                    best_loss = valid_loss
                    best_epoch = epoch
                    counter = tolerance
                    
                    # Save best model
                    #save_path = re.sub("\.pt$", f".{epoch}_epochs.pt", path)
                    save_path = path
                    torch.save(self.summarizer, save_path)
                    
                elif epoch%check_every == 0:
                    counter -= 1   
            
            torch.cuda.empty_cache()
                        
            if counter == 0:
                print(f"Ending training early after {epoch+1} epochs. best epoch: {best_epoch+1}")
                break
        
        print(f"Saved model in {path}.")
    
    def run_epochs(self, epoch, iterator_, cls_weight=4.0, rec_weight=1.0, alpha=0.5, model="lm", mode="Train", debug=False):
        epoch_loss = 0.0
        epoch_rec = 0.0
        epoch_cls = 0.0
        epoch_acc = 0.0
        epoch_acc_no_grl = 0.0
        epoch_sim = 0.0
        
        iter_size = len(iterator_)
        
        ACC_SIZE = self.acc_size
        accumulation_steps = 0
        
        # for i in tqdm(range(iter_size), unit="batch"):
        for i in range(iter_size):
            batch = iterator_.next()
            
            src_input = batch.enc_input.permute(1, 0).contiguous().to(self.device) # [seq_len, batch_size]
            trg_input = batch.enc_input.permute(1, 0).contiguous().to(self.device) # [seq_len, batch_size]
            src_senti = batch.src_senti.to(self.device) # [batch_size]
            src_len = batch.enc_len.to(self.device)
            
            batch_size = src_input.size(1)
            
            rec_loss = torch.tensor([0.0]).to(self.device)
            cls_loss = torch.tensor([0.0]).to(self.device)
            sim_loss = torch.tensor([0.0]).to(self.device)
            acc = 0.0
            acc_no_grl = 0.0
            
            accumulation_steps += batch_size
            
            if model == "lm":
                outputs, cls_preds, cls_preds_tmp = self.lm(src_input, trg_input, src_senti, alpha, tf_ratio=self.hp.tf_ratio)
                # output = [seq_len, batch_size, output_dim]
                # Compute reconstruction loss
                if outputs is not None:
                    output_dim = outputs.shape[-1]
                    outputs = outputs[1:].view(-1, output_dim)  # [(seq_len-1) * batch_size, output_dim]
                    trg_input = trg_input[1:].view(-1)  # [(seq_len-1) * batch_size]
                    #rec_criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.pad())
                    rec_criterion = nn.NLLLoss(ignore_index=self.vocab.pad())
                    rec_loss = rec_criterion(outputs, trg_input)
                
                # Compute classification loss
                if cls_preds is not None:
                    cls_criterion = nn.BCELoss()
                    cls_loss = cls_criterion(cls_preds, src_senti.to(torch.float32))
                    # Compute classification accuracy
                    acc = binary_accuracy(cls_preds, src_senti)
                    acc_no_grl = binary_accuracy(cls_preds_tmp, src_senti)
                
                loss = rec_weight*rec_loss + cls_weight*cls_loss
                    
            elif model == "summarizer":
                outputs, cos_sim = self.summarizer(src_input, trg_input, src_len, tf_ratio=self.hp.tf_ratio, gumbel_hard=True)
                # outputs: [seq_len, batch_size, output_dim]
                output_dim = outputs.shape[-1]
                outputs = outputs[1:].view(-1, output_dim)  # [(seq_len-1) * batch_size, output_dim]
                trg_input = trg_input[1:].view(-1)  # [(seq_len-1) * batch_size]
                #rec_criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.pad())
                rec_criterion = nn.NLLLoss(ignore_index=self.vocab.pad())
                rec_loss = rec_criterion(outputs, trg_input)
                    
                # Compute cosine similarity
                sim_loss = 1.0 - abs(cos_sim)
                loss = sim_loss
            else:
                raise Error(f"Model {model} was not found!")
            
            if mode == "Train":
                loss.register_hook(lambda grad: grad/batch_size)
                loss.backward()
                if accumulation_steps >= ACC_SIZE:
                    params = self.lm.parameters() if model == "lm" else self.summarizer.parameters()
                    torch.nn.utils.clip_grad_norm_(params, self.clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # Reset accumulation
                    accumulation_steps = 0
            
            epoch_loss += loss.item()
            epoch_rec += rec_loss.item()
            epoch_cls += cls_loss.item()
            epoch_acc += acc
            epoch_acc_no_grl += acc_no_grl
            epoch_sim += sim_loss.item()
        
        if self.writer is not None:
            self.writer.add_scalar(f"{mode}/Loss", epoch_loss / iter_size, global_step=epoch)
            if model == "lm":
                self.writer.add_scalar(f"{mode}/Rec Loss", epoch_rec / iter_size, global_step=epoch)
                self.writer.add_scalar(f"{mode}/Cls Loss", epoch_cls / iter_size, global_step=epoch)
                self.writer.add_scalar(f"{mode}/Cls Accuracy", epoch_acc / iter_size, global_step=epoch)
                self.writer.add_scalar(f"{mode}/Cls Accuracy (no GRL)", epoch_acc_no_grl / iter_size, global_step=epoch)
            else:
                self.writer.add_scalar(f"{mode}/Sim Loss", epoch_sim / iter_size, global_step=epoch)
                self.writer.add_scalar(f"{mode}/Rec Loss", epoch_rec / iter_size, global_step=epoch)
            
        return epoch_loss / iter_size

    def reconstruct_reviews(self, itr, lm_path, batch_idx=None):
        """
        Reconstruct reviews
        """
        print(f"Loading Language Model from {lm_path}")
        
        # Load pretrained language model
        lm = LanguageModel(self.vocab, hp=self.hp).to(self.device)
        lm.load_state_dict(torch.load(lm_path))
        
        lm.eval()
        all_reviews = []
        iterator_ = iter(itr)
        with torch.no_grad():
            for i in range(len(iterator_)):
                batch = iterator_.next()
                if batch_idx is not None and i not in batch_idx:
                    continue
                batch.to(self.device)
                # Unpack batch
                src_input = batch.enc_input.permute(1,0).contiguous() 
                trg = batch.enc_input.permute(1,0).contiguous()
                reviews = lm.inference(src_input, trg)
                prod_ids = batch.src_prod_id
                for j, (ids, rev) in enumerate(zip(src_input.permute(1,0), reviews)):
                    pid = prod_ids
                    og_rev = " ".join([self.vocab.id2word(t) for t in ids if t not in [0, 2, 3]])
                    all_reviews.append((pid, rev, og_rev))
        
        return all_reviews
    
    def generate_summaries(self, itr, model_name, batch_idx=None):
        """
        Generate summaries
        """
        path = f"./outputs/models/summ/{model_name}"
        
        print(f"Loading summarizer from {path}")
        
        # Load summarizer
        summarizer = torch.load(path)
        summarizer.hp = self.hp
        
        summarizer.eval()
        
        all_summaries = []
        
        iterator_ = iter(itr)
        tot_elements = len(batch_idx) if batch_idx else len(iterator_)
        
        with torch.no_grad():
            for i in tqdm(range(tot_elements)):
                batch = iterator_.next()
                if batch_idx is not None and i not in batch_idx:
                    continue
                batch.to(self.device)
                # Unpack batch
                src_input = batch.enc_input.permute(1,0).contiguous()
                revs = [self.vocab.outputids2words(ids) for ids in batch.enc_input]
                src_len = batch.enc_len
                prod_id = batch.src_prod_id
                summaries, hiddens, mean_hiddens = summarizer.inference(src_input, src_len)
                all_summaries.append((prod_id, summaries))
        
        return all_summaries
    
    def gen_model_name(self, model="lm"):
        """
        Generate a name of a model
        """
        hidden_states = {0: "baseH", 1: "Hhat", 2: "Htilt"}
        dec_hid = hidden_states.get(self.hp.dec_hidden_type)
        sum_hid = hidden_states.get(self.hp.mean_hidden_type)
        batch_size = self.ds.max_num_reviews
        
        params = []
        
        if self.hp.use_gat:
            params.append("GAT")
        if self.hp.use_grl:
            params.append("GRL")
        if self.hp.use_proj:
            params.append("proj")
        
        if len(params) == 3:
            lm_name = "fullLM"
        else:
            if not self.hp.use_rec and self.hp.use_cls:
                lm_name = "cls"
            elif not self.hp.use_cls and self.hp.use_rec:
                lm_name = "rec"
            else:
                lm_name = "baseLM"
            
            lm_name = f"{lm_name}.{'.'.join(params)}"
            
        if model == "lm":
            n_epochs = self.hp.lm_epochs
            name = f"{lm_name}.batch_{batch_size}_docs.{dec_hid}.lr{self.hp.lm_lr}.{self.hp.lm_epochs}_epochs.pt"
        elif model == "summarizer":
            n_epochs = self.hp.summarizer_epochs
            name = f"summ.{lm_name}.batch_{batch_size}_docs.lm_lr{self.hp.lm_lr}.dec_{dec_hid}.sum_{sum_hid}.lr{self.hp.summ_lr}.{self.hp.summarizer_epochs}_epochs.pt"
        else:
            raise Error(f"model {model} is not recognized")
            
        return name

    
def binary_accuracy(preds, y):
    """
    Compute the accuracy score
    """
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float()
    acc = correct.sum() / preds.shape[0]
    return acc
