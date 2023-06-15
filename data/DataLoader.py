#!/usr/bin/env python
# coding: utf-8


from collections import Counter
import spacy
import pandas as pd
import numpy as np

#import gluonnlp as nlp
import torch
#from konlpy.tag import Mecab
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from data.Utils import collate_tokens, rev_max_len
from data.Vocab import Vocab
from data.preprocess import TextProcessing

from tqdm import tqdm


PAD_TOKEN = '<pad>'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNK_TOKEN = '<unk>'  # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '<sos>'  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '<eos>'  # This has a vocab id, which is used at the end of untruncated target sequences


class AmazonDataset(Dataset):
    def __init__(self, datapath, max_num_reviews=None, is_train=True, refs_path=None, vocab=None, max_len_rev=None, preprocess=False):
        '''
        datapath: path to the data file
        refs_path: path to reference summaries
        vocab: vocabulary
        preprocess: whether or not to preprocess texts
        '''
        super().__init__()
        
        self.is_train = is_train
        self.max_len_rev = max_len_rev
        
        # Load csv dataset, output text, and batch_ids
        # src_text: list of (input text, polarity) tuples
        # src_prod_idx: list of index of the first review of a given product
        # ref_dict: list of references (output texts)
        self.src_reviews, self.batch_indexes, self.src_prod_ids, self.references_dict = self.load_dataset(datapath, 
                                                                                                          max_num_reviews=max_num_reviews, 
                                                                                                          refs_path=refs_path)
        # Preprocess source texts
        if preprocess:
            processing = TextProcessing()
            print("Preprocessing dataset...")
            for i, (sentence, _) in  enumerate(tqdm(self.src_reviews)):
                self.src_reviews[i][0] = processing.preprocess(sentence)
        
        self.nlp = spacy.load("en_core_web_sm")
        self.vocab = vocab
        
    def __getitem__(self, index):
        start_idx = self.batch_indexes[index]
        end_idx = self.batch_indexes[index + 1]
        src_batch = []
        src_len_batch = []
        src_senti_batch = []
        src_prod_id = self.src_prod_ids[index]
        for index_b in range(start_idx, end_idx):
            txt = self.src_reviews[index_b][0]
            senti = self.src_reviews[index_b][1]
            tokens = self.truncate(self.tokenize(txt))
            length = len(tokens)
            
            src_batch.append(tokens)
            src_senti_batch.append(senti)
            src_len_batch.append(length)
            
        return src_batch, src_senti_batch, src_len_batch, src_prod_id
    
    def __len__(self):
        return len(self.batch_indexes) - 1  #Return batch size since it is the number of different product in dataset
    
    def build_vocab(self, vocab_size, min_freq, specials):
        counter = Counter()
        for (t,_) in tqdm(self.src_reviews, desc="Build vocabulary"):
            tokens = self.tokenize(t)
            counter.update(tokens)
            
        self.vocab = Vocab.from_counter(counter=counter, vocab_size=vocab_size, min_freq=min_freq, specials=specials)
        print(f"Vocabulary size: {len(self.vocab)}")
        return self.vocab
    
    def get_vocab(self):
        return self.vocab
    
    def tokenize(self, text):
        return [str(word) for word in self.nlp(str(text))]
    
    def preprocess(self, text):
        # TODO: add preprocessing functions relevant to our dataset
        text = text.lower()
        return text
    
    def truncate(self, tokens):
        if self.max_len_rev is not None and len(tokens) > self.max_len_rev:
            return tokens[:self.max_len_rev]
        else:
            return tokens
    
    def get_references(self):
        return self.references_dict
    
    def load_dataset(self, reviews_path, max_num_reviews=None, refs_path=None):
        def senti2num(sentiment):
            sentiment_map = {"positive":1.0, "negative":0.0, "neutral":2.0}
            return sentiment_map[str(sentiment)]
        
        # read data from file
        df = pd.read_csv(reviews_path, sep=",")
        prod_ids = df["prod_id"].unique().flatten().tolist()
        reviews_list = []
        batch_idx_list = []
        prod_id_list = []
        
        # Load references summaries
        if refs_path:
            refs_df = pd.read_csv(refs_path, sep=",")
            refs_dict = dict()
        else:
            refs_dict = dict()
        
        batch_idx = 0
        n_elements = df.shape[0]
        counter = 0
        duplicated = 0
        n_imbalanced = 0
        
        for i,prod_id in tqdm(enumerate(prod_ids), total=len(prod_ids), desc="Loading data", unit="item"):
            prod_reviews = df[df["prod_id"] == prod_id].copy()
            
            # Shuffle reviews
            prod_reviews = prod_reviews.sample(frac=1, random_state=42)
            prod_refs = []
            if refs_path:
                refs = refs_df[refs_df["prod_id"] == prod_id].copy()
                for col in col_summ:
                    summary = prod_reviews.loc[0, "{}".format(col)]
                    if summary != '' and pd.notna(summary):
                        prod_refs.append(summary)
                
            if max_num_reviews is None or prod_reviews.shape[0] <= max_num_reviews:
                # save batch index
                batch_idx_list.append(batch_idx)
                # get reviews texts
                reviews_text = prod_reviews["review"].values.flatten().tolist()
                # get reviews sentiments
                reviews_sentiment = prod_reviews["polarity"].apply(senti2num).values.flatten().tolist()
                # save references
                refs_dict[batch_idx] = prod_refs
                # add sentiment and text data
                reviews_list.extend(list(zip(reviews_text, reviews_sentiment)))
                prod_id_list.append(prod_id)
                batch_idx += prod_reviews.shape[0]
            # Split batch reviews into batches of size max_num_reviews
            else:
                # calc number of sub-batches to split current batch
                st = 0
                fn = min(st + max_num_reviews, prod_reviews.shape[0])
                while st < fn:
                    revs = prod_reviews[st:fn].copy()
                    if revs.shape[0] > 0:
                        # Make sure all batches have the same size
                        if revs.shape[0] < max_num_reviews:
                            n_missing = max_num_reviews - revs.shape[0]
                            duplicated += n_missing
                            tmp = prod_reviews[:st].sample(n=n_missing, random_state=42)
                            revs = pd.concat([revs, tmp])
                            assert revs.shape[0] == max_num_reviews
                        
                        # Ensure a mix of positive and negative in all batches
                        if revs[revs.polarity == "negative"].shape[0] == 0:
                            tmp = prod_reviews.copy()
                            revs = pd.concat([revs, tmp.loc[tmp.polarity == "negative"].sample(n=3, random_state=42)])[3:]
                            n_imbalanced += 1
                        elif revs[revs.polarity == "positive"].shape[0] == 0:
                            tmp = prod_reviews.copy()
                            revs = pd.concat([revs, tmp.loc[tmp.polarity == "positive"].sample(n=3, random_state=42)])[3:]
                            n_imbalanced += 1
                        
                        batch_idx_list.append(batch_idx)
                        # extract reviews text
                        revs_text = revs["review"].values.flatten().tolist()
                        # extract reviews sentiment
                        revs_sentiment = revs["polarity"].apply(senti2num).values.flatten().tolist()
                        # save references
                        refs_dict[batch_idx] = prod_refs
                        # add sentiment and text data
                        reviews_list.extend(list(zip(revs_text, revs_sentiment)))
                        prod_id_list.append(prod_id)
                        batch_idx += revs.shape[0]
                    st = fn
                    fn = min(st + max_num_reviews, prod_reviews.shape[0])
        
        # Taking last position into account for loader later
        batch_idx_list.append(len(reviews_list))
        
        assert n_elements == len(reviews_list) - duplicated
        assert sum([int(idx <= len(reviews_list)) for idx in batch_idx_list]) == len(batch_idx_list)
        
        if refs_path:
            refs_dict = None
            
        # print(f"{duplicated} ({100*duplicated/n_elements:.2f}%) duplicated reviews added.")
        # print(f"{n_imbalanced} imbalanced batches found.")
        
        return reviews_list, batch_idx_list, prod_id_list, refs_dict


class Batch:
    """
    Called by Dataloader and create/format elements in batch
    Args : iterable dataset and vocabulary provided by build_dataset function
    """
    def __init__(self, data, vocab):
        #src, src_len = list(zip(*data)) #receive a list of n text for each 1 element of batch
        src, src_senti, src_len, src_prod_id = data[0][0], data[0][1], data[0][2], data[0][3]
        
        self.vocab = vocab
        self.pad_id = self.vocab.pad()

        # Encoder info
        self.enc_input, self.enc_len = None, None
        # Additional info for pointer-generator network
        self.enc_input_ext, self.max_oov_len, self.src_oovs = None, None, None
        
        # Build batch inputs
        self.init_encoder_seq(src, src_len)

        # Save original strings
        self.src_text = src
        
        # Save polarity
        self.src_senti = torch.LongTensor(src_senti)
        
        # Save product id
        self.src_prod_id = src_prod_id

    def init_encoder_seq(self, src, src_len):
        """
        Take source texts and transform into tensors of ids corresponding to the vocabulary
        Transform list of src_len and group_idx into tensors
        """
        #Create list list of token ids corresonding in position in vocabulary
        src_ids = []
        for s in src:
            temp = [self.vocab.start()]
            temp.extend(self.vocab.tokens2ids(s))
            temp += [self.vocab.stop()]
            src_ids.append(temp)
        
        #Taking into account the adding of the token in src size
        src_len = [x + 2 for x in src_len]
        
        #Original src_ids function - don't implement sos and eos tags in src sentences
        #src_ids = [self.vocab.tokens2ids(s) for s in src]
        
        #Function transforming into tensor, padding it to max_len of batch
        self.enc_input = collate_tokens(values=src_ids, pad_idx=self.pad_id)
        
        #Transform into tensor length and group ids
        self.enc_len = torch.LongTensor(src_len)

        # Save additional info for pointer-generator - Determine max number of source text OOVs in this batch
        # Create list of list of token ids based on extended vocabulary and getting list of OOVs
        src_ids_temp, oovs = zip(*[self.vocab.source2ids_ext(s) for s in src])
        # Store the version of the encoder batch that uses article OOV ids
        
        #Adding start and end of sequence tokens to extended vocabulary
        src_ids_ext = []
        for s in src_ids_temp: 
            temp = [self.vocab.start()]
            temp.extend(s)
            temp.extend([self.vocab.stop()])
            src_ids_ext.append(temp)
        self.enc_input_ext = collate_tokens(values=src_ids_ext, pad_idx=self.pad_id)
        
        #Getting maximum of OOVs words in the batch
        self.max_oov_len = max([len(oov) for oov in oovs])
        
        # Store source text OOVs themselves
        self.src_oovs = oovs

    def __len__(self):
        return self.enc_input.size(0)

    def __str__(self):
        
        """
        Create callable object of the batch - batch.enc_len to get length
        """
        batch_info = {
            'src_text': self.src_text,
            'src_senti': self.src_senti,
            'src_prod_id': self.src_prod_id,
            'enc_input': self.enc_input,  # [btch_size x seq_len]
            'enc_input_ext': self.enc_input_ext,  # [batch x seq_len]
            'enc_len': self.enc_len,  # [batch_size]
            'src_oovs': self.src_oovs,  # list of length B
            'max_oov_len': self.max_oov_len,  # single int value
        }
        
        return str(batch_info)

    def to(self, device):
        """
        Store object on proper device - cuda or not cuda?
        """
        self.enc_input = self.enc_input.to(device)
        self.enc_input_ext = self.enc_input_ext.to(device)
        self.enc_len = self.enc_len.to(device)
        self.src_senti = self.src_senti.to(device)
        

# ============= END CLASS DEFINITION =================

def build_dataloader(file_path, vocab_size=10000, vocab_min_freq=1, vocab=None, is_train=True, 
                     shuffle_batch=False, max_num_reviews=None, refs_path=None, max_len_rev=None, pin_memory=True, 
                     num_workers=1, batch_size=1, preprocess=False, device="cpu"):
    """
    Create a dataset using the file in file_path, calls Pytorch DataLoader and returns the 'iterator' object for
    output - batch = next(iter(data_loader))
    """
    # Create dataset processed (tokenized and lowercase) - Field equivalent
    dataset = AmazonDataset(file_path, max_num_reviews, is_train, refs_path, vocab, max_len_rev=max_len_rev, preprocess=False)
    
    if is_train:
        specials = [PAD_TOKEN, UNK_TOKEN, START_DECODING, STOP_DECODING]
        vocab = dataset.build_vocab(vocab_size=vocab_size, min_freq=vocab_min_freq, specials=specials)
    else:
        assert vocab is not None
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_batch,
                             collate_fn=lambda data, v=vocab: Batch(data=data, vocab=v))
    if preprocess:
        references = dataset.get_references()
    else:
        references = None
        
    return data_loader, vocab, references
