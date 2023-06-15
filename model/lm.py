import random
import torch
import torch.nn as nn

from model.revgrad import GradReversal
from utils.model_utils import load_gloveembeddings, project_vector, cosine_similarity, postprocess, is_special, clean_up_tokenization, mask_range, logits_to_prob
from model.attention import GAT
from model.encoder import Encoder
from model.decoder import Decoder
from model.classifier import DomainClassifier

import re

class LanguageModel(nn.Module):
    def __init__(self, vocab, hp):
        super().__init__()

        self.hp = hp  # Hyperparametters
        self.device = self.hp.device
        self.vocab = vocab
        self.src_pad_idx = vocab.pad()
        self.input_dim = len(vocab)
        self.output_dim = len(vocab)
        self.num_tgr_docs_domains = self.hp.num_tgr_docs_domains

        ######### Embeddings
        self.embedding_rec, self.embedding_cls = None, None
        if self.hp.use_pretrained_embs:
            emb_vecs = load_gloveembeddings(vocab, Glove_name=self.hp.glove_name, Glove_dim=self.hp.emb_dim)
            if self.hp.use_rec:
                self.embedding_rec = nn.Embedding.from_pretrained(emb_vecs, freeze=False, padding_idx=self.vocab.pad())
            if self.hp.use_cls:
                self.embedding_cls = nn.Embedding.from_pretrained(emb_vecs, freeze=False, padding_idx=self.vocab.pad())
        else:
            if self.hp.use_rec:
                self.embedding_rec = nn.Embedding(self.input_dim, self.hp.enc_emb_dim, padding_idx=self.vocab.pad())
            if self.hp.use_cls:
                self.embedding_cls = nn.Embedding(self.input_dim, self.hp.enc_emb_dim, padding_idx=self.vocab.pad())
            
        ######### Graph Attention Network
        self.gat_rec, self.gat_cls = None, None
        if self.hp.use_gat:
            if self.hp.use_rec:
                self.gat_rec = GAT(device=self.device,
                                    input_dim=self.hp.hid_dim * 2,
                                    output_dim=self.hp.hid_dim * 2,
                                    hidden_dim=self.hp.gat_hidden,
                                    dropout=self.hp.gat_drop,
                                    alpha=self.hp.gat_alpha,
                                    num_heads=self.hp.gat_heads)
            if self.hp.use_cls:
                self.gat_cls = GAT(device=self.device,
                                    input_dim=self.hp.hid_dim * 2,
                                    output_dim=self.hp.hid_dim * 2,
                                    hidden_dim=self.hp.gat_hidden,
                                    dropout=self.hp.gat_drop,
                                    alpha=self.hp.gat_alpha,
                                    num_heads=self.hp.gat_heads)
                
        ######### Encoders
        self.encoder_rec, self.encoder_cls, self.decoder = None, None, None
        if self.hp.use_rec:
            self.encoder_rec = Encoder(input_dim=self.input_dim,
                                       hid_dim=self.hp.hid_dim,
                                       emb_dim=self.hp.enc_emb_dim,
                                       gat_attn=self.gat_rec,
                                       num_layers=self.hp.enc_layers,
                                       dropout=self.hp.enc_drop)
        if self.hp.use_cls:
            self.encoder_cls = Encoder(input_dim=self.input_dim,
                                       hid_dim=self.hp.hid_dim,
                                       emb_dim=self.hp.enc_emb_dim,
                                       gat_attn=self.gat_cls,
                                       num_layers=self.hp.enc_layers,
                                       dropout=self.hp.enc_drop)

        ######### Decoders
        if self.hp.use_rec:
            self.decoder = Decoder(output_dim=self.output_dim,
                                   hid_dim=self.hp.hid_dim,
                                   emb_dim=self.hp.enc_emb_dim,
                                   dropout=self.hp.dec_drop)
        
        ######### Classifier
        self.classifier = None
        self.grl = None
        if self.hp.use_cls:
            # used to reduce the dimension of the classifier input
            if self.num_tgr_docs_domains > 1:
                self.fc = nn.Linear(self.hp.hid_dim * self.num_tgr_docs_domains, self.hp.hid_dim)
            
            if self.hp.use_grl:
                self.grl = GradReversal()
            
            self.classifier = DomainClassifier(self.hp.hid_dim, self.hp.cls_hidden)
        
    def classify(self, src_senti, cls_hidden, rec_hidden, alpha):
        # cls_hiddden: [batch_size, hid_dim]
        # rec_hidden: [batch_size, hid_dim]
        batch_size = src_senti.shape[0]
        
        # Concat target domain docs representations
        if self.num_tgr_docs_domains == 1:
            hiddens = sample_target_docs(src_senti, cls_hidden, batch_size, pool="opposite", concat=self.hp.concat_docs) # [batch_size, hid_dim]
        elif self.num_tgr_docs_domains == 2:
            same_hiddens = sample_target_docs(src_senti, cls_hidden, batch_size, pool="same", concat=self.hp.concat_docs)
            opp_hiddens = sample_target_docs(src_senti, cls_hidden, batch_size, pool="opposite", concat=self.hp.concat_docs)
            hiddens = torch.concat([opp_hiddens, same_hiddens], dim=1) # [batch_size, hid_dim * 2]
            hiddens = self.fc(hiddens) # [batch_size, hid_dim]
        # No concatenation
        else:
            hiddens = cls_hidden
            
        cls_hidden = hiddens
        
        # Gradient Reversal Layer
        if self.hp.use_grl:
            cls_hidden = self.grl(cls_hidden, alpha=alpha)  # [batch_size, hid_dim]
                
        # Classification
        cls_preds = self.classifier(cls_hidden) # [batch_size]
        
        # Projection Mechanism
        if self.hp.use_proj:
            h_hat = project_vector(rec_hidden, cls_hidden, self.device)  # [batch_size, dec_dim]
            h_tilt = project_vector(rec_hidden, rec_hidden - h_hat, self.device)  # [batch_size, dec_dim]
            # h_hat: domain-shared text feature representations after GRL
            # h_tilt: domain-specific text feature representations
            if self.hp.dec_hidden_type == 1:
                return cls_preds, h_hat
            if self.hp.dec_hidden_type == 2:
                return cls_preds, h_tilt
        
        return cls_preds, rec_hidden
    
    def decode(self, rec_hidden, trg, revs=False, tf_ratio=None):
        # rec_hidden: [batch_size, hid_dim]
        # trg: [seq_len,batch_size]
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.output_dim
        
        # Decoding
        context = rec_hidden # [batch_size, hid_dim]
        hidden = context
        dec_input = trg[0,:] # First input token is the <sos> token
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        for t in range(1, trg_len):
            #insert input token embedding, previous hidden state and the context state
            dec_emb_input = self.embedding_rec(dec_input.unsqueeze(0)) # [1, batch_size, emb_dim]
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(dec_emb_input, hidden.unsqueeze(0), context.unsqueeze(0))
            prob = logits_to_prob(output, method="softmax", tau=1.0, eps=1e-10, gumbel_hard=False)
            #place predictions in a tensor holding predictions for each token
            outputs[t] = prob
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < tf_ratio if tf_ratio else False
            #get the highest predicted token from our predictions
            top1 = prob.argmax(1)
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            dec_input = trg[t] if teacher_force else top1
        
        reviews = None
        if revs:
            rev_hyps = outputs.permute(1, 2, 0).argmax(1)  # [batch_size, seq_len]
            rev_hyp_words = [self.vocab.outputids2words(hyp) for i, hyp in enumerate(rev_hyps)]
            reviews = [postprocess(words, skip_special_tokens=True, clean_up_tokenization_spaces=True) for words in rev_hyp_words]
        
        return outputs, reviews
        
    def forward(self, src_input, trg, src_senti, alpha=0.5, tf_ratio=0.5):
        # src_input: [seq_len,batch_size]
        # trg: [seq_len,batch_size]
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.output_dim
        ######## Encoding
        rec_hidden, cls_hidden = None, None
        if self.hp.use_rec:
            emb_input_rec = self.embedding_rec(src_input) # [seq_len, batch_size, emb_dim]
            rec_hidden = self.encoder_rec(emb_input_rec, self.device) # [batch_size, hid_dim]
        if self.hp.use_cls:
            emb_input_cls = self.embedding_cls(src_input) # [seq_len, batch_size, emb_dim]
            cls_hidden = self.encoder_cls(emb_input_cls, self.device) # [batch_size, hid_dim]
        ######## Classification
        if self.hp.use_cls:
            cls_preds, rec_hidden = self.classify(src_senti, cls_hidden, rec_hidden, alpha)
            # For analysis
            cls_preds_tmp = torch.clone(cls_preds).detach()
        else:
            cls_preds = None
            cls_preds_tmp = None
        
        ######## Decoding
        if self.hp.use_rec:
            outputs, _ = self.decode(rec_hidden, trg, tf_ratio=tf_ratio)
        else:
            outputs = None
    
        return outputs, cls_preds, cls_preds_tmp
    
    def inference(self, src_input, trg):
        # src_input: [seq_len,batch_size]
        # trg: [seq_len,batch_size]
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim 
        
        # Embeddings
        
        
        # Encoding
        if self.hp.use_rec:
            emb_input_rec = self.embedding_rec(src_input) # [seq_len, batch_size, emb_dim]
            rec_hidden = self.encoder_rec(emb_input_rec, self.device) # [batch_size, hid_dim]
        else:
            raise Error("Reconstruction disabled")
            
        if self.hp.use_cls:
            emb_input_cls = self.embedding_cls(src_input) # [seq_len, batch_size, emb_dim]
            cls_hidden = self.encoder_cls(emb_input_cls, self.device) # [batch_size, hid_dim]
        else:
            cls_hidden = None
        
        # Projection Mechanism
        if self.hp.use_proj:
            h_hat = project_vector(rec_hidden, cls_hidden, self.device)  # [batch_size, dec_dim]
            h_tilt = project_vector(rec_hidden, rec_hidden - h_hat, self.device)  # [batch_size, dec_dim]
            # h_hat: domain-shared text feature representations after GRL
            # h_tilt: domain-specific text feature representations
            if self.hp.dec_hidden_type == 1:
                context = h_hat
            elif self.hp.dec_hidden_type == 2:
                context = h_tilt
            else:
                context = rec_hidden
        else:
            context = rec_hidden
        
        # Decoding
        _, reviews = self.decode(context, trg, revs=True, tf_ratio=None)
        
        return reviews
        

def sample_target_docs(domain, hidden, batch_size, pool="same", concat=False):
    out_hidden = []
    for idx in range(batch_size):
        if pool == "opposite":
            h = hidden[domain != domain[idx]] # [num target docs in batch, dec_dim]
        elif pool == "same":
            h = hidden[domain == domain[idx]] # [num source docs in batch, dec_dim]
            h = h[[i != idx for i, _ in enumerate(h)]]  # [num source docs in batch - 1, dec_dim]
        else:
            raise Error(f"domain: '{pool}' is not recognized. expected: all, same, or opposite.")
        
        if concat:
            h = torch.mean(h, dim=0).unsqueeze(0)
        else:
            r = random.randint(0, h.size(0) - 1)
            h = h[r].unsqueeze(0)
        
        out_hidden.append(h)
    
    out_hidden = torch.vstack(out_hidden)
    return out_hidden