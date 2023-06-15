import torch
import torch.nn as nn

from model.attention import Attention, GAT
from model.encoder import Encoder

from utils.model_utils import cosine_similarity, load_gloveembeddings, project_vector, cosine_similarity, postprocess, is_special, clean_up_tokenization, mask_range, logits_to_prob

import random
import copy
from collections import OrderedDict

from model.revgrad import GradReversal
from model.beam_decoder import BeamDecoder


class Summarizer(nn.Module):
    def __init__(self, encoder_rec, encoder_cls, decoder, classifier, embedding_rec, embedding_cls, grl, vocab, hp, n_docs=8):
        super().__init__()
        self.vocab = vocab
        self.hp = hp
        self.input_dim = len(vocab)
        self.output_dim = len(vocab)
        self.device = self.hp.device
        
        # Encoder
        self.gat_rec = GAT(device=self.device,
                           input_dim=self.hp.hid_dim * 2,
                           output_dim=self.hp.hid_dim * 2,
                           hidden_dim=self.hp.gat_hidden,
                           dropout=self.hp.gat_drop,
                           alpha=self.hp.gat_alpha,
                           num_heads=self.hp.gat_heads)
        self.encoder = Encoder(input_dim=self.input_dim,
                                   hid_dim=self.hp.hid_dim,
                                   emb_dim=self.hp.enc_emb_dim,
                                   gat_attn=self.gat_rec,
                                   num_layers=self.hp.enc_layers,
                                   dropout=self.hp.enc_drop)
        
        self.encoder_rec = encoder_rec
        self.encoder_cls = encoder_cls
        self.decoder = decoder
        self.classifier = classifier
        self.grl = grl
        self.embedding_rec = embedding_rec
        self.embedding_cls = embedding_cls
        self.num_tgr_docs_domains = self.hp.num_tgr_docs_domains
        
        # used to reduce the dimension of the classifier input
        if self.num_tgr_docs_domains > 1:
            self.fc = nn.Linear(self.hp.dec_hidden * self.num_tgr_docs_domains, self.hp.dec_hidden)
        
        if self.hp.combine_encs == "ff":
            self.combine_encs_net = nn.Sequential(OrderedDict([
                ('ln1', nn.LayerNorm(n_docs * self.hp.hid_dim)),
                ('fc1', nn.Linear(n_docs * self.hp.hid_dim, self.hp.hid_dim)),
                ('relu1', nn.ReLU()),
                ('ln2', nn.LayerNorm(self.hp.hid_dim)),
                ('fc2', nn.Linear(self.hp.hid_dim, self.hp.hid_dim))
            ]))
            
        ########### Initializing beam decoder ###########
        self.beam_size = self.hp.beam_size
        self.min_dec_steps = self.hp.min_dec_steps
        self.num_return_seq = self.hp.num_return_seq
        self.num_return_sum = self.hp.num_return_sum
        self.n_gram_block = self.hp.n_gram_block
        self.beam_decoder = BeamDecoder(
            self.device, 
            self.vocab, 
            self.embedding_rec, 
            self.decoder, 
            self.hp.beam_size, 
            self.hp.min_dec_steps, 
            self.hp.num_return_seq, 
            self.hp.num_return_sum, 
            self.hp.n_gram_block
        )    

            
    def classify(self, cls_hidden, rec_hidden, alpha):
        # cls_hiddden: [batch_size, hid_dim]
        # rec_hidden: [batch_size, hid_dim]
        if not self.hp.use_cls:
            return None, rec_hidden, rec_hidden
        
        # Concat target domain docs representations
        if self.num_tgr_docs_domains == 1:
            hiddens = sample_target_docs(src_senti, cls_hidden, batch_size, pool="opposite", concat=self.hp.concat_docs) # [batch_size, hid_dim]
        elif self.num_tgr_docs_domains == 2:
            same_hiddens = sample_target_docs(src_senti, cls_hidden, batch_size, domain="same", concat=self.hp.concat_docs)
            opp_hiddens = sample_target_docs(src_senti, cls_hidden, batch_size, domain="opposite", concat=self.hp.concat_docs)
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
                return cls_preds, h_hat, h_hat
            if self.hp.dec_hidden_type == 2:
                return cls_preds, h_tilt, h_hat
        
        return cls_preds, rec_hidden, rec_hidden
    
    def decode_reviews(self, rec_hidden, trg, revs=False, tf_ratio=None):
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
        
        return outputs
    
    def decode_summaries(self, revs_hidden, hiddens, context, trg, src_len, gumbel_hard=True):
        vocab_size = self.output_dim
        # Set summary length to the mean review length in the batch
        avg_len = int(torch.ceil(torch.mean(src_len.float())))
        sum_len = min(max(avg_len *2, 20), 75)
        
        # Compute mean representation
        if self.hp.combine_encs == "ff":
            mean_hidden = hiddens.contiguous().view(-1) # [batch_size * hid_dim]
            mean_context = context.contiguous().view(-1) # [batch_size * hid_dim]
            
            mean_hidden = self.combine_encs_net(mean_hidden.unsqueeze(0)) # [1, hid_dim]
            mean_context = self.combine_encs_net(mean_context.unsqueeze(0)) # [1, hid_dim]
        elif self.hp.combine_encs == "mean":
            mean_hidden = torch.mean(hiddens, dim=0).unsqueeze(0) # [1, hid_dim]
            mean_context = torch.mean(context, dim=0).unsqueeze(0) # [1, hid_dim]
        else:
            raise Error(f"Concatenation method '{self.hp.combine_encs}' is not supported")
        
        sum_hidden = mean_hidden
        sum_context = mean_context
        
        sum_outputs = torch.zeros(sum_len, 1, vocab_size).to(self.device)
        sum_dec_input = trg[0, 0].unsqueeze(0)
        for t in range(1, sum_len):
            sum_dec_emb_input = self.embedding_rec(sum_dec_input.unsqueeze(0)) # [1, batch_size, emb_dim]
            #if gumbel_hard and t != 0:
            #    print(dec_input.shape, self.lm.embedding.weight.shape)
            #    input_emb = torch.matmul(dec_input, self.lm.embedding.weight)
            #    print(input_emb.shape)
            
            sum_output, sum_hidden = self.decoder(sum_dec_emb_input, sum_hidden.unsqueeze(0), sum_context.unsqueeze(0))
            prob = logits_to_prob(sum_output, method="gumbel", tau=1.0, eps=1e-10, gumbel_hard=True)
            sum_outputs[t] = prob
            top1 = prob.argmax(1)
            sum_dec_input = top1
        
        sum_ids = sum_outputs.permute(1, 2, 0).argmax(1) # [1, sum_len]
        sum_ids = sum_ids.permute(1, 0) # [sum_len, 1]
        
        # Encode summary
        emb_sum = self.embedding_rec(sum_ids) # [sum_len, 1, emb_dim]
        emb_sum.requires_grad = True
        sum_hidden = self.encoder(emb_sum, self.device) # [1, hid_dim]
        
        # Compute cosine similarity
        cos_sim = []
        for h in revs_hidden:
            score = cosine_similarity(sum_hidden.squeeze(0), h).unsqueeze(0)
            cos_sim.append(score.unsqueeze(0))
        cos_sim = torch.mean(torch.vstack(cos_sim), dim=0)  # [batch_size, 1]
        
        return sum_outputs, cos_sim
    
    def forward(self, src_input, trg, src_len, alpha=0.5, tf_ratio=0.5, gumbel_hard=True):
        # src_input: [seq_len, batch_size]
        # trg: [seq_len, batch_size]
        
        ######## Embeddings
        emb_input_rec = self.embedding_rec(src_input)
        emb_input_cls = self.embedding_cls(src_input)
        
        ######## Encoding
        revs_hidden = self.encoder_rec(emb_input_rec, self.device) # [batch_size, hid_dim]
        cls_hidden = None if not self.hp.use_cls else self.encoder_cls(emb_input_cls, self.device) # [batch_size, hid_dim]
        
        ######## Projection Mechanism
        if self.hp.use_proj:
            h_hat = project_vector(revs_hidden, cls_hidden, self.device)  # [batch_size, dec_dim]
            h_tilt = project_vector(revs_hidden, revs_hidden - h_hat, self.device)  # [batch_size, dec_dim]
            # h_hat: domain-shared text feature representations after GRL
            # h_tilt: domain-specific text feature representations
            
            def select_hidden(hidden_type):
                mapping = {0: revs_hidden, 1: h_hat, 2: h_tilt}
                return mapping[hidden_type]
            
            sum_hidden = select_hidden(self.hp.mean_hidden_type)
            ref_hidden = select_hidden(self.hp.ref_hidden_type)
            context = select_hidden(self.hp.mean_context_type)
        
        else:
            sum_hidden = revs_hidden
            ref_hidden = revs_hidden
            context = revs_hidden
        
        ######## Decoding
        revs_outputs = self.decode_reviews(revs_hidden, trg, tf_ratio=tf_ratio)
        sum_outputs, cos_sim = self.decode_summaries(ref_hidden, sum_hidden, context, trg, src_len, gumbel_hard)
    
        return revs_outputs, cos_sim
    
    def inference(self, src_input, src_len):
        # src_input: [seq_len,batch_size]
        
        ######## Embeddings
        emb_input_rec = self.embedding_rec(src_input)
        emb_input_cls = self.embedding_cls(src_input)
        
        ######## Encoding
        rec_hidden = self.encoder_rec(emb_input_rec, self.device) # [batch_size, hid_dim]
        cls_hidden = None if not self.hp.use_cls else self.encoder_cls(emb_input_cls, self.device) # [batch_size, hid_dim]
        
        # Projection Mechanism
        if self.hp.use_proj:
            h_hat = project_vector(rec_hidden, cls_hidden, self.device)  # [batch_size, dec_dim]
            h_tilt = project_vector(rec_hidden, rec_hidden - h_hat, self.device)  # [batch_size, dec_dim]
            # h_hat: domain-shared text feature representations after GRL
            # h_tilt: domain-specific text feature representations
            
            def select_hidden(hidden_type):
                mapping = {0: rec_hidden, 1: h_hat, 2: h_tilt}
                return mapping[hidden_type]
                
            hiddens = select_hidden(self.hp.gen_hidden_type)
            context = select_hidden(self.hp.mean_context_type)
        else:
            hiddens = rec_hidden
            context = rec_hidden
        
        # Compute mean representation
        if self.hp.combine_encs == "ff":
            mean_hidden = hiddens.contiguous().view(-1) # [batch_size * hid_dim]
            mean_context = context.contiguous().view(-1) # [batch_size * hid_dim]
            
            mean_hidden = self.combine_encs_net(mean_hidden.unsqueeze(0)) # [1, hid_dim]
            mean_context = self.combine_encs_net(mean_context.unsqueeze(0)) # [1, hid_dim]
        elif self.hp.combine_encs == "mean":
            mean_hidden = torch.mean(hiddens, dim=0).unsqueeze(0) # [1, hid_dim]
            mean_context = torch.mean(context, dim=0).unsqueeze(0) # [1, hid_dim]
        else:
            raise Error(f"Concatenation method '{self.hp.combine_encs}' is not supported")
        
        avg_len = int(torch.ceil(torch.median(src_len.float())))
        sum_len = 75 #min(max(avg_len * 2, 20), 75)
        vocab_size = self.decoder.output_dim
        
        if self.hp.beam_decode:
            summaries = self.beam_decoder.decode(hidden=mean_hidden, context=mean_context, gen_len=sum_len, batch_size=1)
        else:
            hidden = mean_hidden
            context = mean_context
            outputs = torch.zeros(sum_len, 1, vocab_size).to(self.device)
            dec_input = src_input[0, 0].unsqueeze(0)
        
            for t in range(1, sum_len):
                dec_emb_input = self.embedding_rec(dec_input.unsqueeze(0)) # [1, batch_size, emb_dim]
                output, hidden = self.decoder(dec_emb_input, hidden.unsqueeze(0), context.unsqueeze(0))
                prob = logits_to_prob(output, method="softmax", tau=1.0, eps=1e-10, gumbel_hard=False)
                outputs[t] = prob
                top1 = prob.argmax(1)
                dec_input = top1
        
            outputs = outputs.permute(1, 2, 0) # [batch_size, vocab_size, seq-len]
            rev_hyps = outputs.argmax(1)  # [batch_size, seq_len]
            rev_hyp_words = [self.vocab.outputids2words(hyp) for i, hyp in enumerate(rev_hyps)]
            summaries = [postprocess(words, skip_special_tokens=True, clean_up_tokenization_spaces=True) for words in rev_hyp_words]
        
        return summaries, hiddens, mean_hidden
