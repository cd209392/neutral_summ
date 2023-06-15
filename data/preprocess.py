import pandas as pd
import numpy as np

import re
import json
import string

from bs4 import BeautifulSoup

import unicodedata

import nltk
from nltk.corpus import stopwords

import spacy
from spacy.lang.en import English
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex

from configs.config import DatasetConfig

ds_config = DatasetConfig()

# Do not split tokens by "-". Consider composed words like "like-minded" as one token
# Credit: https://stackoverflow.com/questions/58105967/spacy-tokenization-of-hyphenated-words
custom_infixes = (
    LIST_ELLIPSES
    + LIST_ICONS
    + [
        r"(?<=[0-9])[+\-*^](?=[0-9-])",
        r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
            al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
        ),
        r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
        r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
    ]
)

custom_infix_re = compile_infix_regex(custom_infixes)

nlp = English()
nlp.tokenizer.infix_finditer = custom_infix_re.finditer

NER = spacy.load("en_core_web_sm")


DATA_DIR = ds_config.data_dir

# Based on the work of:
# https://towardsdatascience.com/nlp-building-text-cleanup-and-preprocessing-pipeline-eba4095245a0
# https://towardsdatascience.com/text-normalization-7ecc8e084e31

with open(f"{DATA_DIR}english_contractions.json", "r") as fd:
    CONTRACTION_MAP = json.loads(fd.read())


with open(f"{DATA_DIR}eng_abbrv.json", "r") as fd:
    ABBRV_MAP = json.loads(fd.read())


with open(f"{DATA_DIR}eng_misspelled_dict.json", "r") as fd:
    MISSPELLED_MAP = json.loads(fd.read())


class TextProcessing():
    def __init__(self, stopwords=True, punctuations=False, correct_spelling=False):
        self.stopwords = stopwords
        self.punctuations = punctuations
        self.correct_spelling = correct_spelling
        
    def preprocess(self, text):
        text = self.remove_markup_tags(text)
        text = self.to_lowercase(text)
        text = self.remove_urls(text)
        text = self.remove_emojis(text)
        text = self.normalize_punctuations(text)
        text = self.normalize_whitespaces(text)
        text = self.expand_contractions(text)
        text = self.expand_abbreviations(text)
        #text = self.replace_by_type(text)
        text = self.remove_special_chars(text)
        text = self.sep_punctuations(text)
        
        if self.correct_spelling:
            text = self.spelling_correction(text)
            
        if not self.stopwords:
            text = self.remove_stopwords(text)
            
        if not self.punctuations:
            text = self.remove_punctuations(text)
        
        text = re.sub(r'\s+|( )+', ' ', text).strip()
        
        return text
    
    def remove_markup_tags(self, text):
        """
        Remove HTML and XML tags
        """
        new_text = BeautifulSoup(text, 'html.parser').get_text()
        return new_text

    def to_lowercase(self, text):
        return text.lower()
    
    def remove_urls(self, text):
        """
        Remove URLs, emails, and phone numbers
        """
        text = str(text)
        url_re = r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+' +\
            r'[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]' +\
            r'\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}' +\
            r'|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
        email_re = r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
        phone_re = r"^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}$"
    
        text = re.sub(url_re, " ", text)
        text = re.sub(email_re, " ", text)
        text = re.sub(phone_re, " ", text)
    
        return text
    
    def remove_emojis(self, text):
        # Remove exagerated parenthesis
        text = re.sub(r"([\(\)])\1+", "\1", text)
    
        # Remove emoticons and emojis
        emoticon_re = r"\:\-?\)|\:\-?\(|\;\-?\)|\:\-?\>|\:\-?p|\:\-?\/|\:\-?\\|\:\-?1|\:\-?d|\%\*\}|\:\-?\*|\'\:\-?\)" +\
                  r"|\:\-\/|\:p|\:\/|\:o" +\
                  r"|\:\-?o|\:\'\-?\(|\:\-?\{\)|\:\-?\)\>|\%\-?\)|\&\:\-?\\|\(\:\-?\)|o\:\-?\)|\>\:\-?\>|8\=x|\(p\-?\||\:\-?\[|\<\:\+d"
        emojis_re = u"(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])"
        text = re.sub(emojis_re, " ", text)
        text = re.sub(emojis_re, " ", text)
        return text
    
    def normalize_punctuations(self, text):
        text = str(text)
        text = re.sub(r'([\!\?,;\:])\1+', r'\1', text)
        text = re.sub(r'\.{2,}', r'...', text)
        text = re.sub(r'\.{3}( )+', r'. ', text)
        text = re.sub(r'\.{3}', ', ', text)
        return text
    
    def normalize_whitespaces(self, text):
        text = str(text)
        text = re.sub(r"//t", r"\t", text)
        text = re.sub(r"( )+", r" ", text)
        text = re.sub(r"(\n)+", r" ", text)
        text = re.sub(r"(\r)+", r" ", text)
        text = re.sub(r"(\t)+", r" ", text)
        return text.strip(" ")

    def __expand_word_form(self, text, map_):
        pattern = re.compile(r'(^|\W+)({})(\W+|$)'.format('|'.join(map_.keys())), flags=re.IGNORECASE|re.DOTALL)
        def get_match(contraction):
            reg = r'({})'.format('|'.join([re.escape(k) for k in map_.keys()]))
            match = contraction.group(2)
            first_char = match[0]
            expanded = map_.get(match) if map_.get(match) else map_.get(match.lower())
            expanded = expanded  #first_char + expanded[1:]
            if expanded is None:
                return expanded
            return contraction.group(1) + expanded + contraction.group(3) 
        new_text = pattern.sub(get_match, text)
        new_text = re.sub("'", "", new_text)
        return new_text

    def expand_contractions(self, text):
        """
        won't -> will not
        they're -> they are
        """
        text = str(text)
        return self.__expand_word_form(text, map_=CONTRACTION_MAP)

    def expand_abbreviations(self, text):
        text = str(text)
        return self.__expand_word_form(text, map_=ABBRV_MAP)

    def __reduce_exaggerations(self, text):
        """
        Auxiliary function to help with exxagerated words.
        Examples:
            woooooords -> woords
            yaaaaaaaaaaaaaaay -> yaay
        """
        correction = str(text)
        return re.sub(r'([\w])\1{3,}', r'\1', correction)

    def is_numeric(self, text):
        if not re.search("[0-9,\%\.\$]", text):
            return False
        return True

    def spelling_correction(self, text):
        text = str(text)
        if len(text) < 1:
            return ""
        
        token_list = [str(tok) for tok in nlp.tokenizer(text)]
        for word_pos in range(len(token_list)):
            word = str(token_list[word_pos])
            if word is None:
                token_list[word_pos] = ""
                continue
            if word not in string.punctuation and not self.is_numeric(word):
                replacement = self.__reduce_exaggerations(word)
                replacement = self.__expand_word_form(text, map_=MISSPELLED_MAP)
                word = replacement
                token_list[word_pos] = word
        
        return " ".join(token_list).strip()

    def replace_by_type(self, text):
        DATE_RE = r"(^(19|20)\d\d[- /.](0[1-9]|1[012])[- /.](0[1-9]|[12][0-9]|3[01])$)" +\
              r"|(^(0[1-9]|1[012])[- /.](0[1-9]|[12][0-9]|3[01])[- /.](19|20)\d\d$)" +\
              r"|(^(0[1-9]|[12][0-9]|3[01])[- /.](0[1-9]|1[012])[- /.](19|20)\d\d$)"
    
        NUMERIC_TOKENS = ["DATE", "TIME", "MONEY", "QUANTITY"]
    
        # First, we remove bracket as they are used for type tokens repalcement
        text = re.sub(r"\[|\]", " ", text)
    
        # We replace certain numeric tokens with their type
        doc = NER(text)
        for tok in doc.ents:
            #print(tok.text, tok.label_)
            tag = str(tok.label_)
            if re.search(DATE_RE, tok.text):
                tag = "DATE"
            
            if tag in NUMERIC_TOKENS:
                if tag == "DATE" and re.search(r"years?( |\s|-)+old|y/o|yrs?( |\s|-)+old", tok.text):
                    tag = "AGE"
                text = re.sub(re.escape(tok.text), f" {tag} ", text)
        return text
    
    def remove_special_chars(self, text):
        text = str(text)
        
        # Replace parenthesis with spaces
        new_text = re.sub(r"\(|\)", r" ", text)
        
        # Normalize dashes
        new_text = re.sub("\–", "-", new_text)
        
        # Normalise '
        new_text = re.sub("\`", "'", new_text)
            
        # Turn "--" and underscores (_) to spaces
        new_text = re.sub(r"(\-|\–){2,}|_+", r" ", new_text)
            
        # handle forward slash
        #slash_pattern = re.compile(r"([a-zA-Z\'])(\/)([a-zA-Z\'])", flags=re.IGNORECASE|re.DOTALL)
        #new_text = slash_pattern.sub(
        #    lambda contraction: contraction.group(1).strip() + " or " +  contraction.group(3).strip(), new_text)
        
        # define the pattern to keep
        exclude_re = r'[^a-zA-Z0-9\.,!\?\/:;\'\-\–\[\] ]+'
        new_text = re.sub(exclude_re, ' ', new_text)
        
        # remove hanging dash
        new_text = re.sub(r"( )+\-( )+", " ", new_text)
        new_text = re.sub(r"(^| +|\s+|$)[\/\-\–\\]+|[\/\-\–\\]+(^| +|\s+|$)", " ", new_text)
        
        return new_text
    
    def sep_punctuations(self, text):
        text = " ".join([str(w) for w in nlp.tokenizer(text)])
        return text
    
    def remove_stopwords(self, text):
        text = [str(w) for w in nlp.tokenizer(text) if str(w) not in stopwords.words('english')]
        text = " ".join(text)
        # nlp.tokenizer adds space around punctuations and certain special characters
        # We remove the space around bracket as they are part of special tokens such as [MONEY]
        text = re.sub("\[ +", "[", text)
        text = re.sub("\] +", "]", text)
        return text
    
    def remove_punctuations(self, text):
        PUNCTUATIONS = r"[\.;,\:\?\!\^\(\)]( +|\n+|\s+|\r+|$)"
        text = ''.join([c for c in text if not re.search(PUNCTUATIONS, c)])
        return text
