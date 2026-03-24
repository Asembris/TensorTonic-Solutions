import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        self.word_to_id[self.pad_token]=0
        self.id_to_word[0]=self.pad_token
        self.word_to_id[self.unk_token]=1
        self.id_to_word[1]=self.unk_token
        self.word_to_id[self.bos_token]=2
        self.id_to_word[2]=self.bos_token
        self.word_to_id[self.eos_token]=3
        self.id_to_word[3]=self.eos_token
        cur=4
        for sent in texts:
            for e in sent.split():
                if e not in self.word_to_id:
                    self.vocab_size+=1
                    self.word_to_id[e]=cur
                    self.id_to_word[cur]=e
                    cur+=1
        self.vocab_size = len(self.word_to_id)
        pass
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        res=[self.word_to_id.get(e,self.word_to_id[self.unk_token]) for e in text.split()]
        return res
        pass
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        res=[]
        for e in ids:
            res.append(self.id_to_word.get(e,self.unk_token))
        return " ".join(res)
        pass
