"""
Transductive Unlabeled TF-IDF Vectorizer Module

Fits TF-IDF vectorizers on the UNION of unlabeled raw text across
train and test sets (Source 1, Source 2, Source 3) for names and addresses.
Mitigates out-of-vocabulary and country drift (e.g. France in test set)
without label leakage.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from config import TFIDF_CONFIG

class TransductiveTFIDF:
    def __init__(self, max_features=None, token_pattern=None):
        self.max_features = max_features or TFIDF_CONFIG['max_features']
        self.token_pattern = token_pattern or TFIDF_CONFIG['token_pattern']
        self.name_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            token_pattern=self.token_pattern,
            lowercase=False
        )
        self.addr_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            token_pattern=self.token_pattern,
            lowercase=False
        )
        self.is_fitted = False
        
    def fit_from_text_iterables(self, name_iter, addr_iter):
        """
        Fits vectorizers using pure unlabeled text streams.
        """
        names = [str(n) for n in name_iter if n and str(n).strip()]
        addrs = [str(a) for a in addr_iter if a and str(a).strip()]
        
        self.name_vectorizer.fit(names if names else [''])
        self.addr_vectorizer.fit(addrs if addrs else [''])
        self.is_fitted = True
        return self
        
    def get_models_dict(self):
        """
        Returns dictionary matching features.py expected interface.
        """
        if not self.is_fitted:
            raise RuntimeError("TransductiveTFIDF must be fitted before obtaining models dict.")
        return {
            'name_tfidf': self.name_vectorizer,
            'address_tfidf': self.addr_vectorizer
        }
