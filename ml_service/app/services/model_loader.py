import os
import joblib
import pickle
import torch
import warnings
import tensorflow as tf
from transformers import BertTokenizerFast, BertForSequenceClassification
from app.config import settings

warnings.filterwarnings("ignore")

class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.device = torch.device("xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.models_loaded = {
            "logistic_regression": False,
            "lstm": False,
            "bert": False
        }
        
        # LR specific
        self.lr_model = None
        self.lr_vectorizer = None
        
        # LSTM specific
        self.lstm_model = None
        self.lstm_fast_predict = None
        self.lstm_tokenizer = None
        self.lstm_config = None
        
        # BERT specific
        self.bert_model = None
        self.bert_tokenizer = None
        self.bert_threshold = 0.5

        self._initialized = True

    def load_all_models(self):
        self.load_lr()
        self.load_lstm()
        self.load_bert()
        self.warmup_models()
        
    def warmup_models(self):
        # Warmup to absorb initialization costs (like TF graph building)
        print("Warming up models...")
        try:
            if self.models_loaded["logistic_regression"]:
                self.lr_model.predict(self.lr_vectorizer.transform(["warmup"]))
        except Exception as e: print(f"LR warmup failed: {e}")
            
        try:
            if self.models_loaded["lstm"]:
                seq = self.lstm_tokenizer.texts_to_sequences(["warmup text"])
                # Note: Keras pad_sequences requires absolute import here to avoid circular dependencies
                from keras.preprocessing.sequence import pad_sequences
                seq = pad_sequences(seq, maxlen=self.lstm_config.get("max_len", 300))
                self.lstm_fast_predict(tf.convert_to_tensor(seq))
        except Exception as e: print(f"LSTM warmup failed: {e}")
            
        try:
            if self.models_loaded["bert"]:
                inputs = self.bert_tokenizer("warmup text", return_tensors="pt", truncation=True, padding=True, max_length=128)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    self.bert_model(**inputs)
        except Exception as e: print(f"BERT warmup failed: {e}")
        print("Warmup complete.")

    def load_lr(self):
        try:
            vec_path = os.path.join(settings.MODELS_ML_DIR, "tfidf_vectorizer.pkl")
            model_path = os.path.join(settings.MODELS_ML_DIR, "sentiment_model.pkl")
            if os.path.exists(vec_path) and os.path.exists(model_path):
                self.lr_vectorizer = joblib.load(vec_path)
                self.lr_model = joblib.load(model_path)
                if not hasattr(self.lr_model, 'multi_class'):
                    self.lr_model.multi_class = 'ovr'
                self.models_loaded["logistic_regression"] = True
        except Exception as e:
            print(f"Failed to load Logistic Regression: {e}")

    def load_lstm(self):
        try:
            model_path = os.path.join(settings.MODELS_LSTM_DIR, "best_lstm_model.keras")
            tok_path = os.path.join(settings.DATA_DIR, "tokenizer.pkl")
            cfg_path = os.path.join(settings.DATA_DIR, "config.pkl")
            
            if os.path.exists(model_path) and os.path.exists(tok_path) and os.path.exists(cfg_path):
                self.lstm_model = tf.keras.models.load_model(model_path)
                
                @tf.function(reduce_retracing=True)
                def fast_predict(x):
                    return self.lstm_model(x, training=False)
                self.lstm_fast_predict = fast_predict
                
                with open(tok_path, "rb") as f:
                    self.lstm_tokenizer = pickle.load(f)
                with open(cfg_path, "rb") as f:
                    self.lstm_config = pickle.load(f)
                self.models_loaded["lstm"] = True
        except Exception as e:
            print(f"Failed to load LSTM: {e}")

    def load_bert(self):
        try:
            if os.path.exists(settings.MODELS_BERT_DIR):
                self.bert_tokenizer = BertTokenizerFast.from_pretrained(settings.MODELS_BERT_DIR, local_files_only=True)
                self.bert_model = BertForSequenceClassification.from_pretrained(settings.MODELS_BERT_DIR, local_files_only=True)
                self.bert_model.to(self.device)
                self.bert_model.eval()
                
                thresh_path = os.path.join(settings.MODELS_BERT_DIR, "threshold.pkl")
                if os.path.exists(thresh_path):
                    with open(thresh_path, "rb") as f:
                        self.bert_threshold = pickle.load(f).get("threshold", 0.5)
                        
                self.models_loaded["bert"] = True
        except Exception as e:
            print(f"Failed to load BERT: {e}")

model_manager = ModelManager()
