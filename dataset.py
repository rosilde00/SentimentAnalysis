from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pandas as pd

class SentimentDataset(Dataset):
    def __init__(self, sentences, status, tokenizer):
        self.sentences = sentences
        self.status = status
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.status)
    
    def __getitem__(self, index):
        label = self.status[index]
        tokenized_sentence = self.tokenizer(self.sentences[index], max_length=500, truncation=True, padding='max_length', return_tensors="pt")
        return tokenized_sentence, label
    
    
def get_dataloader(data, tokenizer, batch_size):
    
    sentences, label, sentiment= preproc_data(data)
    
    dataset = SentimentDataset(sentences, label, tokenizer)
    
    splitted_dataset = random_split(dataset, [0.7, 0.3])
    train_dataloader = DataLoader(splitted_dataset[0], batch_size=batch_size)
    validation_dataloader = DataLoader(splitted_dataset[1], batch_size=batch_size)
    
    return train_dataloader, validation_dataloader, sentiment

def preproc_data(data):
    sentences = list(map(lambda st: str(st).replace('"', ''), list(data['statement'].values)))
    label, _ = pd.factorize(data['status'].values)
    sentiment = torch.tensor(list(set(label))).float()
    return sentences, label, sentiment