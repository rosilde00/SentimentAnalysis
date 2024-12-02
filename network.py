from torch import nn
import torch
from transformers import RobertaForSequenceClassification, RobertaConfig
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class SentimentNetwork(nn.Module):
    def __init__(self, labels):
        super().__init__()
        self.labels = labels
        config = RobertaConfig()
        self.roberta = RobertaForSequenceClassification(config).from_pretrained('distilroberta-base', num_labels=7, problem_type="multi_label_classification")
        
    def forward(self, input_ids, mask):
        roberta_output = self.roberta(input_ids, attention_mask = mask)
        return roberta_output.logits
    
def create_model(labels):
    return SentimentNetwork(labels)

def train_loop(dataloader, model, loss_fn, optimizer, batch_size, device):
    size = len(dataloader.dataset) 
    model.train() 
    final_crossentropy = 0
    
    for batch, (x, y) in enumerate(dataloader): 
        y = y.to(device)
        pred = model(x['input_ids'].squeeze().to(device), x['attention_mask'].squeeze().to(device))
        loss = loss_fn(pred.squeeze(), y.long())
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        final_crossentropy += loss
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(y)
            print(f"MSE: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    final_crossentropy /= len(dataloader)
    print(f"In questa epoca, la cross entropy media è {final_crossentropy}")
    
def validation_loop(dataloader, model, loss_fn, device):
    model.eval()
    num_batches = len(dataloader)
    avg_cross, avg_acc = 0, 0
    label = np.array([])
    prediction = None

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x['input_ids'].squeeze(), x['attention_mask'].squeeze())
            
            y_np, pred_np = y.cpu().detach().numpy(), pred.cpu().detach().numpy()
            label = np.append(label, y_np, axis=0)
            if prediction is None:
                prediction = pred_np
            else:
                prediction = np.append(prediction, pred_np, axis=0)
            
            avg_cross += loss_fn(pred.squeeze(), y.long()).item()

    avg_cross /= num_batches

    encoder = OneHotEncoder(sparse=False) 
    label = encoder.fit_transform(label.reshape(-1, 1))
    prediction = onehot(prediction)
    avg_acc = accuracy_score(label, prediction)
    
    return avg_cross, avg_acc

def onehot(pred):
    argmax = np.argmax(pred, axis=1)
    onehot_pred = None
    for a in argmax:
        onehot_p = np.zeros(7)
        onehot_p[a] = 1
        if onehot_pred is None:
            onehot_pred = np.array([onehot_p])
        else:
            onehot_pred = np.append(onehot_pred, np.array([onehot_p]),  axis=0)  
    return onehot_pred

class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.count = 0
        self.early_stop = False

    def __call__(self, validation_loss):
        score = -validation_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.count += 1
            if self.count >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.count = 0
        return self.early_stop