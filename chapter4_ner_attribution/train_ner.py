import os, gc, json, pickle
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

class Config:
    DEVICE = torch.device('cuda')
    BATCH_SIZE = 8
    EPOCHS = 5
    LR = 2e-5
    MAX_LEN = 128
    DIR = Path("/home/macierz/mohabdal/TerrorismNER_Project")

class NERModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(out.last_hidden_state)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {'loss': loss, 'logits': logits}

class NERDataset(Dataset):
    def __init__(self, data, tokenizer, label2id):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer(item['tokens'], is_split_into_words=True, truncation=True, max_length=Config.MAX_LEN, padding='max_length', return_tensors='pt')
        word_ids = enc.word_ids()
        labels = []
        for w in word_ids:
            if w is None or w >= len(item['labels']):
                labels.append(-100)
            else:
                labels.append(self.label2id.get(item['labels'][w], 0))
        return {'input_ids': enc['input_ids'].squeeze(), 'attention_mask': enc['attention_mask'].squeeze(), 'labels': torch.tensor(labels)}

def calc_f1(preds, trues):
    tp = sum(1 for p, t in zip(preds, trues) if p == t and p != 0)
    fp = sum(1 for p, t in zip(preds, trues) if p != 0 and p != t)
    fn = sum(1 for p, t in zip(preds, trues) if t != 0 and p == 0)
    p = tp/(tp+fp) if (tp+fp) > 0 else 0
    r = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
    return {'precision': p, 'recall': r, 'f1': f1}

def main():
    print("TerrorismNER Training")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    with open(Config.DIR / "checkpoints/ner_data_basic.pkl", 'rb') as f:
        data = pickle.load(f)
    print(f"Train: {len(data['train'])}, Val: {len(data['val'])}, Test: {len(data['test'])}")
    
    label2id = {'O': 0, 'B-TERROR_GROUP': 1, 'I-TERROR_GROUP': 2}
    models_dict = {'distilbert': 'distilbert-base-uncased', 'bert': 'bert-base-uncased', 'roberta': 'roberta-base'}
    results = {}
    
    for name, path in models_dict.items():
        print(f"\nTraining: {name}")
        gc.collect()
        torch.cuda.empty_cache()
        
        tok = AutoTokenizer.from_pretrained(path)
        if 'roberta' in path:
            tok = AutoTokenizer.from_pretrained(path, add_prefix_space=True)
        
        train_dl = DataLoader(NERDataset(data['train'], tok, label2id), batch_size=Config.BATCH_SIZE, shuffle=True)
        val_dl = DataLoader(NERDataset(data['val'], tok, label2id), batch_size=Config.BATCH_SIZE)
        test_dl = DataLoader(NERDataset(data['test'], tok, label2id), batch_size=Config.BATCH_SIZE)
        
        model = NERModel(path, 3).to(Config.DEVICE)
        opt = AdamW(model.parameters(), lr=Config.LR)
        total_steps = len(train_dl) * Config.EPOCHS
        sch = get_linear_schedule_with_warmup(opt, total_steps//10, total_steps)
        
        best_f1 = 0
        for ep in range(Config.EPOCHS):
            model.train()
            loss_sum = 0
            for b in tqdm(train_dl, desc=f"Epoch {ep+1}"):
                opt.zero_grad()
                out = model(b['input_ids'].to(Config.DEVICE), b['attention_mask'].to(Config.DEVICE), b['labels'].to(Config.DEVICE))
                out['loss'].backward()
                opt.step()
                sch.step()
                loss_sum += out['loss'].item()
            
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for b in val_dl:
                    out = model(b['input_ids'].to(Config.DEVICE), b['attention_mask'].to(Config.DEVICE))
                    pred = out['logits'].argmax(-1).cpu()
                    for p, l, m in zip(pred, b['labels'], b['attention_mask']):
                        for pi, li, mi in zip(p.tolist(), l.tolist(), m.tolist()):
                            if mi == 1 and li != -100:
                                preds.append(pi)
                                trues.append(li)
            
            metrics = calc_f1(preds, trues)
            print(f"Epoch {ep+1}: Loss={loss_sum/len(train_dl):.4f}, F1={metrics['f1']:.4f}")
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                torch.save(model.state_dict(), Config.DIR / f"checkpoints/{name}_best.pt")
        
        model.load_state_dict(torch.load(Config.DIR / f"checkpoints/{name}_best.pt"))
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for b in test_dl:
                out = model(b['input_ids'].to(Config.DEVICE), b['attention_mask'].to(Config.DEVICE))
                pred = out['logits'].argmax(-1).cpu()
                for p, l, m in zip(pred, b['labels'], b['attention_mask']):
                    for pi, li, mi in zip(p.tolist(), l.tolist(), m.tolist()):
                        if mi == 1 and li != -100:
                            preds.append(pi)
                            trues.append(li)
        
        metrics = calc_f1(preds, trues)
        results[name] = metrics
        print(f"Test: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    with open(Config.DIR / "results/training_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nFINAL RESULTS:")
    for name, m in results.items():
        print(f"{name}: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}")

if __name__ == "__main__":
    main()
