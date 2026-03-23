#!/usr/bin/env python3
import os,gc,json,pickle,random,time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.optim import AdamW
from transformers import AutoModel,AutoTokenizer,RobertaModel,RobertaTokenizerFast,get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score,accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class Config:
    BASE_DIR=Path("/home/macierz/mohabdal/TerrorismNER_Research")
    CHECKPOINT_DIR=BASE_DIR/"checkpoints"
    RESULTS_DIR=BASE_DIR/"results"
    FIGURES_DIR=BASE_DIR/"figures"
    MODELS_DIR=BASE_DIR/"models"
    DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED=42;MAX_LEN=256;BATCH_SIZE=8;EPOCHS=5;LR=2e-5
    TRANSFORMER_MODELS={'distilbert':'distilbert-base-uncased','bert':'bert-base-uncased','roberta':'roberta-base'}
    GROUP_CONFIGS=[10,20,50]

def set_seed(seed=42):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
set_seed(42)

def load_classification_data(n_groups):
    df=pickle.load(open(Config.CHECKPOINT_DIR/f"classification_data_{n_groups}.pkl",'rb'))
    print(f"✅ Loaded {n_groups} groups: {len(df)} samples")
    le=LabelEncoder()
    df['label']=le.fit_transform(df['canonical_group'])
    label2id={name:i for i,name in enumerate(le.classes_)}
    id2label={i:name for i,name in enumerate(le.classes_)}
    train_df,temp_df=train_test_split(df,test_size=0.3,random_state=42,stratify=df['label'])
    val_df,test_df=train_test_split(temp_df,test_size=0.5,random_state=42,stratify=temp_df['label'])
    def df_to_list(d):return [{'text':row['clean_text'],'label':row['label']} for _,row in d.iterrows()]
    data={'train':df_to_list(train_df),'val':df_to_list(val_df),'test':df_to_list(test_df),'label2id':label2id,'id2label':id2label}
    print(f"   Train={len(data['train'])}, Val={len(data['val'])}, Test={len(data['test'])}")
    return data

class TraditionalClassifier:
    def __init__(self):
        self.vectorizer=TfidfVectorizer(max_features=10000,ngram_range=(1,2))
        self.classifier=LogisticRegression(max_iter=1000,class_weight='balanced',n_jobs=-1)
    def train(self,texts,labels):
        X=self.vectorizer.fit_transform(texts);self.classifier.fit(X,labels)
    def predict(self,texts):return self.classifier.predict(self.vectorizer.transform(texts))
    def evaluate(self,texts,labels):
        preds=self.predict(texts)
        return {'accuracy':accuracy_score(labels,preds),'f1_macro':f1_score(labels,preds,average='macro'),'f1_weighted':f1_score(labels,preds,average='weighted'),'precision':precision_score(labels,preds,average='macro'),'recall':recall_score(labels,preds,average='macro')}

def train_traditional(data,n_groups):
    print(f"\n{'='*60}\n🔬 TF-IDF + LogisticRegression ({n_groups} groups)\n{'='*60}")
    start=time.time()
    train_texts=[item['text'] for item in data['train']];train_labels=[item['label'] for item in data['train']]
    test_texts=[item['text'] for item in data['test']];test_labels=[item['label'] for item in data['test']]
    clf=TraditionalClassifier();clf.train(train_texts,train_labels);metrics=clf.evaluate(test_texts,test_labels)
    results={'model':'TF-IDF+LR','n_groups':n_groups,'accuracy':round(metrics['accuracy'],4),'f1_macro':round(metrics['f1_macro'],4),'f1_weighted':round(metrics['f1_weighted'],4),'precision':round(metrics['precision'],4),'recall':round(metrics['recall'],4),'training_time':round(time.time()-start,2)}
    print(f"✅ Acc={results['accuracy']:.4f}, F1={results['f1_macro']:.4f}")
    return results,confusion_matrix(test_labels,clf.predict(test_texts))

class ClassificationDataset(Dataset):
    def __init__(self,texts,labels,tokenizer,max_len=256):
        self.texts=texts;self.labels=labels;self.tokenizer=tokenizer;self.max_len=max_len
    def __len__(self):return len(self.texts)
    def __getitem__(self,idx):
        enc=self.tokenizer(str(self.texts[idx]),truncation=True,max_length=self.max_len,padding='max_length',return_tensors='pt')
        return {'input_ids':enc['input_ids'].squeeze(),'attention_mask':enc['attention_mask'].squeeze(),'label':torch.tensor(self.labels[idx],dtype=torch.long)}

class TransformerClassifier(nn.Module):
    def __init__(self,model_name,num_classes,dropout=0.1):
        super().__init__()
        self.encoder=AutoModel.from_pretrained(model_name);self.dropout=nn.Dropout(dropout)
        self.classifier=nn.Linear(self.encoder.config.hidden_size,num_classes)
    def forward(self,input_ids,attention_mask):
        out=self.encoder(input_ids=input_ids,attention_mask=attention_mask)
        return self.classifier(self.dropout(out.last_hidden_state[:,0,:]))

class FocalLoss(nn.Module):
    def __init__(self,alpha=None,gamma=2.0):
        super().__init__();self.alpha=alpha;self.gamma=gamma
    def forward(self,inputs,targets):
        ce=nn.functional.cross_entropy(inputs,targets,reduction='none');pt=torch.exp(-ce)
        loss=((1-pt)**self.gamma)*ce
        if self.alpha is not None:loss=self.alpha[targets]*loss
        return loss.mean()

def compute_class_weights(labels):
    counts=np.bincount(labels);return torch.FloatTensor(len(labels)/(len(counts)*counts+1e-6))

def train_transformer(model_name,model_path,data,n_groups,epochs=5):
    print(f"\n{'='*60}\n🚀 {model_name} ({n_groups} groups)\n{'='*60}")
    gc.collect();torch.cuda.empty_cache()
    tokenizer=AutoTokenizer.from_pretrained(model_path)
    if 'roberta' in model_path:tokenizer=AutoTokenizer.from_pretrained(model_path,add_prefix_space=True)
    train_texts=[item['text'] for item in data['train']];train_labels=[item['label'] for item in data['train']]
    val_texts=[item['text'] for item in data['val']];val_labels=[item['label'] for item in data['val']]
    test_texts=[item['text'] for item in data['test']];test_labels=[item['label'] for item in data['test']]
    train_loader=DataLoader(ClassificationDataset(train_texts,train_labels,tokenizer,Config.MAX_LEN),batch_size=Config.BATCH_SIZE,shuffle=True)
    val_loader=DataLoader(ClassificationDataset(val_texts,val_labels,tokenizer,Config.MAX_LEN),batch_size=Config.BATCH_SIZE)
    test_loader=DataLoader(ClassificationDataset(test_texts,test_labels,tokenizer,Config.MAX_LEN),batch_size=Config.BATCH_SIZE)
    model=TransformerClassifier(model_path,len(data['label2id'])).to(Config.DEVICE)
    criterion=FocalLoss(alpha=compute_class_weights(train_labels).to(Config.DEVICE),gamma=2.0)
    optimizer=AdamW(model.parameters(),lr=Config.LR)
    scheduler=get_linear_schedule_with_warmup(optimizer,len(train_loader)*epochs//10,len(train_loader)*epochs)
    best_f1=0;best_state=None;start=time.time()
    for epoch in range(epochs):
        model.train();total_loss=0
        for batch in tqdm(train_loader,desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            logits=model(batch['input_ids'].to(Config.DEVICE),batch['attention_mask'].to(Config.DEVICE))
            loss=criterion(logits,batch['label'].to(Config.DEVICE));loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();scheduler.step();total_loss+=loss.item()
        model.eval();val_preds,val_true=[],[]
        with torch.no_grad():
            for batch in val_loader:
                logits=model(batch['input_ids'].to(Config.DEVICE),batch['attention_mask'].to(Config.DEVICE))
                val_preds.extend(logits.argmax(dim=-1).cpu().numpy());val_true.extend(batch['label'].numpy())
        val_f1=f1_score(val_true,val_preds,average='macro')
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Val F1={val_f1:.4f}")
        if val_f1>best_f1:best_f1=val_f1;best_state=model.state_dict().copy()
    model.load_state_dict(best_state);model.eval();test_preds,test_true=[],[]
    with torch.no_grad():
        for batch in test_loader:
            logits=model(batch['input_ids'].to(Config.DEVICE),batch['attention_mask'].to(Config.DEVICE))
            test_preds.extend(logits.argmax(dim=-1).cpu().numpy());test_true.extend(batch['label'].numpy())
    results={'model':model_name,'n_groups':n_groups,'accuracy':round(accuracy_score(test_true,test_preds),4),'f1_macro':round(f1_score(test_true,test_preds,average='macro'),4),'f1_weighted':round(f1_score(test_true,test_preds,average='weighted'),4),'precision':round(precision_score(test_true,test_preds,average='macro'),4),'recall':round(recall_score(test_true,test_preds,average='macro'),4),'training_time':round(time.time()-start,2)}
    print(f"✅ Acc={results['accuracy']:.4f}, F1={results['f1_macro']:.4f}")
    Config.MODELS_DIR.mkdir(exist_ok=True);torch.save(best_state,Config.MODELS_DIR/f"{model_name}_cls_{n_groups}.pt")
    del model;gc.collect();torch.cuda.empty_cache()
    return results,confusion_matrix(test_true,test_preds)

class ProposedClassifier(nn.Module):
    def __init__(self,num_classes,lstm_hidden=256,dropout=0.1):
        super().__init__()
        self.encoder=RobertaModel.from_pretrained('roberta-base')
        self.bilstm=nn.LSTM(self.encoder.config.hidden_size,lstm_hidden,num_layers=2,bidirectional=True,batch_first=True,dropout=dropout)
        self.attention=nn.Sequential(nn.Linear(lstm_hidden*2,lstm_hidden),nn.Tanh(),nn.Linear(lstm_hidden,1))
        self.dropout=nn.Dropout(dropout)
        self.classifier=nn.Sequential(nn.Linear(lstm_hidden*2,lstm_hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(lstm_hidden,num_classes))
    def forward(self,input_ids,attention_mask):
        out=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        lstm_out,_=self.bilstm(out)
        attn=self.attention(lstm_out).masked_fill(attention_mask.unsqueeze(-1)==0,float('-inf'))
        attn=torch.softmax(attn,dim=1)
        pooled=torch.sum(lstm_out*attn,dim=1)
        return self.classifier(self.dropout(pooled))

def train_proposed(data,n_groups,epochs=5):
    print(f"\n{'='*60}\n🚀 Proposed Model (RoBERTa+BiLSTM+Attention) ({n_groups} groups)\n{'='*60}")
    gc.collect();torch.cuda.empty_cache()
    tokenizer=RobertaTokenizerFast.from_pretrained('roberta-base',add_prefix_space=True)
    train_texts=[item['text'] for item in data['train']];train_labels=[item['label'] for item in data['train']]
    val_texts=[item['text'] for item in data['val']];val_labels=[item['label'] for item in data['val']]
    test_texts=[item['text'] for item in data['test']];test_labels=[item['label'] for item in data['test']]
    train_loader=DataLoader(ClassificationDataset(train_texts,train_labels,tokenizer,Config.MAX_LEN),batch_size=Config.BATCH_SIZE,shuffle=True)
    val_loader=DataLoader(ClassificationDataset(val_texts,val_labels,tokenizer,Config.MAX_LEN),batch_size=Config.BATCH_SIZE)
    test_loader=DataLoader(ClassificationDataset(test_texts,test_labels,tokenizer,Config.MAX_LEN),batch_size=Config.BATCH_SIZE)
    model=ProposedClassifier(len(data['label2id'])).to(Config.DEVICE)
    criterion=FocalLoss(alpha=compute_class_weights(train_labels).to(Config.DEVICE),gamma=2.0)
    optimizer=AdamW(model.parameters(),lr=Config.LR)
    scheduler=get_linear_schedule_with_warmup(optimizer,len(train_loader)*epochs//10,len(train_loader)*epochs)
    best_f1=0;best_state=None;start=time.time()
    for epoch in range(epochs):
        model.train();total_loss=0
        for batch in tqdm(train_loader,desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            logits=model(batch['input_ids'].to(Config.DEVICE),batch['attention_mask'].to(Config.DEVICE))
            loss=criterion(logits,batch['label'].to(Config.DEVICE));loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();scheduler.step();total_loss+=loss.item()
        model.eval();val_preds,val_true=[],[]
        with torch.no_grad():
            for batch in val_loader:
                logits=model(batch['input_ids'].to(Config.DEVICE),batch['attention_mask'].to(Config.DEVICE))
                val_preds.extend(logits.argmax(dim=-1).cpu().numpy());val_true.extend(batch['label'].numpy())
        val_f1=f1_score(val_true,val_preds,average='macro')
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Val F1={val_f1:.4f}")
        if val_f1>best_f1:best_f1=val_f1;best_state=model.state_dict().copy()
    model.load_state_dict(best_state);model.eval();test_preds,test_true=[],[]
    with torch.no_grad():
        for batch in test_loader:
            logits=model(batch['input_ids'].to(Config.DEVICE),batch['attention_mask'].to(Config.DEVICE))
            test_preds.extend(logits.argmax(dim=-1).cpu().numpy());test_true.extend(batch['label'].numpy())
    results={'model':'Proposed(RoBERTa+BiLSTM+Attn)','n_groups':n_groups,'accuracy':round(accuracy_score(test_true,test_preds),4),'f1_macro':round(f1_score(test_true,test_preds,average='macro'),4),'f1_weighted':round(f1_score(test_true,test_preds,average='weighted'),4),'precision':round(precision_score(test_true,test_preds,average='macro'),4),'recall':round(recall_score(test_true,test_preds,average='macro'),4),'training_time':round(time.time()-start,2)}
    print(f"✅ Acc={results['accuracy']:.4f}, F1={results['f1_macro']:.4f}")
    torch.save(best_state,Config.MODELS_DIR/f"proposed_cls_{n_groups}.pt")
    del model;gc.collect();torch.cuda.empty_cache()
    return results,confusion_matrix(test_true,test_preds)

def plot_cm(cm,labels,model_name,n_groups,path):
    plt.figure(figsize=(10,8))
    cm_norm=cm.astype('float')/(cm.sum(axis=1)[:,np.newaxis]+1e-6)
    if len(labels)>15:sns.heatmap(cm_norm,annot=False,cmap='Blues')
    else:sns.heatmap(cm_norm,annot=True,fmt='.2f',cmap='Blues',xticklabels=labels,yticklabels=labels)
    plt.title(f'{model_name} ({n_groups} groups)');plt.xlabel('Predicted');plt.ylabel('True')
    plt.tight_layout();plt.savefig(path,dpi=150);plt.close()

def plot_results(results,path):
    df=pd.DataFrame(results);fig,axes=plt.subplots(2,2,figsize=(14,10))
    for i,ng in enumerate([10,20,50]):
        sub=df[df['n_groups']==ng]
        axes[0,0].barh(np.arange(len(sub))+i*0.25,sub['f1_macro'],height=0.25,label=f'{ng}g')
    axes[0,0].set_xlabel('F1 Macro');axes[0,0].legend();axes[0,0].set_title('F1 by Model')
    df.pivot(index='model',columns='n_groups',values='accuracy').plot(kind='bar',ax=axes[0,1])
    axes[0,1].set_title('Accuracy');axes[0,1].tick_params(axis='x',rotation=45)
    for m in df['model'].unique():
        md=df[df['model']==m].sort_values('n_groups')
        axes[1,0].plot(md['n_groups'],md['f1_macro'],'o-',label=m)
    axes[1,0].set_xlabel('Groups');axes[1,0].set_ylabel('F1');axes[1,0].legend(fontsize=7);axes[1,0].grid(True,alpha=0.3)
    df.pivot(index='model',columns='n_groups',values='training_time').plot(kind='bar',ax=axes[1,1])
    axes[1,1].set_title('Training Time (s)');axes[1,1].tick_params(axis='x',rotation=45)
    plt.tight_layout();plt.savefig(path,dpi=150);plt.close();print(f"📊 Saved: {path}")

def main():
    print("="*70+"\n🚀 Comprehensive Classification\n"+"="*70)
    print(f"Started: {datetime.now()}\nDevice: {Config.DEVICE}")
    if torch.cuda.is_available():print(f"GPU: {torch.cuda.get_device_name(0)}")
    Config.RESULTS_DIR.mkdir(exist_ok=True);Config.FIGURES_DIR.mkdir(exist_ok=True);Config.MODELS_DIR.mkdir(exist_ok=True)
    all_results=[]
    for ng in Config.GROUP_CONFIGS:
        print(f"\n{'#'*70}\n# {ng} GROUPS\n{'#'*70}")
        data=load_classification_data(ng);labels=list(data['label2id'].keys())
        try:
            r,cm=train_traditional(data,ng);all_results.append(r)
            plot_cm(cm,labels,'TF-IDF+LR',ng,Config.FIGURES_DIR/f"cm_tfidf_{ng}.png")
        except Exception as e:print(f"❌ TF-IDF: {e}")
        for mn,mp in Config.TRANSFORMER_MODELS.items():
            try:
                r,cm=train_transformer(mn,mp,data,ng,Config.EPOCHS);all_results.append(r)
                plot_cm(cm,labels,mn,ng,Config.FIGURES_DIR/f"cm_{mn}_{ng}.png")
            except Exception as e:print(f"❌ {mn}: {e}")
        try:
            r,cm=train_proposed(data,ng,Config.EPOCHS);all_results.append(r)
            plot_cm(cm,labels,'Proposed',ng,Config.FIGURES_DIR/f"cm_proposed_{ng}.png")
        except Exception as e:print(f"❌ Proposed: {e}")
    with open(Config.RESULTS_DIR/"classification_results.json",'w') as f:json.dump(all_results,f,indent=2)
    plot_results(all_results,Config.FIGURES_DIR/"classification_comparison.png")
    print("\n"+"="*70+"\n📊 FINAL RESULTS\n"+"="*70)
    print(f"{'Model':<35}{'Groups':>8}{'Acc':>8}{'F1-M':>8}{'F1-W':>8}{'Time':>8}")
    print("-"*75)
    for r in sorted(all_results,key=lambda x:(x['n_groups'],-x['f1_macro'])):
        print(f"{r['model']:<35}{r['n_groups']:>8}{r['accuracy']:>8.4f}{r['f1_macro']:>8.4f}{r['f1_weighted']:>8.4f}{r['training_time']:>7.1f}s")
    print("="*70+f"\n✅ Done: {datetime.now()}\n"+"="*70)
    return all_results

if __name__=="__main__":main()
