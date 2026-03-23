#!/usr/bin/env python3
import os,gc,json,pickle,random,time,sys
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
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from transformers import AutoModel,AutoTokenizer,RobertaModel,RobertaTokenizerFast,T5ForConditionalGeneration,T5Tokenizer,get_linear_schedule_with_warmup
from sklearn.metrics import f1_score,precision_score,recall_score
import warnings
warnings.filterwarnings('ignore')

class Config:
    BASE_DIR=Path("/home/macierz/mohabdal/TerrorismNER_Project")
    CHECKPOINT_DIR=BASE_DIR/"checkpoints"
    RESULTS_DIR=BASE_DIR/"results"
    FIGURES_DIR=BASE_DIR/"figures"
    DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED=42;MAX_LEN=128;BATCH_SIZE=8;EPOCHS=5;LR=2e-5
    LLM_MODEL="google/flan-t5-base"
    FEW_SHOT_SIZES=[10,50,100]

def set_seed(seed=42):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
set_seed(42)

GAZETTEER={"Taliban":["Taliban","Afghan Taliban"],"Al-Qaeda":["Al-Qaeda","Al-Qaida","Al Qaeda"],"Islamic State":["Islamic State","ISIS","ISIL","Daesh"],"Boko Haram":["Boko Haram"],"Al-Shabaab":["Al-Shabaab","Al Shabaab"],"Hamas":["Hamas"],"Hezbollah":["Hezbollah","Hizbollah"],"TTP":["TTP","Tehrik-i-Taliban Pakistan"],"LTTE":["LTTE","Tamil Tigers"],"PKK":["PKK","Kurdistan Workers Party"],"FARC":["FARC"],"IRA":["IRA","Irish Republican Army"]}
GAZETTEER_FLAT={alias.lower():canon for canon,aliases in GAZETTEER.items() for alias in aliases}

def load_ner_data():
    with open(Config.CHECKPOINT_DIR/"ner_data_basic.pkl",'rb') as f:data=pickle.load(f)
    print(f"✅ NER Data: Train={len(data['train'])}, Val={len(data['val'])}, Test={len(data['test'])}")
    return data

def load_previous_results():
    try:
        with open(Config.RESULTS_DIR/"training_results.json",'r') as f:return json.load(f)
    except:return {}

def extract_entities(tokens,labels):
    entities=[];i=0
    while i<len(labels):
        if labels[i]=='B-TERROR_GROUP':
            ent=[tokens[i]];j=i+1
            while j<len(labels) and labels[j]=='I-TERROR_GROUP':ent.append(tokens[j]);j+=1
            entities.append(' '.join(ent));i=j
        else:i+=1
    return entities

def calc_ner_metrics(preds,truths):
    tp=fp=fn=0
    for p,t in zip(preds,truths):
        ps,ts=set(p),set(t)
        tp+=len(ps&ts);fp+=len(ps-ts);fn+=len(ts-ps)
    prec=tp/(tp+fp) if tp+fp>0 else 0
    rec=tp/(tp+fn) if tp+fn>0 else 0
    f1=2*prec*rec/(prec+rec) if prec+rec>0 else 0
    return {'precision':round(prec,4),'recall':round(rec,4),'f1':round(f1,4)}

# ============ MODEL 1: SpaCy NER ============
def run_spacy_ner(test_data):
    print(f"\n{'='*60}\n🔬 SpaCy NER\n{'='*60}")
    try:
        import spacy
        nlp=spacy.load("en_core_web_lg")
    except:
        print("❌ SpaCy not installed. Run: pip install spacy && python -m spacy download en_core_web_lg")
        return None
    start=time.time()
    preds,truths=[],[]
    for sample in tqdm(test_data,desc="SpaCy NER"):
        text=' '.join(sample['tokens'])
        doc=nlp(text)
        found=[]
        for ent in doc.ents:
            if ent.label_ in ['ORG','GPE','NORP']:
                if ent.text.lower() in GAZETTEER_FLAT:found.append(ent.text)
        for alias in GAZETTEER_FLAT:
            if alias in text.lower() and alias not in [f.lower() for f in found]:
                start_idx=text.lower().find(alias)
                found.append(text[start_idx:start_idx+len(alias)])
        preds.append(found)
        truths.append(extract_entities(sample['tokens'],sample['labels']))
    metrics=calc_ner_metrics(preds,truths)
    metrics['model']='SpaCy';metrics['time']=round(time.time()-start,2)
    print(f"✅ P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
    return metrics

# ============ MODEL 2: Zero-shot/Few-shot LLM ============
class LLMExtractor:
    def __init__(self):
        print(f"📥 Loading {Config.LLM_MODEL}...")
        self.tokenizer=T5Tokenizer.from_pretrained(Config.LLM_MODEL)
        self.model=T5ForConditionalGeneration.from_pretrained(Config.LLM_MODEL,torch_dtype=torch.float16).to(Config.DEVICE)
        self.model.eval()
        print("✅ LLM loaded")
    def extract(self,text,examples=None):
        if examples:
            prompt="Extract terrorist group names.\n\n"
            for ex in examples[:3]:
                ex_text=' '.join(ex['tokens'])[:150]
                ex_ents=extract_entities(ex['tokens'],ex['labels'])
                prompt+=f"Text: {ex_text}\nGroups: {', '.join(ex_ents) if ex_ents else 'None'}\n\n"
            prompt+=f"Text: {text[:300]}\nGroups:"
        else:
            prompt=f"Extract terrorist group names from: {text[:400]}\nGroups:"
        inputs=self.tokenizer(prompt,return_tensors="pt",max_length=512,truncation=True).to(Config.DEVICE)
        with torch.no_grad():
            out=self.model.generate(**inputs,max_new_tokens=50,num_beams=2)
        resp=self.tokenizer.decode(out[0],skip_special_tokens=True)
        if not resp or resp.lower() in ['none','no']:return []
        return [e.strip() for e in resp.replace(';',',').split(',') if e.strip() and e.lower() not in ['none','no']]

def run_llm_ner(test_data,train_data,mode='zero-shot',n_examples=0):
    print(f"\n{'='*60}\n🔬 Flan-T5 ({mode})\n{'='*60}")
    llm=LLMExtractor()
    examples=random.sample(train_data,n_examples) if n_examples>0 else None
    test_subset=test_data[:300]
    start=time.time()
    preds,truths=[],[]
    for sample in tqdm(test_subset,desc=f"LLM {mode}"):
        text=' '.join(sample['tokens'])
        found=llm.extract(text,examples)
        preds.append(found)
        truths.append(extract_entities(sample['tokens'],sample['labels']))
    metrics=calc_ner_metrics(preds,truths)
    metrics['model']=f'Flan-T5({mode})';metrics['time']=round(time.time()-start,2)
    print(f"✅ P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
    del llm;gc.collect();torch.cuda.empty_cache()
    return metrics

# ============ MODEL 3: Proposed (RoBERTa+BiLSTM+CRF+Gazetteer) ============
class CRF(nn.Module):
    def __init__(self,num_tags):
        super().__init__()
        self.num_tags=num_tags
        self.transitions=nn.Parameter(torch.randn(num_tags,num_tags))
        self.start_trans=nn.Parameter(torch.randn(num_tags))
        self.end_trans=nn.Parameter(torch.randn(num_tags))
        with torch.no_grad():self.transitions[0,2]=-10000;self.start_trans[2]=-10000
    def forward(self,emissions,tags,mask):
        emissions=emissions.transpose(0,1);tags=tags.transpose(0,1);mask=mask.transpose(0,1)
        score=self._score(emissions,tags,mask);norm=self._norm(emissions,mask)
        return -(score-norm).mean()
    def _score(self,emissions,tags,mask):
        seq_len,batch=tags.shape;score=self.start_trans[tags[0]]+emissions[0,torch.arange(batch),tags[0]]
        for i in range(1,seq_len):score+=(self.transitions[tags[i-1],tags[i]]+emissions[i,torch.arange(batch),tags[i]])*mask[i]
        ends=mask.long().sum(0)-1;score+=self.end_trans[tags[ends,torch.arange(batch)]]
        return score
    def _norm(self,emissions,mask):
        seq_len,batch,ntags=emissions.shape;score=self.start_trans+emissions[0]
        for i in range(1,seq_len):
            next_score=score.unsqueeze(2)+self.transitions+emissions[i].unsqueeze(1)
            next_score=torch.logsumexp(next_score,dim=1)
            score=torch.where(mask[i].unsqueeze(1),next_score,score)
        return torch.logsumexp(score+self.end_trans,dim=1)
    def decode(self,emissions,mask):
        emissions=emissions.transpose(0,1);mask=mask.transpose(0,1)
        seq_len,batch,ntags=emissions.shape;score=self.start_trans+emissions[0];history=[]
        for i in range(1,seq_len):
            next_score=score.unsqueeze(2)+self.transitions+emissions[i].unsqueeze(1)
            next_score,idx=next_score.max(dim=1)
            score=torch.where(mask[i].unsqueeze(1),next_score,score);history.append(idx)
        score+=self.end_trans;ends=mask.long().sum(0)-1;result=[]
        for b in range(batch):
            _,best=score[b].max(0);tags=[best.item()]
            for h in reversed(history[:ends[b]]):tags.append(h[b][tags[-1]].item())
            result.append(list(reversed(tags)))
        return result

class GazetteerEncoder(nn.Module):
    def __init__(self,dim=64):
        super().__init__()
        self.emb=nn.Embedding(4,dim);self.proj=nn.Linear(dim,dim)
    def forward(self,tokens_batch,seq_len,device):
        batch=len(tokens_batch);feats=torch.zeros(batch,seq_len,64,device=device)
        for b,tokens in enumerate(tokens_batch):
            for i,tok in enumerate(tokens[:seq_len]):
                match=1 if tok.lower() in GAZETTEER_FLAT else (2 if any(tok.lower() in a for a in GAZETTEER_FLAT) else 0)
                feats[b,i]=self.emb(torch.tensor(match,device=device))
        return self.proj(feats)

class ProposedNER(nn.Module):
    def __init__(self,num_labels=3,use_crf=True,use_gaz=True):
        super().__init__()
        self.encoder=RobertaModel.from_pretrained('roberta-base')
        self.bilstm=nn.LSTM(768,256,num_layers=2,bidirectional=True,batch_first=True,dropout=0.1)
        self.lstm_proj=nn.Linear(512,768)
        self.use_gaz=use_gaz;self.use_crf=use_crf
        if use_gaz:self.gaz=GazetteerEncoder();self.fuse=nn.Linear(832,768)
        self.dropout=nn.Dropout(0.1);self.classifier=nn.Linear(768,num_labels)
        if use_crf:self.crf=CRF(num_labels)
    def forward(self,input_ids,attention_mask,labels=None,tokens=None):
        out=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        if self.use_gaz and tokens:
            gaz=self.gaz(tokens,input_ids.size(1),input_ids.device)
            out=self.fuse(torch.cat([out,gaz],dim=-1))
        lengths=attention_mask.sum(1).cpu();sorted_len,sorted_idx=lengths.sort(descending=True)
        packed=pack_padded_sequence(out[sorted_idx],sorted_len.tolist(),batch_first=True)
        lstm_out,_=self.bilstm(packed)
        unpacked,_=pad_packed_sequence(lstm_out,batch_first=True,total_length=input_ids.size(1))
        _,orig_idx=sorted_idx.sort();out=self.lstm_proj(unpacked[orig_idx])
        emissions=self.classifier(self.dropout(out))
        loss=None
        if self.use_crf:
            if labels is not None:
                crf_labels=labels.clone();crf_labels[crf_labels==-100]=0
                loss=self.crf(emissions,crf_labels,attention_mask.bool())
            preds=self.crf.decode(emissions,attention_mask.bool())
        else:
            if labels is not None:loss=nn.CrossEntropyLoss(ignore_index=-100)(emissions.view(-1,3),labels.view(-1))
            preds=emissions.argmax(-1).tolist()
        return {'loss':loss,'preds':preds}

class NERDataset(Dataset):
    def __init__(self,data,tokenizer,label2id,max_len=128):
        self.data=data;self.tokenizer=tokenizer;self.label2id=label2id;self.max_len=max_len
    def __len__(self):return len(self.data)
    def __getitem__(self,idx):
        item=self.data[idx]
        enc=self.tokenizer(item['tokens'],is_split_into_words=True,truncation=True,max_length=self.max_len,padding='max_length',return_tensors='pt')
        word_ids=enc.word_ids();labels=[]
        for w in word_ids:
            if w is None or w>=len(item['labels']):labels.append(-100)
            else:labels.append(self.label2id.get(item['labels'][w],0))
        return {'input_ids':enc['input_ids'].squeeze(),'attention_mask':enc['attention_mask'].squeeze(),'labels':torch.tensor(labels),'tokens':item['tokens']}

def collate_fn(batch):
    return {'input_ids':torch.stack([x['input_ids'] for x in batch]),'attention_mask':torch.stack([x['attention_mask'] for x in batch]),'labels':torch.stack([x['labels'] for x in batch]),'tokens':[x['tokens'] for x in batch]}

def train_proposed_ner(data,use_crf=True,use_gaz=True,epochs=5):
    name=f"Proposed(CRF={use_crf},Gaz={use_gaz})"
    print(f"\n{'='*60}\n🚀 {name}\n{'='*60}")
    gc.collect();torch.cuda.empty_cache()
    tokenizer=RobertaTokenizerFast.from_pretrained('roberta-base',add_prefix_space=True)
    label2id={'O':0,'B-TERROR_GROUP':1,'I-TERROR_GROUP':2}
    train_loader=DataLoader(NERDataset(data['train'],tokenizer,label2id),batch_size=Config.BATCH_SIZE,shuffle=True,collate_fn=collate_fn)
    val_loader=DataLoader(NERDataset(data['val'],tokenizer,label2id),batch_size=Config.BATCH_SIZE,collate_fn=collate_fn)
    test_loader=DataLoader(NERDataset(data['test'],tokenizer,label2id),batch_size=Config.BATCH_SIZE,collate_fn=collate_fn)
    model=ProposedNER(use_crf=use_crf,use_gaz=use_gaz).to(Config.DEVICE)
    optimizer=AdamW(model.parameters(),lr=Config.LR)
    scheduler=get_linear_schedule_with_warmup(optimizer,len(train_loader)*epochs//10,len(train_loader)*epochs)
    best_f1=0;best_state=None;start=time.time()
    for epoch in range(epochs):
        model.train();total_loss=0
        for batch in tqdm(train_loader,desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            out=model(batch['input_ids'].to(Config.DEVICE),batch['attention_mask'].to(Config.DEVICE),batch['labels'].to(Config.DEVICE),batch['tokens'])
            out['loss'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step();scheduler.step();total_loss+=out['loss'].item()
        model.eval();all_preds,all_labels,all_masks=[],[],[]
        with torch.no_grad():
            for batch in val_loader:
                out=model(batch['input_ids'].to(Config.DEVICE),batch['attention_mask'].to(Config.DEVICE),tokens=batch['tokens'])
                preds=out['preds']
                if isinstance(preds[0],list):
                    for p in preds:all_preds.append(p+[0]*(Config.MAX_LEN-len(p)))
                else:all_preds.extend(preds)
                all_labels.extend(batch['labels'].tolist());all_masks.extend(batch['attention_mask'].tolist())
        tp=fp=fn=0
        for p,l,m in zip(all_preds,all_labels,all_masks):
            for pi,li,mi in zip(p[:len(l)],l,m):
                if mi==1 and li!=-100:
                    if pi==li and pi!=0:tp+=1
                    elif pi!=0 and pi!=li:fp+=1
                    elif li!=0 and pi==0:fn+=1
        val_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn>0 else 0
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Val F1={val_f1:.4f}")
        if val_f1>best_f1:best_f1=val_f1;best_state=model.state_dict().copy()
    model.load_state_dict(best_state);model.eval();all_preds,all_labels,all_masks=[],[],[]
    with torch.no_grad():
        for batch in test_loader:
            out=model(batch['input_ids'].to(Config.DEVICE),batch['attention_mask'].to(Config.DEVICE),tokens=batch['tokens'])
            preds=out['preds']
            if isinstance(preds[0],list):
                for p in preds:all_preds.append(p+[0]*(Config.MAX_LEN-len(p)))
            else:all_preds.extend(preds)
            all_labels.extend(batch['labels'].tolist());all_masks.extend(batch['attention_mask'].tolist())
    tp=fp=fn=0
    for p,l,m in zip(all_preds,all_labels,all_masks):
        for pi,li,mi in zip(p[:len(l)],l,m):
            if mi==1 and li!=-100:
                if pi==li and pi!=0:tp+=1
                elif pi!=0 and pi!=li:fp+=1
                elif li!=0 and pi==0:fn+=1
    prec=tp/(tp+fp) if tp+fp>0 else 0
    rec=tp/(tp+fn) if tp+fn>0 else 0
    f1=2*prec*rec/(prec+rec) if prec+rec>0 else 0
    results={'model':name,'precision':round(prec,4),'recall':round(rec,4),'f1':round(f1,4),'time':round(time.time()-start,2)}
    print(f"✅ P={prec:.4f}, R={rec:.4f}, F1={f1:.4f}")
    del model;gc.collect();torch.cuda.empty_cache()
    return results

def plot_comparison(results,prev_results,path):
    all_res={**prev_results}
    for r in results:all_res[r['model']]=r
    models=list(all_res.keys());f1s=[all_res[m].get('f1',0) for m in models]
    plt.figure(figsize=(12,6))
    colors=['green' if 'Proposed' in m else 'steelblue' if m in prev_results else 'coral' for m in models]
    bars=plt.barh(models,f1s,color=colors)
    for bar,f1 in zip(bars,f1s):plt.text(f1+0.01,bar.get_y()+bar.get_height()/2,f'{f1:.4f}',va='center')
    plt.xlabel('F1 Score');plt.title('NER Model Comparison');plt.xlim(0,1.1);plt.tight_layout()
    plt.savefig(path,dpi=150);plt.close();print(f"📊 Saved: {path}")

def main():
    print("="*70+f"\n🚀 Comprehensive NER Comparison\n"+"="*70)
    print(f"Started: {datetime.now()}\nDevice: {Config.DEVICE}")
    if torch.cuda.is_available():print(f"GPU: {torch.cuda.get_device_name(0)}")
    Config.RESULTS_DIR.mkdir(exist_ok=True);Config.FIGURES_DIR.mkdir(exist_ok=True)
    data=load_ner_data();prev_results=load_previous_results()
    print(f"\n📊 Previous Results: {list(prev_results.keys())}")
    all_results=[]
    # 1. SpaCy
    r=run_spacy_ner(data['test']);
    if r:all_results.append(r)
    # 2. Zero-shot LLM
    r=run_llm_ner(data['test'],data['train'],'zero-shot',0);all_results.append(r)
    # 3. Few-shot LLM
    for n in Config.FEW_SHOT_SIZES:
        r=run_llm_ner(data['test'],data['train'],f'{n}-shot',n);all_results.append(r)
    # 4. Proposed Models
    r=train_proposed_ner(data,use_crf=True,use_gaz=True);all_results.append(r)
    r=train_proposed_ner(data,use_crf=True,use_gaz=False);all_results.append(r)
    r=train_proposed_ner(data,use_crf=False,use_gaz=True);all_results.append(r)
    # Save & Plot
    with open(Config.RESULTS_DIR/"ner_comparison_results.json",'w') as f:json.dump(all_results,f,indent=2)
    plot_comparison(all_results,prev_results,Config.FIGURES_DIR/"ner_comparison.png")
    print("\n"+"="*70+"\n📊 FINAL NER RESULTS\n"+"="*70)
    print(f"{'Model':<40}{'P':>10}{'R':>10}{'F1':>10}{'Time':>10}")
    print("-"*80)
    for m,v in prev_results.items():print(f"{m:<40}{v.get('precision','N/A'):>10}{v.get('recall','N/A'):>10}{v.get('f1','N/A'):>10}{'--':>10}")
    for r in sorted(all_results,key=lambda x:-x['f1']):print(f"{r['model']:<40}{r['precision']:>10.4f}{r['recall']:>10.4f}{r['f1']:>10.4f}{r['time']:>9.1f}s")
    print("="*70+f"\n✅ Done: {datetime.now()}\n"+"="*70)

if __name__=="__main__":main()
