"""Martin's A5 block: grouped out-of-fold stacking, with an untouched OLID test.

The meta-model is fitted on training-corpus OOF predictions only. Every fold
starts from the original pretrained checkpoint. Full-data models are used only
to produce the final test features. Saved manifests permit safe resumption.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import time
from datetime import datetime, timezone

os.environ.setdefault('HF_HOME',str(Path('.hf_cache').resolve()))
os.environ.setdefault('MPLCONFIGDIR',str(Path('.mpl_cache').resolve()))
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('WANDB_MODE','disabled')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED=42
K=3
MODELS={
 'bert':('google-bert/bert-base-uncased','86b5e0934494bd15c9632b12f734a8a67f723594'),
 'hatebert':('GroNLP/hateBERT','ab7a2d40f6c973cb57e63f23bfa2b51d981d028c'),
 'fbert':('diptanu/fBERT','c089daa35cb177238dd91ffe3010c2d8bbd069bf'),
}
TRAIN_FILES={'in_domain':'olid-train-small.csv','cross_domain':'hasoc-train.csv'}
PROB_COLUMNS=[f'{m}_prob' for m in MODELS]
EXTRA_COLUMNS=['log_characters','log_words','uppercase_fraction','exclamation_count','question_count','hashtag_count']
PARAMS=dict(num_train_epochs=3,learning_rate=2e-5,train_batch_size=8,
 gradient_accumulation_steps=2,eval_batch_size=32,max_seq_length=128,
 optimizer='AdamW',scheduler='linear_schedule_with_warmup',warmup_ratio=.06,
 warmup_steps=0,weight_decay=.01,adam_epsilon=1e-8,max_grad_norm=1.,
 manual_seed=SEED,fp16=True,do_lower_case=True,use_early_stopping=False,
 evaluate_during_training=False,save_model_every_epoch=False,
 save_eval_checkpoints=False,save_steps=-1,save_optimizer_and_scheduler=False,
 overwrite_output_dir=False,reprocess_input_data=True,no_cache=True,
 use_multiprocessing=False,use_multiprocessing_for_evaluation=False,
 dataloader_num_workers=0,thread_count=4,process_count=1,n_gpu=1,
 logging_steps=50,silent=False,wandb_project=None)

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def write_json(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str),encoding='utf-8')

def load_data(root):
 data={name:pd.read_csv(Path(root)/name) for name in [*TRAIN_FILES.values(),'olid-test.csv']}
 for name,d in data.items():
  assert {'id','text','labels'}.issubset(d.columns)
  assert d.id.is_unique and d.text.notna().all() and d.labels.isin([0,1]).all()
  assert len(d)==(860 if name=='olid-test.csv' else 5852)
 return data

def text_features(texts):
 rows=[]
 for text in texts:
  letters=[c for c in text if c.isalpha()]
  rows.append([np.log1p(len(text)),np.log1p(len(text.split())),
   sum(c.isupper() for c in letters)/max(1,len(letters)),
   text.count('!'),text.count('?'),text.count('#')])
 return pd.DataFrame(rows,columns=EXTRA_COLUMNS)

def split_folds(train,setting,out):
 splitter=StratifiedGroupKFold(n_splits=K,shuffle=True,random_state=SEED)
 fold_ids=np.full(len(train),-1,dtype=int);splits=[]
 for fold,(fit,held) in enumerate(splitter.split(train,train.labels,groups=train.text)):
  assert not set(fit)&set(held)
  assert not set(train.iloc[fit].text)&set(train.iloc[held].text)
  fold_ids[held]=fold;splits.append((fit,held))
 assert (fold_ids>=0).all()
 membership=train[['id','labels']].copy();membership['fold']=fold_ids
 membership.to_csv(out/f'folds_{setting}.csv',index=False)
 return splits,fold_ids

def seed_everything():
 import torch
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.use_deterministic_algorithms(True,warn_only=True)

def probability_frame(frame,pred,logits):
 logits=np.asarray(logits,dtype=np.float64)
 x=np.exp(logits-logits.max(axis=1,keepdims=True));p=x/x.sum(axis=1,keepdims=True)
 assert np.isfinite(p).all() and np.allclose(p.sum(axis=1),1)
 result=frame[['id','text','labels']].copy()
 result['prediction']=np.asarray(pred,dtype=int)
 result['probability_offensive']=p[:,1];result['probability_non_offensive']=p[:,0]
 result['logit_0']=logits[:,0];result['logit_1']=logits[:,1]
 assert np.array_equal(result.prediction,np.argmax(logits,axis=1))
 return result

def fit_predict(model_name,setting,fold,train,held,out,model_root,cache_dir):
 import torch
 from simpletransformers.classification import ClassificationModel,ClassificationArgs
 from huggingface_hub import snapshot_download
 task=f'{model_name}_{setting}_{fold}'
 manifest_path=out/'runs'/f'{task}.json'
 pred_path=out/'base_predictions'/f'{task}.csv'
 if manifest_path.exists():
  manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
  assert manifest['status']=='completed' and manifest['predictions_sha256']==sha(pred_path)
  restored=pd.read_csv(pred_path)
  assert restored[['id','text','labels']].reset_index(drop=True).equals(held[['id','text','labels']].reset_index(drop=True))
  print('Reusing verified completed task:',task,flush=True)
  return restored
 assert torch.cuda.is_available(),'CUDA GPU required for the documented training recipe.'
 seed_everything();repo,revision=MODELS[model_name]
 # Serial downloads avoid a Windows race in the hub's symlink-support probe.
 snapshot=snapshot_download(repo,revision=revision,cache_dir=str(cache_dir),max_workers=1,
  allow_patterns=['config.json','model.safetensors','vocab.txt','tokenizer.json','tokenizer_config.json','special_tokens_map.json'])
 save=fold=='full'
 destination=model_root/f'{model_name}_{setting}' if save else out/'temporary_model_paths'/task
 if save and destination.exists() and any(destination.iterdir()):
  raise FileExistsError(f'Incomplete or existing model folder: {destination}; preserve it before restarting.')
 args=ClassificationArgs();args.update_from_dict(dict(PARAMS,
  no_save=not save,output_dir=str(destination),
  cache_dir=str(out/'feature_cache'/task),tensorboard_dir=str(out/'tensorboard'/task),
  best_model_dir=str(out/'unused_best_models'/task)))
 started=datetime.now(timezone.utc).isoformat();t0=time.perf_counter()
 print(f'BEGIN {task}: train={len(train)}, predict={len(held)}, source={repo}@{revision}',flush=True)
 write_json(out/'progress.json',{'task':task,'status':'training','started_utc':started})
 model=ClassificationModel('bert',snapshot,num_labels=2,args=args,use_cuda=True)
 assert model.tokenizer.do_lower_case
 model.model.config.id2label={0:'non-offensive',1:'offensive'}
 model.model.config.label2id={'non-offensive':0,'offensive':1}
 steps,loss=model.train_model(train[['text','labels']].reset_index(drop=True),show_running_loss=False)
 pred,logits=model.predict(held.text.tolist())
 frame=probability_frame(held,pred,logits)
 pred_path.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(pred_path,index=False)
 if save:
  model.model.save_pretrained(destination,safe_serialization=True);model.tokenizer.save_pretrained(destination)
 manifest={'status':'completed','task':task,'source':repo,'revision':revision,
  'setting':setting,'fold':fold,'parameters':PARAMS,'resolved_args':vars(model.args),
  'train_ids':train.id.tolist(),'prediction_ids':held.id.tolist(),
  'train_rows':len(train),'prediction_rows':len(held),'optimizer_steps':int(steps),
  'reported_training_loss':float(loss),'elapsed_seconds':time.perf_counter()-t0,
  'started_utc':started,'completed_utc':datetime.now(timezone.utc).isoformat(),
  'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'python':platform.python_version(),
  'versions':{n:importlib.metadata.version(n) for n in ['torch','simpletransformers','transformers','numpy','pandas','scikit-learn']},
  'predictions_sha256':sha(pred_path),'full_model_saved':save,
  'selection':'Fixed final epoch; no development or test score used for selection'}
 if save:manifest['model_files_sha256']={p.name:sha(p) for p in destination.iterdir() if p.is_file()}
 write_json(manifest_path,manifest)
 print(f'END {task}: steps={steps}, seconds={manifest["elapsed_seconds"]:.1f}',flush=True)
 del model;gc.collect();torch.cuda.empty_cache()
 return frame

def reuse_hatebert(setting,test,out,model_root,a4_dir):
 """Reload A4 weights and verify their outputs; no retraining or test selection."""
 import torch
 from simpletransformers.classification import ClassificationModel
 origin=Path(a4_dir)/'hatebert_models'/setting
 dest=model_root/f'hatebert_{setting}'
 if not dest.exists():shutil.copytree(origin,dest)
 manifest=json.loads((Path(a4_dir)/'hatebert_results'/f'run_hatebert_{setting}.json').read_text(encoding='utf-8'))
 for filename,digest in manifest['model_files_sha256'].items():assert sha(dest/filename)==digest
 seed_everything()
 model=ClassificationModel('bert',str(dest),use_cuda=True,args=dict(PARAMS,silent=True))
 pred,logits=model.predict(test.text.tolist());frame=probability_frame(test,pred,logits)
 original=pd.read_csv(Path(a4_dir)/'hatebert_results'/f'results_hatebert_{setting}.csv')
 assert np.array_equal(frame.prediction,original.prediction)
 assert np.allclose(frame.probability_offensive,original.probability_offensive,atol=1e-7,rtol=0)
 p=out/'base_predictions'/f'hatebert_{setting}_full.csv';p.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(p,index=False)
 write_json(out/'runs'/f'hatebert_{setting}_full.json',{
  'status':'completed','task':f'hatebert_{setting}_full','source':'A4 saved checkpoint',
  'original_run':manifest,'predictions_sha256':sha(p),'all_860_predictions_match_A4':True,
  'model_files_sha256':{f.name:sha(f) for f in dest.iterdir() if f.is_file()},
  'parameters':PARAMS,'setting':setting,'fold':'full'})
 del model;gc.collect();torch.cuda.empty_cache()
 return frame

def generate(root='.',a4_dir=None):
 root=Path(root).resolve();out=root/'stacking_results';out.mkdir(exist_ok=True)
 model_root=root/'base_models';model_root.mkdir(exist_ok=True)
 data=load_data(root);test=data['olid-test.csv']
 write_json(out/'prespecified_protocol.json',{
  'k':K,'seed':SEED,'cv':'StratifiedGroupKFold; identical texts stay together',
  'models':MODELS,'base_parameters':PARAMS,'probability_features':PROB_COLUMNS,
  'extra_features':EXTRA_COLUMNS,'meta_model':'StandardScaler + LogisticRegression',
  'meta_parameters':{'C':1.0,'max_iter':2000,'class_weight':None,'solver':'lbfgs','random_state':42},
  'selection':'Prespecified; final epoch; no test-based selection',
  'test_usage':'Predict only, then evaluate once all training outputs are fixed',
  'data_sha256':{name:sha(root/name) for name in data},
  'primary_test_pool':'Matched full-training recipe; reused A4 HateBERT, newly trained BERT/fBERT',
  'diagnostics':'Probability-only stacking ablation; shared-pool sensitivity evaluated separately, never used for selection'})
 for setting,filename in TRAIN_FILES.items():
  train=data[filename];splits,fold_ids=split_folds(train,setting,out)
  oof=train[['id','text','labels']].copy();oof['fold']=fold_ids
  test_features=test[['id','text','labels']].copy()
  for model_name in MODELS:
   values=np.full(len(train),np.nan)
   for fold,(fit,held) in enumerate(splits):
    frame=fit_predict(model_name,setting,fold,train.iloc[fit],train.iloc[held],out,model_root,root/'.hf_cache')
    assert np.array_equal(frame.id,train.iloc[held].id)
    values[held]=frame.probability_offensive
   assert np.isfinite(values).all()
   oof[f'{model_name}_prob']=values
   if model_name=='hatebert' and a4_dir:
    full=reuse_hatebert(setting,test,out,model_root,a4_dir)
   else:full=fit_predict(model_name,setting,'full',train,test,out,model_root,root/'.hf_cache')
   test_features[f'{model_name}_prob']=full.probability_offensive.to_numpy()
   full.to_csv(root/f'{model_name}_probs_{setting}.csv',index=False)
   if model_name=='hatebert':full.to_csv(root/f'results_hatebert_{setting}.csv',index=False)
  oof=pd.concat([oof,text_features(train.text)],axis=1)
  test_features=pd.concat([test_features,text_features(test.text)],axis=1)
  oof.to_csv(out/f'oof_features_{setting}.csv',index=False)
  test_features.to_csv(out/f'test_features_{setting}.csv',index=False)
  print('Feature generation complete:',setting,flush=True)
 write_json(out/'progress.json',{'status':'all_base_predictions_complete'})

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='.')
 p.add_argument('--a4-dir',default=None,help='Optional original A4 package, used only to reuse verified HateBERT weights.')
 a=p.parse_args();generate(a.root,a.a4_dir)
