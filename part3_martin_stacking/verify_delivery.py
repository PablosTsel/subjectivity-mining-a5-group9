"""Independent final verification of OOF provenance, metrics and checkpoint inference."""
from pathlib import Path
import argparse
import gc
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
from a5_stacking import MODELS,PROB_COLUMNS,EXTRA_COLUMNS,sha,seed_everything,write_json,load_data
from evaluate_stacking import audit_features

def verify(root='.',reload_base_models=False):
 root=Path(root).resolve();out=root/'stacking_results';data=load_data(root)
 checks=audit_features(root,out,data);test=data['olid-test.csv'];metric_checks=[]
 for setting in ['in_domain','cross_domain']:
  for stem in ['stacking','stacking_probabilities_only','hard_voting_matched','soft_voting_matched']:
   r=pd.read_csv(out/f'results_{stem}_{setting}.csv');m=pd.read_csv(out/f'metrics_{stem}_{setting}.csv').iloc[0]
   assert r[['id','text','labels']].equals(test[['id','text','labels']])
   assert r.prediction.isin([0,1]).all() and np.array_equal(r.correct,r.labels==r.prediction)
   p,rec,f,_=precision_recall_fscore_support(r.labels,r.prediction,average='macro',zero_division=0)
   for name,value in [('accuracy',accuracy_score(r.labels,r.prediction)),('macro_precision',p),('macro_recall',rec),('macro_f1',f)]:assert np.isclose(m[name],value,atol=1e-12)
   p,rec,f,s=precision_recall_fscore_support(r.labels,r.prediction,labels=[0,1],zero_division=0)
   for label in [0,1]:
    for name,value in [('precision',p[label]),('recall',rec[label]),('f1',f[label]),('support',s[label])]:assert np.isclose(m[f'{name}_{label}'],value,atol=1e-12)
   assert np.array_equal(confusion_matrix(r.labels,r.prediction,labels=[0,1]),pd.read_csv(out/f'confusion_{stem}_{setting}.csv',index_col=0).values)
   if stem=='stacking':
    # Portable JSON alone reproduces the sklearn metamodel probabilities.
    params=json.loads((out/f'stacking_parameters_{setting}.json').read_text())
    x=pd.read_csv(out/f'test_features_{setting}.csv')[params['features']].to_numpy()
    z=(x-np.array(params['scaler_mean']))/np.array(params['scaler_scale'])
    logits=z@np.array(params['coef'])+params['intercept']
    probabilities=1/(1+np.exp(-logits))
    np.testing.assert_allclose(probabilities,r.probability_offensive,atol=1e-12)
    assert np.array_equal((logits>0).astype(int),r.prediction)
   metric_checks.append({'setting':setting,'method':stem,'all_metrics_match':True,'rows':len(r)})
 base_checks=[]
 if reload_base_models:
  import torch
  from simpletransformers.classification import ClassificationModel
  for setting in ['in_domain','cross_domain']:
   for name in MODELS:
    path=root/'base_models'/f'{name}_{setting}'
    manifest=json.loads((out/'runs'/f'{name}_{setting}_full.json').read_text(encoding='utf-8'))
    for filename,digest in manifest['model_files_sha256'].items():assert sha(path/filename)==digest
    seed_everything()
    model=ClassificationModel('bert',str(path),use_cuda=True,args={
     'fp16':True,'eval_batch_size':32,'max_seq_length':128,'do_lower_case':True,
     'silent':True,'use_multiprocessing':False,'use_multiprocessing_for_evaluation':False,
     'no_cache':True,'reprocess_input_data':True,'thread_count':4,'wandb_project':None})
    pred,logits=model.predict(test.text.tolist())
    saved=pd.read_csv(root/f'{name}_probs_{setting}.csv')
    assert np.array_equal(pred,saved.prediction)
    difference=float(np.max(np.abs(np.asarray(logits)-saved[['logit_0','logit_1']].to_numpy())))
    np.testing.assert_allclose(logits,saved[['logit_0','logit_1']].to_numpy(),rtol=1e-5,atol=1e-5)
    values=np.asarray(logits,dtype=float);values=np.exp(values-values.max(axis=1,keepdims=True))
    probabilities=values[:,1]/values.sum(axis=1)
    np.testing.assert_allclose(probabilities,saved.probability_offensive,rtol=1e-5,atol=1e-6)
    base_checks.append({'model':name,'setting':setting,'all_predictions_match':True,'probabilities_match':True,'max_logit_difference':difference})
    print('Checkpoint verified:',name,setting,flush=True)
    del model;gc.collect();torch.cuda.empty_cache()
 result={'status':'passed','oof_provenance':checks,'metric_checks':metric_checks,
  'portable_meta_equations_verified':True,'base_checkpoint_reload_checks':base_checks,
  'base_reload_performed':reload_base_models}
 filename='verification_full.json' if reload_base_models else 'verification_metrics.json'
 write_json(out/filename,result)
 print('Delivery verification passed.',flush=True)
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--reload-base-models',action='store_true')
 a=p.parse_args();verify(reload_base_models=a.reload_base_models)
