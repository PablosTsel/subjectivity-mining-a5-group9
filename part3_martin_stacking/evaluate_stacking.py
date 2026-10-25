"""Fit the prespecified meta-model using OOF training rows, then evaluate test.

Re-running this module requires no GPU: it consumes the saved OOF/base outputs.
It never fits anything to the OLID test labels. No model selection is performed.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from shared_evaluation import evaluate_predictions
from a5_stacking import PROB_COLUMNS,EXTRA_COLUMNS,MODELS,TRAIN_FILES,SEED,K,text_features,sha,write_json,load_data

SETTINGS={'in_domain':'In-domain','cross_domain':'Cross-domain'}

def meta_model():
 return Pipeline([('scale',StandardScaler()),('classifier',LogisticRegression(
  C=1.0,solver='lbfgs',max_iter=2000,class_weight=None,random_state=SEED))])

def export(frame,pred,prob,stem,name,setting,out,figures=True):
 result=frame[['id','text','labels']].copy()
 result['prediction']=np.asarray(pred,dtype=int)
 if prob is not None:result['probability_offensive']=prob
 result['correct']=result.labels==result.prediction
 result.to_csv(out/f'results_{stem}_{setting}.csv',index=False)
 m=evaluate_predictions(result.labels,result.prediction,name,SETTINGS[setting],show_matrix=False)
 m.to_csv(out/f'metrics_{stem}_{setting}.csv',index=False)
 cm=confusion_matrix(result.labels,result.prediction,labels=[0,1])
 pd.DataFrame(cm,index=['true_0','true_1'],columns=['pred_0','pred_1']).to_csv(out/f'confusion_{stem}_{setting}.csv')
 if figures:
  fig,ax=plt.subplots(figsize=(5.8,4.8))
  ConfusionMatrixDisplay(cm,display_labels=['Non-offensive (0)','Offensive (1)']).plot(ax=ax,cmap='Blues',values_format='d',colorbar=False)
  ax.set_title(f'{name}: {SETTINGS[setting]}\nOLID test (n = 860)')
  fig.tight_layout();fig.savefig(out/f'confusion_{stem}_{setting}.png',dpi=180);plt.close(fig)
 return result,m

def audit_features(root,out,data):
 audit=[]
 protocol=json.loads((out/'prespecified_protocol.json').read_text())
 for name,digest in protocol['data_sha256'].items():assert sha(root/name)==digest
 for setting,filename in TRAIN_FILES.items():
  train=data[filename];test=data['olid-test.csv']
  oof=pd.read_csv(out/f'oof_features_{setting}.csv')
  final=pd.read_csv(out/f'test_features_{setting}.csv')
  membership=pd.read_csv(out/f'folds_{setting}.csv')
  assert oof[['id','text','labels']].equals(train[['id','text','labels']])
  assert final[['id','text','labels']].equals(test[['id','text','labels']])
  assert np.array_equal(membership.fold,oof.fold)
  assert oof.groupby('text').fold.nunique().max()==1
  assert np.isfinite(oof[PROB_COLUMNS+EXTRA_COLUMNS]).all().all()
  assert np.isfinite(final[PROB_COLUMNS+EXTRA_COLUMNS]).all().all()
  assert oof[PROB_COLUMNS].ge(0).all().all() and oof[PROB_COLUMNS].le(1).all().all()
  np.testing.assert_allclose(oof[EXTRA_COLUMNS],text_features(train.text),atol=1e-12)
  np.testing.assert_allclose(final[EXTRA_COLUMNS],text_features(test.text),atol=1e-12)
  for model in MODELS:
   seen=[]
   for fold in range(K):
    r=json.loads((out/'runs'/f'{model}_{setting}_{fold}.json').read_text(encoding='utf-8'))
    fit=set(r['train_ids']);held=set(r['prediction_ids'])
    assert not fit&held and fit|held==set(train.id)
    assert held==set(oof.loc[oof.fold==fold,'id'])
    assert not set(train.loc[train.id.isin(fit),'text'])&set(train.loc[train.id.isin(held),'text'])
    assert r['revision']==MODELS[model][1] and r['status']=='completed'
    p=out/'base_predictions'/f'{model}_{setting}_{fold}.csv'
    assert sha(p)==r['predictions_sha256']
    df=pd.read_csv(p).set_index('id')
    matched=oof.loc[oof.fold==fold].set_index('id')
    np.testing.assert_allclose(df.loc[matched.index,'probability_offensive'],matched[f'{model}_prob'],atol=1e-12)
    seen.extend(held)
   assert len(seen)==len(set(seen))==len(train)
  audit.append({'setting':setting,'oof_rows':len(oof),'n_folds':K,'all_rows_predicted_once_per_model':True,
   'train_heldout_ids_disjoint':True,'duplicate_texts_kept_together':True,
   'test_rows_unchanged':True,'probabilities_finite':True})
 return audit

def evaluate(root='.'):
 root=Path(root).resolve();out=root/'stacking_results';data=load_data(root)
 checks=audit_features(root,out,data);all_metrics=[];stack_results={};coefs=[]
 for setting in SETTINGS:
  train=pd.read_csv(out/f'oof_features_{setting}.csv');test=pd.read_csv(out/f'test_features_{setting}.csv')
  for stem,features,name in [('stacking',PROB_COLUMNS+EXTRA_COLUMNS,'Stacking'),('stacking_probabilities_only',PROB_COLUMNS,'Stacking (probabilities only)')]:
   model=meta_model();model.fit(train[features],train.labels)
   assert model.named_steps['classifier'].n_iter_[0]<2000
   pred=model.predict(test[features]);prob=model.predict_proba(test[features])[:,1]
   result,m=export(test,pred,prob,stem,name,setting,out,figures=stem=='stacking')
   m['pool']='Matched A5';all_metrics.append(m)
   saved=root/'meta_models'/f'{stem}_{setting}.joblib';saved.parent.mkdir(exist_ok=True)
   joblib.dump(model,saved)
   reloaded=joblib.load(saved)
   assert np.array_equal(reloaded.predict(test[features]),pred)
   np.testing.assert_allclose(reloaded.predict_proba(test[features])[:,1],prob,atol=1e-14)
   params={'features':features,'coef':model.named_steps['classifier'].coef_[0].tolist(),
    'intercept':float(model.named_steps['classifier'].intercept_[0]),
    'scaler_mean':model.named_steps['scale'].mean_.tolist(),'scaler_scale':model.named_steps['scale'].scale_.tolist(),
    'classes':model.classes_.tolist(),'training_rows':len(train),'fit_data':'OOF training features, never test labels',
    'parameters':{'C':1.,'solver':'lbfgs','max_iter':2000,'class_weight':None,'seed':42},
    'oof_sha256':sha(out/f'oof_features_{setting}.csv'),'meta_model_sha256':sha(saved)}
   write_json(out/f'{stem}_parameters_{setting}.json',params)
   if stem=='stacking':
    stack_results[setting]=result
    for feature,coef in zip(features,params['coef']):coefs.append({'setting':SETTINGS[setting],'feature':feature,'standardized_coefficient':coef})
  for model in MODELS:
   prob=test[f'{model}_prob'].to_numpy();pred=(prob>.5).astype(int)
   _,m=export(test,pred,prob,model+'_matched',model,setting,out,figures=False)
   m['pool']='Matched A5';all_metrics.append(m)
  hard=(test[PROB_COLUMNS].gt(.5).sum(axis=1)>=2).astype(int)
  soft_prob=test[PROB_COLUMNS].mean(axis=1);soft=(soft_prob>=.5).astype(int)
  for stem,name,pred,prob in [('hard_voting_matched','Hard voting (matched)',hard,None),('soft_voting_matched','Soft voting (matched)',soft,soft_prob)]:
   _,m=export(test,pred,prob,stem,name,setting,out,figures=False)
   m['pool']='Matched A5';all_metrics.append(m)
  # Re-evaluate the supplied outputs, retaining their different provenance.
  for owner,stem,name,pool in [('ines','hard_voting','Hard voting (Ines)','Provided A4 labels'),('helian','soft_voting','Soft voting (Helian)','Provided Helian probabilities')]:
   provided=pd.read_csv(root/'provided'/owner/f'results_{stem}_{setting}.csv')
   assert provided[['id','text','labels']].equals(test[['id','text','labels']])
   m=evaluate_predictions(provided.labels,provided.prediction,name,SETTINGS[setting],show_matrix=False)
   supplied=pd.read_csv(root/'provided'/owner/f'metrics_{stem}_{setting}.csv')
   np.testing.assert_allclose(m.select_dtypes('number'),supplied.select_dtypes('number'),atol=1e-12)
   m['pool']=pool;all_metrics.append(m)
  # Sensitivity only: apply the already fitted primary meta-model to the shared pool.
  # Different full-model recipes make this a diagnostic, not the primary comparison.
  shared=pd.read_csv(root/'provided/helian'/f'results_soft_voting_{setting}.csv')
  shared_features=pd.concat([shared[['id','text','labels',*PROB_COLUMNS]],text_features(shared.text)],axis=1)
  primary=joblib.load(root/'meta_models'/f'stacking_{setting}.joblib')
  _,m=export(shared_features,primary.predict(shared_features[PROB_COLUMNS+EXTRA_COLUMNS]),
   primary.predict_proba(shared_features[PROB_COLUMNS+EXTRA_COLUMNS])[:,1],
   'stacking_shared_pool_sensitivity','Stacking (shared-pool sensitivity)',setting,out,figures=False)
  m['pool']='Shared pool; base-fit mismatch';all_metrics.append(m)
 metrics=pd.concat(all_metrics,ignore_index=True)
 metrics.to_csv(out/'metrics_all_comparisons.csv',index=False)
 primary=metrics[metrics.model=='Stacking'].drop(columns='pool')
 primary.to_csv(out/'metrics_stacking_comparison.csv',index=False)
 metrics[metrics.pool=='Matched A5'].to_csv(out/'metrics_matched_pool_comparison.csv',index=False)
 pd.DataFrame(coefs).to_csv(out/'meta_coefficients.csv',index=False)
 changes=[]
 for name,group in metrics.groupby('model',sort=False):
  f=group.set_index('setting').macro_f1
  changes.append({'model':name,'macro_f1_in_domain':f['In-domain'],'macro_f1_cross_domain':f['Cross-domain'],
   'absolute_drop':f['In-domain']-f['Cross-domain'],'percentage_point_drop':100*(f['In-domain']-f['Cross-domain']),
   'relative_drop_percent':100*(f['In-domain']-f['Cross-domain'])/f['In-domain']})
 pd.DataFrame(changes).to_csv(out/'domain_changes.csv',index=False)
 error_rows=[];length_rows=[]
 for setting,r in stack_results.items():
  errors=r[~r.correct].copy();errors['error_type']=np.where(errors.labels==0,'false_positive','false_negative')
  errors.to_csv(out/f'errors_stacking_{setting}.csv',index=False)
  tn,fp,fn,tp=confusion_matrix(r.labels,r.prediction,labels=[0,1]).ravel()
  error_rows.append({'setting':SETTINGS[setting],'TN':tn,'FP':fp,'FN':fn,'TP':tp,'false_positive_rate':fp/(tn+fp),'false_negative_rate':fn/(tp+fn)})
  bins=pd.cut(r.text.str.split().str.len(),[-1,15,30,float('inf')],labels=['0-15 words','16-30 words','31+ words'])
  for group in bins.cat.categories:
   subset=r[bins==group]
   length_rows.append({'setting':SETTINGS[setting],'length_group':group,'n':len(subset),'offensive_support':int((subset.labels==1).sum()),'errors':int((~subset.correct).sum()),'error_rate':float((~subset.correct).mean())})
  if setting=='cross_domain':
   sample=pd.concat([errors[errors.error_type==kind].sample(n=min(10,len(errors[errors.error_type==kind])),random_state=SEED)
    for kind in ['false_positive','false_negative']]).sort_values(['error_type','id'])
   sample.to_csv(out/'cross_domain_review_sample.csv',index=False)
 pd.DataFrame(error_rows).to_csv(out/'error_summary_stacking.csv',index=False)
 pd.DataFrame(length_rows).to_csv(out/'errors_by_length_stacking.csv',index=False)
 # Plots compare methods holding the base-model pool fixed.
 names=['bert','hatebert','fbert','Hard voting (matched)','Soft voting (matched)','Stacking']
 fig,ax=plt.subplots(figsize=(10,5.5));x=np.arange(len(names));width=.36
 for offset,setting,color in [(-width/2,'In-domain','#236786'),(width/2,'Cross-domain','#a3b7c2')]:
  vals=metrics[(metrics.setting==setting)&(metrics.pool=='Matched A5')].set_index('model').loc[names,'macro_f1']
  bars=ax.bar(x+offset,vals,width,label=setting,color=color);ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
 ax.set_xticks(x,['BERT','HateBERT','fBERT','Hard voting','Soft voting','Stacking']);ax.set_ylim(0,1)
 ax.set_ylabel('Macro-F1 (0-1)');ax.set_xlabel('Model or ensemble strategy')
 ax.set_title('Same base-model pool, same OLID test set (860 messages)');ax.legend(frameon=False)
 ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(out/'matched_model_comparison.png',dpi=180);plt.close(fig)
 write_json(out/'verification.json',{'status':'passed','oof_checks':checks,'provided_metrics_recomputed':True,
  'meta_model_reload_predictions_identical':True,'test_labels_never_used_for_meta_fit':True,
  'no_test_based_selection':True,'prediction_rows_per_setting':860})
 print(primary.to_string(index=False),flush=True)
 print('Verified OOF provenance, fixed test rows, supplied metrics and saved meta-model reloads.',flush=True)
 return metrics

if __name__=='__main__':evaluate()
