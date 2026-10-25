"""Record differences between the supplied group model pools without changing them."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

def audit(root='.'):
 root=Path(root);out=root/'stacking_results'
 labels=pd.read_csv(root/'provided/ines/a4_component_predictions.csv')
 rows=[]
 for setting,suffix in [('in_domain','in'),('cross_domain','cross')]:
  soft=pd.read_csv(root/'provided/helian'/f'results_soft_voting_{setting}.csv')
  assert labels[['id','text','labels']].equals(soft[['id','text','labels']])
  for model in ['bert','hatebert','fbert']:
   old=labels[f'{model}_{suffix}'].to_numpy()
   supplied=(soft[f'{model}_prob'].to_numpy()>=.5).astype(int)
   row={'setting':setting,'model':model,'rows':len(old),
    'ines_vs_helian_prediction_disagreements':int((old!=supplied).sum()),
    'ines_component_macro_f1':f1_score(labels.labels,old,average='macro'),
    'helian_component_macro_f1':f1_score(labels.labels,supplied,average='macro')}
   ours=root/f'{model}_probs_{setting}.csv'
   if ours.exists():
    own=pd.read_csv(ours)
    assert own[['id','text','labels']].equals(soft[['id','text','labels']])
    row['a5_matched_vs_helian_prediction_disagreements']=int((own.prediction!=supplied).sum())
    row['a5_matched_vs_helian_max_probability_difference']=float(np.max(np.abs(own.probability_offensive-soft[f'{model}_prob'])))
   rows.append(row)
 result=pd.DataFrame(rows);result.to_csv(out/'supplied_pool_audit.csv',index=False)
 print(result.to_string(index=False))
 return result

if __name__=='__main__':audit()
