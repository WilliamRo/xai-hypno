import os
os.environ['BYPASS_TF'] = '1'

from pictor.xomics.omix import Omix
from pictor.xomics.evaluation.pipeline import Pipeline
from roma import console, io



# -----------------------------------------------------------------------------
#  O Configuration
# -----------------------------------------------------------------------------
WORK_DIR = r'..\..\..\data\shhs'
OMIX_ALL_PATH = os.path.join(WORK_DIR, r'shhs_omix\SHHS-all-250724.omix')

KEYS = ['CVD', 'CHD', 'Insomnia', 'Hypertension', 'Diabetes']
key = KEYS[0]

SEED = 42

PKG_PATH = os.path.join(WORK_DIR, f'shhs_temp_{SEED}_{key}.pkg')
overwrite = 0
# -----------------------------------------------------------------------------
#  I Load data
# -----------------------------------------------------------------------------
omix = Omix.load(OMIX_ALL_PATH)

omix = omix.set_targets(key, return_new_omix=True)
omix.data_name += f'_{key}'

console.section('Whole Dataset'), omix.report()
omix.report()

# omix.show_in_explorer()
# -----------------------------------------------------------------------------
#  II Split data, train and divide test set
# -----------------------------------------------------------------------------
if os.path.exists(PKG_PATH) and not overwrite:
  pkg, train_omix, test_omix = io.load_file(PKG_PATH, True)
else:
  train_omix, test_omix = omix.split(
    5, 5, random_state=SEED, data_labels=('train_set', 'test_set'))

  console.section('Train Set'), train_omix.report()
  console.section('Test Set'), test_omix.report()

  console.section('Building model on train set')

  p = Pipeline(train_omix, save_models=True)
  p.create_sub_space('ucp', k=200, t=0.9, nested=1)
  p.fit_traverse_spaces('lr', nested=1)

  p.report()
  pkg = p.evaluate_best_pipeline(test_omix, omix_refit=train_omix)

  # Save package
  io.save_file((pkg, train_omix, test_omix), PKG_PATH, verbose=True)

# Report AUC on test set
pkg.report(omix=test_omix, show_signature=0)
scores = pkg.predict_proba(test_omix.features)

# -----------------------------------------------------------------------------
#  III Calculate effective size and plot
# -----------------------------------------------------------------------------
console.section('Calculating effective size')

import statsmodels.api as sm
import pandas as pd
import numpy as np

# Prepare data
data = {'Signature': scores[:, 1], key: test_omix.targets}
df = pd.DataFrame(data)

# Fit logistic regression
X = df[['Signature']]
X = sm.add_constant(X)
y = df[key]

model = sm.Logit(y, X).fit()
print(model.summary())

#  Odds ratio and 95% CI
params = model.params
conf = model.conf_int()
odds_ratio = np.exp(params['Signature'])
ci_lower, ci_upper = np.exp(conf.loc['Signature'])

print(f"Odds Ratio: {odds_ratio:.2f} (95% CI {ci_lower:.2f}-{ci_upper:.2f})")
