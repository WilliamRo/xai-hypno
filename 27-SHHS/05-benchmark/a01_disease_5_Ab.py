import os
os.environ['BYPASS_TF'] = '1'

from pictor.xomics.omix import Omix



# -----------------------------------------------------------------------------
#  O Configuration
# -----------------------------------------------------------------------------
OMIX_ALL_PATH = r'<input-your-path>\SHHS-all-250724.omix'
OMIX_ALL_PATH = r"E:\data\shhs\shhs_omix\SHHS-all-250724.omix"
KEYS = ['CVD', 'CHD', 'Insomnia', 'Hypertension', 'Diabetes']

# -----------------------------------------------------------------------------
#  I Load data
# -----------------------------------------------------------------------------
omix = Omix.load(OMIX_ALL_PATH)

key = KEYS[3]
omix = omix.set_targets(key, return_new_omix=True)
omix.data_name += f'_{key}'
omix.report()

omix.show_in_explorer()
