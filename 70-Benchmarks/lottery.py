from pictor.xomics.omix import Omix



# Configuration
n_samples = 200
# n_features = 1000
n_features = 5000

Omix.gen_psudo_omix(n_samples, n_features).show_in_explorer()
