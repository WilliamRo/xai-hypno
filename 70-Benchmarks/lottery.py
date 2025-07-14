import numpy as np

from pictor.xomics.omix import Omix


n_positive = 1589
n_negative = 6844

# Configuration
n_samples = n_positive + n_negative
n_features = 2101

targets = np.zeros(n_samples, dtype=np.int32)
targets[:n_positive] = 1  # Positive samples

Omix.gen_psudo_omix(n_samples, n_features, targets).show_in_explorer()
