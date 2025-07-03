from pictor.xomics import Omix

import numpy as np



# Define Omix-A
features_A = np.array([
  [0.1, 0.2],
  [0.3, 0.4],
  [0.5, 0.6],
])
targets_A = [0, 0, 0]
omix_A = Omix(features_A, targets_A,
              feature_labels=['feature1', 'feature2'],
              sample_labels=['sample1', 'sample2', 'sample3'],
              target_labels=['Negative', 'Positive'],
              data_name='A')

# Define Omix-B
features_B = np.array([
  [0.7],
  [0.8],
  [0.9],
])
targets_B = [0, 0, 0]
omix_B = Omix(features_B, targets_B,
              feature_labels=['feature3'],
              sample_labels=['sample1', 'sample2', 'sample3'],
              target_labels=['Negative', 'Positive'],
              data_name='B')

# Define Omix-C
features_C = np.array([
  [1.1, 1.2, 1.3],
])
targets_C = [1]
omix_C = Omix(features_C, targets_C,
              feature_labels=['feature1', 'feature2', 'feature3'],
              sample_labels=['sample4'],
              target_labels=['Negative', 'Positive'],
              data_name='C')

omix_A.report()
omix_B.report()
omix_C.report()

# Merge Omix-A and Omix-B
omix_AB = omix_A * omix_B
omix_AB.report()

# Merge Omix-AB and Omix-C
omix_ABC = omix_AB + omix_C
omix_ABC.report()
