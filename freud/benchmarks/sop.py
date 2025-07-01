from freud.benchmarks.algorithm import Algorithm
from freud.datasets.dataset_base import HypnoDataset
from roma import console, Nomear
from pictor.xomics.omix import Omix
from pictor.xomics.evaluation.pipeline import Pipeline

import os


class BenchmarkSOP(Nomear):
  """A base class of benchmarking SOPs (Standard Operating Procedures)"""

  def __init__(self, hypno_data: HypnoDataset, model: Algorithm,
               test_bench='pipeline', overwrite=False, study_name=None):
    self.hypno_data = hypno_data
    self.model = model
    self.overwrite = overwrite
    self.study_name = study_name

    assert test_bench in ('pipeline', )
    self.test_bench = test_bench


  @property
  def omix_path(self):
    return os.path.join(self.hypno_data.omix_dir, f'{self.study_name}.omix')


  def generate_omix(self, target_key, target_labels, data_name='Omix') -> Omix:
    if not self.overwrite and self.study_name is not None:
      if os.path.exists(self.omix_path): return Omix.load(self.omix_path)

    # Extract features
    features, pkg = self.model.extract_features()
    feature_names = pkg['feature_names']

    # Generate targets
    meta = self.hypno_data.load_meta()
    targets = [meta[sg_lb][target_key] for sg_lb in self.hypno_data.sg_labels]
    if len(target_labels) > 1:
      lb2int = {lb: i for i, lb in enumerate(target_labels)}
      targets = [lb2int[y] for y in targets]

    # Wrap the data into Omix
    omix = Omix(features=features,
                targets=targets,
                feature_labels=feature_names,
                sample_labels=self.hypno_data.sg_labels,
                target_labels=target_labels,
                data_name=data_name)

    # Save omix if necessary
    if self.study_name is not None: omix.save(self.omix_path)

    return omix


  def pipeline_test_bench(self,
                          omix: Omix,
                          sf_config: list,
                          ml_config: list,
                          report = True,
                          plot_matrix = False,
                          **kwargs):

    # (0) Create pipeline
    pi = Pipeline(omix, ignore_warnings=1, save_models=0)

    # (1) Feature selection
    for key, config in sf_config:
      kwargs = {'repeats': 1, 'nested': 1, 'show_progress': 1}
      kwargs.update(config)
      pi.create_sub_space(key, **kwargs)

    # (2) Machine learning
    for key, config in ml_config:
      kwargs = {'repeats': 1, 'nested': 1, 'show_progress': 1, 'verbose': 1}
      kwargs.update(config)
      pi.fit_traverse_spaces(key, **kwargs)

    # (3) Finalize
    if report:
      pi.report()
      if plot_matrix: pi.plot_matrix()

    return pi



if __name__ == '__main__':
  pass
