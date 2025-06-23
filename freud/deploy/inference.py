from freud.talos_utils.slp_config import SleepConfig
from freud.talos_utils.slp_set import SleepSet
from pictor.objects.signals.signal_group import Annotation, SignalGroup

import importlib.util
from tframe.utils import console


def stage_alpha(sg: SignalGroup, t_file_path: str) -> Annotation:
  # (0) Load module from t-file
  # Load model
  module_name = 'this_name_does_not_matter'
  spec = importlib.util.spec_from_file_location(module_name, t_file_path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)

  th: SleepConfig = mod.core.th
  th.developer_code += 'deactivate'

  # Execute main to load basic module settings
  mod.main(None)

  # (1) Prepare data set
  # Get displayed signal_group
  ds = SleepSet(signal_groups=[sg])

  # Set CHANNELS for extracting tapes during configuration
  # TODO: channels
  # channels = '1,2'
  channel_list = [c for c, _, _ in sg.name_tick_data_list]
  ds.CHANNELS = {f'{i + 1}': k for i, k in enumerate(channel_list)}
  # th.data_config = f'whatever {channels}'

  ds.configure()
  ds = ds.extract_data_set(include_targets=False)

  # (2) Prepare model
  from tframe import Classifier

  model: Classifier = th.model()
  preds = model.classify(ds, batch_size=128, verbose=True)
  model.shutdown()

  # (3) Set preds to annotations
  stage_permutation = '1,2,3,4,5'
  stage_map = {
    int(str_i) - 1: i for i, str_i in enumerate(stage_permutation.split(','))}

  t0 = sg.digital_signals[0].ticks[0]
  intervals = [(t0 + i * 30, t0 + (i + 1) * 30) for i, _ in enumerate(preds)]
  annotations = [stage_map[i] for i in preds]
  anno = Annotation(intervals, annotations, labels=SleepSet.AASM_LABELS)
  return anno


def compare(sg:SignalGroup, pred_anno: Annotation,show_confusion_matrix=True):
  import numpy as np
  from freud.talos_utils.slp_set import SleepSet


  cm = None
  gt_anno = sg.annotations['stage Ground-Truth']
  map_dict=SleepSet.get_map_dict(sg)
  # target=map_dict.get(gt_anno.annotations)
  target=[map_dict.get(annotation) for annotation in gt_anno.annotations]
  labels=[]
  for i,interval in enumerate(gt_anno.intervals):
    index_start = int(interval[0])
    index_end = int(interval[1])
    index_len = int((index_end - index_start) / 30)
    labels.append( [target[i]] * index_len)
    # self.data = self.signal[:, index_start * 100:index_end * 100]
  gt_labels=np.concatenate(labels)

  labels = []
  for i, interval in enumerate(pred_anno.intervals):
    index_start = int(interval[0])
    index_end = int(interval[1])
    index_len = int((index_end - index_start) / 30)
    labels.append([pred_anno.annotations[i]] * index_len)
  pred_labels = np.concatenate(labels)
  from tframe.utils.maths.confusion_matrix import ConfusionMatrix
  cm = ConfusionMatrix(
    num_classes=5,
    class_names=['W','1','2','3','R'])
  cm.fill(pred_labels, gt_labels)
  if show_confusion_matrix:
    console.show_info('Confusion Matrix:')
    # console.write_line(cm.matrix_table(kwargs.get('cell_width', None)))
  # console.show_info(f'Evaluation Result ({data_set.name}):')
  console.write_line(cm.make_table(
    decimal=4, class_details=True))

  return cm

if __name__=='__main__':
  from tframe.utils import console

  show_confusion_matrix=True
  pred_labels=[1,4,0,1,1]
  gt_labels=[1,2,3,4,0]
  from tframe.utils.maths.confusion_matrix import ConfusionMatrix
  cm = ConfusionMatrix(
    num_classes=5,
    class_names=['W','1','2','3','R'])
  cm.fill(pred_labels,gt_labels)
  if show_confusion_matrix:
    console.show_info('Confusion Matrix:')
    # console.write_line(cm.matrix_table(kwargs.get('cell_width', None)))
  # console.show_info(f'Evaluation Result ({data_set.name}):')
  console.write_line(cm.make_table(
    decimal=4, class_details=True))
