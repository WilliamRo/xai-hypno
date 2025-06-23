import numpy as np

from pictor.plotters import Monitor
from roma import console



class SleepMonitor(Monitor):

  def register_shortcuts(self):
    super(SleepMonitor, self).register_shortcuts()

    self.register_a_shortcut('O', self.pictor.open,
                             description='Open a .edf file')

  # region: Auto Sttaging

  def stage(self,
            channels: str,
            t_file_path: str = None,
            model_name: str = None,
            stage_permutation: str = None):
    """Auto-stage current PSG record using pretrained talos model"""
    from pictor.objects.signals.signal_group import SignalGroup, Annotation
    from freud.talos_utils.slp_config import SleepConfig
    from freud.talos_utils.slp_set import SleepSet

    import importlib.util

    if self._selected_signal is None:
      console.show_status(' ! No signal group found.')
      return

    # (0) Load module from t-file
    if t_file_path is None:
      t_file_path = self.pictor.load_file_dialog('Please select a t-file')
      if t_file_path == '': return
    module_name = 'this_name_does_not_matter'
    spec = importlib.util.spec_from_file_location(module_name, t_file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    th: SleepConfig = mod.core.th
    th.developer_code += 'deactivate'

    # Execute main to load basic module settings
    mod.main(None)
    th.use_batch_mask = False

    # (1) Prepare data set
    # Get displayed signal_group
    sg: SignalGroup = self._selected_signal
    ds = SleepSet(signal_groups=[sg])

    # Set CHANNELS for extracting tapes during configuration
    ds.CHANNELS = {f'{i}': k for i, k in enumerate(self.channel_list)}
    th.data_config = f'whatever {channels}'
    ds.configure()

    if th.use_rnn: ds = ds.extract_seq_set(include_targets=False)
    else: ds = ds.extract_data_set(include_targets=False)

    # (2) Prepare model
    from tframe import Classifier, context

    # Do some cleaning
    context.logits_tensor_dict = {}

    if model_name is None: model_name = mod.model_name
    model: Classifier = th.model()
    batch_size = 1 if th.use_rnn else 128
    preds = model.classify(ds, batch_size=batch_size, verbose=True)
    if th.use_rnn: preds = np.concatenate(preds)
    model.shutdown()

    # (3) Set preds to annotations
    if stage_permutation is None: stage_permutation = '1,2,3,4,5'
    stage_map = {
      int(str_i) - 1: i for i, str_i in enumerate(stage_permutation.split(','))}

    t0 = sg.digital_signals[0].ticks[0]
    intervals = [(t0 + i * 30, t0 + (i + 1) * 30) for i, _ in enumerate(preds)]
    annotations = [stage_map[i] for i in preds]
    sg.annotations[f'stage {model_name}'] = Annotation(
      intervals, annotations, labels=SleepSet.AASM_LABELS)

    print() # this is for creating a new line
    console.show_status(f'`{model_name}` set to sg.Annotation')

    self.toggle_annotation('stage', model_name, force_on=True)

  # endregion: Auto Staging



if __name__ == '__main__':
  from tframe import console
  from freud.talos_utils.sleep_sets.sleepedfx import SleepEDFx

  console.suppress_logging()
  data_dir = r'E:\xai-sleep\data\sleepedfx'

  preprocess = 'trim;iqr'
  ds = SleepEDFx.load_as_sleep_set(data_dir, overwrite=0, preprocess=preprocess)

  freud = ds.show(return_freud=True)
  m: SleepMonitor = freud.monitor
  m._selected_signal = freud.objects[0]
  t_file_path = r'E:\xai-sleep\08-FNN\01_cnn_v1\checkpoints\0315_cnn_v1(16-s16-32-s32-64)\0315_cnn_v1(16-s16-32-s32-64).py'
  m.stage('1,2', t_file_path)
  m.stage('1,2', t_file_path)
