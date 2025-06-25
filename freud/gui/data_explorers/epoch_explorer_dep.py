from freud.data_io.mne_based import read_digital_signals_mne
from freud.gui.sleep_monitor import SleepMonitor
from pictor import Pictor
from pictor.plotters.plotter_base import Plotter
from pictor.objects.signals.signal_group import SignalGroup, Annotation
from pictor.plugins import DialogUtilities
from roma import console

import os
import matplotlib.pyplot as plt
import numpy as np



class EpochExplorer(Pictor):
  """Dimensions:
  - objects (signal groups)
    - shortcuts: 'N' and 'P'
  - stages
    - shortcuts: 'h' and 'l'
  - epochs
    - shortcuts: 'k' and 'j'
  - channels
    - shortcuts: 'n' and 'p'
  """

  class Keys(Pictor.Keys):
    STAGES = 'StAgEs'
    EPOCHS = 'EpOcHs'
    CHANNELS = 'ChAnNeLs'

    STAGE_EPOCH_DICT = 'stage_epoch_dict'
    ANNO_KEY_GT_STAGE = 'stage Ground-Truth'
    MAP_DICT = 'Keys::map_dict'

  STAGE_KEYS = ('W', 'N1', 'N2', 'N3', 'R')


  def __init__(self, title='Epoch Explorer', figure_size=(10, 6)):
    # Call parent's constructor
    super(EpochExplorer, self).__init__(title, figure_size=figure_size)

    self.rhythm_plotter = self.add_plotter(RhythmPlotter(self))

    # Create dimensions for epochs and channels
    self.create_dimension(self.Keys.STAGES)
    self.create_dimension(self.Keys.EPOCHS)
    self.create_dimension(self.Keys.CHANNELS)

    # Set dimension
    self.set_to_axis(self.Keys.STAGES, self.STAGE_KEYS, overwrite=True)

  # region: Properties

  @property
  def selected_signal_group(self) -> SignalGroup:
    return self.get_element(self.Keys.OBJECTS)

  @property
  def selected_stage(self) -> str:
    return self.get_element(self.Keys.STAGES)

  @property
  def selected_channel_index(self):
    return self.get_element(self.Keys.CHANNELS)

  @property
  def selected_channel_name(self):
    sg = self.selected_signal_group
    c = self.selected_channel_index
    return sg.digital_signals[0].channels_names[c]

  @property
  def selected_signal(self):
    c = self.get_element(self.Keys.CHANNELS)
    epoch = self.get_element(self.Keys.EPOCHS)
    se = self.get_sg_stage_epoch_dict(self.selected_signal_group)
    return se[self.selected_stage][epoch][:, c]

  # endregion: Properties

  @classmethod
  def get_map_dict(cls, sg: SignalGroup):
    anno: Annotation = sg.annotations[cls.Keys.ANNO_KEY_GT_STAGE]

    def _init_map_dict(labels):
      map_dict = {}
      for i, label in enumerate(labels):
        if 'W' in label: j = 0
        elif '1' in label: j = 1
        elif '2' in label: j = 2
        elif '3' in label or '4' in label: j = 3
        elif 'R' in label: j = 4
        else: j = None
        map_dict[i] = j
        # console.supplement(f'{label} maps to {j}', level=2)
      return map_dict

    return sg.get_from_pocket(
      cls.Keys.MAP_DICT, initializer=lambda: _init_map_dict(anno.labels))


  @classmethod
  def get_sg_stage_epoch_dict(cls, sg: SignalGroup):
    def _init_sg_stage_epoch_dict():
      ds = sg.digital_signals[0]
      T = int(ds.sfreq * 30)
      # Get annotation
      anno: Annotation = sg.annotations[cls.Keys.ANNO_KEY_GT_STAGE]
      # Get reshaped tape
      tape = ds.data.reshape([-1, T, ds.data.shape[-1]])
      # Generate map_dict
      map_dict = cls.get_map_dict(sg)

      se_dict, cursor = {k: [] for k in cls.STAGE_KEYS}, 0
      for interval, anno_id in zip(anno.intervals, anno.annotations):
        sid = cls.STAGE_KEYS[map_dict[anno_id]]
        n = int((interval[-1] - interval[0]) / 30)
        for i in range(cursor, cursor + n): se_dict[sid].append(tape[i])
        cursor += n

      return se_dict

    return sg.get_from_pocket(
      cls.Keys.STAGE_EPOCH_DICT, initializer=_init_sg_stage_epoch_dict)


  def set_cursor(self, key: str, step: int = 0, cursor=None,
                 refresh: bool = False):
    super().set_cursor(key, step, cursor, False)

    sg: SignalGroup = self.get_element(self.Keys.OBJECTS)
    current_stage = self.get_element(self.Keys.STAGES)
    se = self.get_sg_stage_epoch_dict(sg)

    # Re-assign dimension
    if key in (self.Keys.OBJECTS, self.Keys.STAGES):
      # Set epoch dimension
      num_epochs = len(se[current_stage])
      self.set_to_axis(self.Keys.EPOCHS, list(range(num_epochs)),
                       overwrite=True)

      # Set channel dimension
      num_channels = se[current_stage][0].shape[-1]
      if key == self.Keys.OBJECTS:
        self.set_to_axis(self.Keys.CHANNELS, list(range(num_channels)),
                         overwrite=True)
        self.static_title = f'Epoch Explorer - {sg.label}'

    # Refresh if required
    if refresh: self.refresh()


  def _register_default_key_events(self):
    # Allow plotter shortcuts
    self.shortcuts.external_fetcher = self._get_plotter_shortcuts

    register_key = lambda btn, des, key, v: self.shortcuts.register_key_event(
      [btn], lambda: self.set_cursor(key, v, refresh=True),
      description=des, color='yellow')

    register_key('N', 'Next Signal Group', self.Keys.OBJECTS, 1)
    register_key('P', 'Previous Signal Group', self.Keys.OBJECTS, -1)

    register_key('L', 'Next Stage', self.Keys.STAGES, 1)
    register_key('H', 'Previous Stage', self.Keys.STAGES, -1)

    register_key('j', 'Next Epoch', self.Keys.EPOCHS, 1)
    register_key('k', 'Previous Epoch', self.Keys.EPOCHS, -1)

    register_key('n', 'Next Channel', self.Keys.CHANNELS, 1)
    register_key('p', 'Previous Channel', self.Keys.CHANNELS, -1)

    register_stage_key = lambda btn, i: self.shortcuts.register_key_event(
      [btn], lambda: self.set_cursor(self.Keys.STAGES, cursor=i, refresh=True),
      description=f'Select `{self.STAGE_KEYS[i]}` stage', color='yellow')
    for i, k in enumerate(['w', '1', '2', '3', 'r']): register_stage_key(k, i)

  def set_signal_groups(self, signal_groups):
    self.objects = signal_groups
    self.set_cursor(self.Keys.OBJECTS, cursor=0)


  @staticmethod
  def explore(signal_groups, title='EpochExplorer', figure_size=(10, 6),
              **kwargs):
    ee = EpochExplorer(title, figure_size)
    for k, v in kwargs.items():
      ee.rhythm_plotter.set(k, v, auto_refresh=False)
    ee.set_signal_groups(signal_groups)
    ee.show()



class RhythmPlotter(Plotter):

  def __init__(self, pictor):
    super().__init__(self.plot, pictor)
    self.explorer: EpochExplorer = pictor

    # Define settable attributes
    self.new_settable_attr('plot_wave', False, bool, 'Whether to plot wave')
    self.new_settable_attr('pctile_margin', 0.01, float, 'Percentile margin')
    self.new_settable_attr('t1', 0, float, 't1')
    self.new_settable_attr('t2', 30, float, 't2')
    self.new_settable_attr('min_freq', 1, float, 'Minimum frequency')
    self.new_settable_attr('max_freq', 20, float, 'Maximum frequency')
    self.new_settable_attr('column_norm', False, bool,
                           'Option to apply column normalization to spectrum')

    self.new_settable_attr('show_wave_threshold', True, bool,
                           'Option to show wave threshold')

    self.new_settable_attr('dev_mode', False, bool,
                           'Option to toggle developer mode')

    self.new_settable_attr('dev_arg', '5', str, 'Developer mode argument')

  # region: Properties

  # endregion: Properties

  # region: Plot Methods

  def plot(self, ax: plt.Axes):
    # Plot signal or spectrum
    if self.get('plot_wave'): suffix = self._plot_signal(ax)
    else: suffix = self._plot_spectrum(ax)

    # Set title
    stage = self.explorer.selected_stage
    channel_name = self.explorer.selected_channel_name
    title = f'[{stage}] {channel_name} {suffix}'
    ax.set_title(title)

  def _plot_spectrum(self, ax: plt.Axes):
    from scipy.signal import stft

    # Get signal
    s: np.ndarray = self.explorer.selected_signal

    # Compute the Short Time Fourier Transform (STFT)
    fs = self.explorer.selected_signal_group.digital_signals[0].sfreq
    f, t, Zxx = stft(s, fs=fs, nperseg=256)

    # Plot STFT result
    spectrum = np.abs((Zxx))
    if self.get('column_norm'):
      spectrum = spectrum / np.max(spectrum, axis=0, keepdims=True)

    # ax.pcolormesh(t, f, spectrum, vmin=0, shading='gouraud')
    ax.pcolormesh(t, f, spectrum, vmin=0)
    # ax.set_yscale('log')

    # Set styles
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Frequency [Hz]')

    # Set maximum frequency
    ax.set_xlim(t[0], t[-1])
    ymin, ymax = self.get('min_freq'), self.get('max_freq')
    ax.set_ylim(ymin, ymax)

    # Show wave threshold if required
    if self.get('show_wave_threshold'):
      ax.plot([0, 30], [4, 4], 'r:')
      ax.plot([0, 30], [8, 8], 'r:')
      ax.plot([0, 30], [13, 13], 'r:')

      ax2 = ax.twinx()
      ax2.set_yticks([(ymin + 4) / 2, 6, 10.5, (13 + ymax) / 2],
                     [r'$\delta$', r'$\theta$', r'$\alpha$', r'$\beta$'])
      ax2.set_ylim(ymin, ymax)

    return ''

  def _plot_auxiliaries(self, ax: plt.Axes):
    s: np.ndarray = self.explorer.selected_signal
    t = np.linspace(0, 30, num=len(s))

    ks = int(self.get('dev_arg'))
    x = np.convolve(s, [1/ks] * ks, 'same')

    ax.plot(t, x, 'r-')

  def _plot_signal(self, ax: plt.Axes):
    s: np.ndarray = self.explorer.selected_signal
    t = np.linspace(0, 30, num=len(s))

    # Plot signal
    ax.plot(t, s)

    # Plot auxiliary lines if required
    if self.get('dev_mode'): self._plot_auxiliaries(ax)

    # Set style
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Normalized Amplitude')

    ax.set_xlim(self.get('t1'), self.get('t2'))

    # .. Set ylim
    sg = self.explorer.selected_signal_group
    m = self.get('pctile_margin')
    chn_i = self.explorer.selected_channel_index
    ymin, ymax = [p[chn_i] for p in self.get_sg_pencentiles(sg, m)]
    ax.set_ylim(ymin, ymax)

    # Return stats
    return ''

  # endregion: Plot Methods

  # region: Interfaces

  def set_developer_arg(self, v):
    """Set developer argument"""
    self.set('dev_arg', v)
  da = set_developer_arg

  def register_shortcuts(self):
    self.register_a_shortcut('space', lambda: self.flip('plot_wave'),
                             'Toggle `plot_wave`')
    self.register_a_shortcut('equal', lambda: self.flip('column_norm'),
                             'Toggle `column_norm`')

    self.register_a_shortcut('h', lambda: self.move(-1), 'Move backward')
    self.register_a_shortcut('l', lambda: self.move(1), 'Move forward')

    self.register_a_shortcut('i', lambda: self.zoom(0.5), 'Zoom in')
    self.register_a_shortcut('o', lambda: self.zoom(2), 'Zoom out')

    self.register_a_shortcut('d', lambda: self.flip('dev_mode'),
                             'Toggle `dev_mode`')

  def zoom(self, multiplier):
    assert multiplier in (0.5, 2)
    t1, t2 = self.get('t1'), self.get('t2')
    t2 = t1 + (t2 - t1) * multiplier

    if t2 > 30: t1, t2 = t1 - (t2 - 30), 30
    t1 = max(0, t1)

    if t1 == self.get('t1') and t2 == self.get('t2'): return
    if t2 - t1 < 2: return

    self.set('t1', t1, auto_refresh=False)
    self.set('t2', t2, auto_refresh=False)
    self.refresh()

  def move(self, direction):
    assert direction in (-1, 1)
    t1, t2 = self.get('t1'), self.get('t2')

    d = (t2 - t1) * 0.5 * direction
    t1, t2 = t1 + d, t2 + d
    if t1 < 0 or t2 > 30: return

    if t1 == self.get('t1') and t2 == self.get('t2'): return

    self.set('t1', t1, auto_refresh=False)
    self.set('t2', t2, auto_refresh=False)
    self.refresh()

  # endregion: Interfaces

  # region: Processing Methods

  @classmethod
  def get_sg_pencentiles(cls, sg: SignalGroup, m):
    """m is percentile margin, should be in (0, 50)"""

    def _init_percentile():
      return [np.percentile(sg.digital_signals[0].data, q, axis=0)
              for q in (m, 100 - m)]

    key = f'percentile_{m}'
    return sg.get_from_pocket(key, initializer=_init_percentile)

  # endregion: Processing Methods



if __name__ == '__main__':
  from freud.gui.data_explorers.epoch_explorer import EpochExplorer
  from roma import finder
  from roma import io

  # Set directories
  data_dir = r'../../../data/'
  data_dir += 'sleepeason1'

  prefix = ['', 'sleepedfx', 'ucddb', 'rrsh'][1]
  pattern = f'{prefix}*.sg'

  # Select .sg files
  sg_file_list = finder.walk(data_dir, pattern=pattern)

  signal_groups = []
  for path in sg_file_list:
    sg = io.load_file(path, verbose=True)
    signal_groups.append(sg)

  # Visualize signal groups
  EpochExplorer.explore(signal_groups, plot_wave=True)


