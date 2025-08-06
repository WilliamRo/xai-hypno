"""This module is built upon the idea of 2001finelli and 2005gennaro, for
   exploring the heterogeneity of individuals based on nonREM signal spectra
   across different derivations.
"""
from freud.gui.data_explorers.explorer_base import ExplorerBase
from pictor import Pictor
from pictor.plotters.plotter_base import Plotter
from pictor.objects.signals.signal_group import SignalGroup, Annotation
from roma import console

import matplotlib.pyplot as plt
import numpy as np



class SpectraExplorer(ExplorerBase):

  def __init__(self, channels, title='Spectra Explorer', figure_size=(10, 6),
               meta=None, **kwargs):
    super().__init__(title, figure_size, kwargs.get('tool_bar', False))

    self.channels = channels
    self.meta = meta

    plotter_cls = kwargs.get('plotter_class', SpectraViewer)
    self.rhythm_plotter = self.add_plotter(plotter_cls(self))

  # region: Properties

  # endregion: Properties

  # region: Public Methods

  # region: - Entry

  @staticmethod
  def explore(signal_groups, channels, title='Spectra Explorer',
              figure_size=(10, 6), plotter_cls=None, dont_show=False,
              meta=None, **kwargs):
    if plotter_cls is None: plotter_cls = SpectraViewer
    se = SpectraExplorer(channels, title, figure_size, plotter_cls=plotter_cls,
                         meta=meta)

    for k, v in kwargs.items():
      se.rhythm_plotter.set(k, v, auto_refresh=False)

    se.objects = signal_groups

    if not dont_show: se.show()
    else: return se

  # endregion: - Entry

  # endregion: Public Methods



class SpectraViewer(Plotter):

  COLORS = ['#d76033', '#74b256', '#608bc6']

  def __init__(self, pictor, **kwargs):
    super().__init__(self.plot, pictor)
    self.explorer: SpectraExplorer = pictor

    # Define settable attributes
    self.new_settable_attr('stages', 'N1,N2,N3', str, 'Sleep stages')
    self.new_settable_attr('f_min', 8., float, 'Minimal frequency')
    self.new_settable_attr('f_max', 15.5, float, 'Maximal frequency')
    self.new_settable_attr('f_reso', 0.25, float, 'Frequency resolution')
    self.new_settable_attr('zscore', True, bool,
                           'Whether to transform power to z-score')
    self.new_settable_attr('log', False, bool,
                           'Whether to use logarithm')

    self.new_settable_attr('dev_arg', '0', str, 'Developer mode argument')

    self.new_settable_attr('bool_a', False , bool, 'Boolean A')
    self.new_settable_attr('bool_b', False , bool, 'Boolean A')

  # region: Properties

  @property
  def stages(self):
    _stages = self.get('stages').split(',')
    for sk in _stages: assert sk in self.explorer.STAGE_KEYS
    return _stages

  @property
  def freq_range(self): return self.get('f_min'), self.get('f_max')

  @property
  def freq_resolution(self): return self.get('f_reso')

  @property
  def signal_groups(self):
    if isinstance(self.explorer.objects[0], SignalGroup):
      return self.explorer.objects
    sg_list = []
    for sg_group in self.explorer.objects: sg_list.extend(sg_group)
    return sg_list

  # endregion: Properties

  # region: Plot Methods

  def plot(self, fig: plt.Figure):
    """TODO: Currently only single signal group is supported."""

    # (1) Plot spectrum
    obj = self.explorer.selected_signal_group
    if isinstance(obj, SignalGroup):
      ax: plt.Axes = fig.add_subplot(111)
      self.plot_one_psg(ax, obj)
    else:
      n = len(obj)
      for i, sg in enumerate(obj):
        assert isinstance(sg, SignalGroup)
        ax: plt.Axes = fig.add_subplot(1, n, i + 1)
        self.plot_one_psg(ax, sg)

    # (-1) Set style
    # fig.suptitle(f'{label}')


  def _get_channel_style(self, ck):
    line_style = '-'
    if 'M1' in ck or 'A1' in ck: line_style = ':'

    if any([k in ck for k in ('F3', 'F4')]): return self.COLORS[0], line_style
    if any([k in ck for k in ('C3', 'C4')]): return self.COLORS[1], line_style
    if any([k in ck for k in ('O1', 'O2')]): return self.COLORS[2], line_style

    if 'Fpz' in ck: return self.COLORS[1], line_style
    if 'Pz' in ck: return self.COLORS[2], line_style

  def plot_one_psg(self, ax: plt.Axes, sg: SignalGroup):
    # (1) Calculate spectrum
    freqs, spectra_dict, duration = self.get_channel_spectra(sg)

    if freqs is None: return

    # (2) Visualize
    for ck in self.explorer.channels:
      spectrum = spectra_dict[ck]

      if self.get('log'):
        spectrum = 10 * np.log10(spectrum)

      if self.get('zscore'):
        spectrum = (spectrum - np.mean(spectrum)) / np.std(spectrum)

      color, line_style = self._get_channel_style(ck)
      lw = 3 if line_style == ':' else 2
      ax.plot(freqs, spectrum, line_style, color=color,
              linewidth=lw, label=ck, alpha=0.8)

    # (3) Set styles
    ax.set_xlabel('Frequency (Hz)')
    y_label = 'Power Spectral Density (dB/Hz)'

    # Show sigma band using rectangle
    if self.get('zscore'): ax.fill_between(
      [11, 16], [-3, -3], [3, 3], color='gray', alpha=0.1, zorder=-999)

    ax.set_xlim(self.freq_range)
    if self.get('zscore'):
      ax.set_ylim([-3., 3.])
      y_label = 'Normalized PSD'

    if self.get('log'): y_label += ' (log)'

    ax.set_ylabel(y_label)

    ax.grid(True)
    ax.legend()

    title = f'{sg.label} ({self.get("stages")}, {duration:.1f} h)'
    if self.explorer.meta is not None:
      info_dict: dict = self.explorer.meta[sg.label]

      for k, v in info_dict.items():
        if k == 'info': title += f', {v}'
        else: title += f', {k}={v}'

    ax.set_title(title)

  def preload(self):
    with self.explorer.busy('Preloading ...'):
      N = len(self.signal_groups)
      for i, sg in enumerate(self.signal_groups):
        console.print_progress(i, N)
        self.get_channel_spectra(sg)
      console.show_status('Preloading completed.')
  pl = preload

  def get_channel_spectra(self, sg: SignalGroup):
    from scipy import signal

    # (0) Fetch settings
    f_min, f_max, f_reso = [self.get(k) for k in ('f_min', 'f_max', 'f_reso')]
    fs = sg.digital_signals[0].sfreq
    nfft = int(fs / f_reso)

    # (1) Return from pocket if exists
    key = (sg.label, f_min, f_max, f_reso, self.get('stages'),
           ','.join(self.explorer.channels))

    if self.in_pocket(key): return self.get_from_pocket(key)

    # (2) Calculate spectra_dict if not exists
    # (2.1) Gather channel_epoch_dict
    channel_epoch_dict = {ck: [] for ck in self.explorer.channels}
    se = self.explorer.get_sg_stage_epoch_dict(sg)
    for ck in self.explorer.channels:
      for sk in self.stages:
        ci = sg.channel_names.index(ck)
        channel_epoch_dict[ck].extend([s[:, ci] for s in se[sk]])

    # (2.2) Calculate spectrum for each epoch
    spectrum_list_dict = {ck: [] for ck in self.explorer.channels}
    freqs = None
    for ck in self.explorer.channels:
      for s in channel_epoch_dict[ck]:
        # win_len = 8 * fs
        # nperseg = min(nfft, len(s))
        nperseg = 2.0 * fs
        freqs, psd = signal.welch(s, fs, nperseg=nperseg, nfft=nfft)

        # (2.2.1) Truncate psd to range (f_min, f_max)
        freq_indices = (freqs >= f_min) & (freqs <= f_max)
        freqs, psd = freqs[freq_indices], psd[freq_indices]

        spectrum_list_dict[ck].append(psd)

    # (2.3) Merge spectra
    duration = len(spectrum_list_dict[ck]) * 30 / 3600
    spectra_dict = {ck: np.mean(spectrum_list_dict[ck], axis=0)
                    for ck in self.explorer.channels}

    self.put_into_pocket(key, (freqs, spectra_dict, duration), local=True)
    return freqs, spectra_dict, duration

  # endregion: Plot Methods

  # region: APIs

  def register_shortcuts(self):
    self.register_a_shortcut('a', lambda: self.flip('bool_a'),
                             'Toggle `bool_a`')
    self.register_a_shortcut('b', lambda: self.flip('bool_b'),
                             'Toggle `bool_a`')
    self.register_a_shortcut('l', lambda: self.flip('log'),
                             'Toggle `log`')
    self.register_a_shortcut('z', lambda: self.flip('zscore'),
                             'Toggle `zscore`')

    for k, sk in zip(['w', '1', '2', '3', 'r'], ['W', 'N1', 'N2', 'N3', 'R']):
      self.register_a_shortcut(k, lambda _sk=sk: self.toggle_stage(_sk),
                               f'Toggle stage `{sk}`')

    for k, sk in zip(['W', 'exclam', 'at', 'numbersign', 'R'],
                     ['W', 'N1', 'N2', 'N3', 'R']):
      self.register_a_shortcut(k, lambda _sk=sk: self.toggle_stage(_sk, True),
                               f'Toggle stage `{sk}` only')

  def list_bands(self):
    console.show_info('Band range according to AASM:')
    console.supplement('delta: 0.5-4 Hz', level=2)
    console.supplement('theta: 4-8 Hz', level=2)
    console.supplement('alpha: 8-12 Hz', level=2)
    console.supplement('beta: 12-30 Hz', level=2)
    console.supplement('sigma: 11-16 Hz', level=2)
  lsbd = list_bands

  def toggle_stage(self, sk: str, only=False):
    STAGES = ('W', 'N1', 'N2', 'N3', 'R')
    assert sk in STAGES

    if only:
      self.set('stages', sk)
      return

    stages = self.get('stages').split(',')
    if sk in stages: stages.remove(sk)
    else: stages.append(sk)
    stages = [sk for sk in STAGES if sk in stages]
    self.set('stages', ','.join(stages))

  # endregion: APIs
