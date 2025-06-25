from .explorer_base import ExplorerBase
from pictor import Pictor
from pictor.plotters.plotter_base import Plotter
from pictor.objects.signals.signal_group import SignalGroup, Annotation

import matplotlib.pyplot as plt
import numpy as np



class EpochExplorer(ExplorerBase):
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

  def __init__(self, title='Epoch Explorer', figure_size=(10, 6), **kwargs):
    # Call parent's constructor
    super(EpochExplorer, self).__init__(title, figure_size=figure_size)

    plotter_cls = kwargs.get('plotter_class', RhythmPlotter)

    self.rhythm_plotter = self.add_plotter(plotter_cls(self))
    if kwargs.get('add_layer_2', False):
      self.rhythm_plotter_2 = self.add_plotter(plotter_cls(self, layer=2))

    # Create dimensions for epochs and channels
    self.create_dimension(self.Keys.STAGES)
    self.create_dimension(self.Keys.EPOCHS)
    self.create_dimension(self.Keys.CHANNELS)

    # Set dimension
    self.set_to_axis(self.Keys.STAGES, self.STAGE_KEYS, overwrite=True)

    # Remove Escape shortcut
    self.shortcuts._library.pop('Escape')

  # region: Properties

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

  def set_cursor(self, key: str, step: int = 0, cursor=None,
                 refresh: bool = False):
    super().set_cursor(key, step, cursor, False)

    sg: SignalGroup = self.get_element(self.Keys.OBJECTS)
    se = self.get_sg_stage_epoch_dict(sg)

    # Get selected stage
    if key == self.Keys.OBJECTS and len(se) < 5:
      self.set_to_axis(self.Keys.STAGES, list(se.keys()), overwrite=True)

    current_stage = self.get_element(self.Keys.STAGES)

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

    register_key('greater', 'Next Plotter', self.Keys.PLOTTERS, 1)
    register_key('less', 'Previous Plotter', self.Keys.PLOTTERS, -1)

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
              add_layer_2=False, plotter_cls=None, dont_show=False, **kwargs):
    if plotter_cls is None: plotter_cls = RhythmPlotter
    ee = EpochExplorer(title, figure_size, add_layer_2=add_layer_2,
                       plotter_class=plotter_cls)
    for k, v in kwargs.items():
      ee.rhythm_plotter.set(k, v, auto_refresh=False)
    ee.set_signal_groups(signal_groups)
    if not dont_show: ee.show()
    else: return ee



class RhythmPlotter(Plotter):

  def __init__(self, pictor, **kwargs):
    super().__init__(self.plot, pictor)
    self.explorer: EpochExplorer = pictor

    # Define settable attributes
    self.new_settable_attr('plot_wave', True, bool, 'Whether to plot wave')
    self.new_settable_attr('pctile_margin', 0.01, float, 'Percentile margin')
    self.new_settable_attr('t1', 0, float, 't1')
    self.new_settable_attr('t2', 30, float, 't2')
    self.new_settable_attr('min_freq', 0.5, float, 'Minimum frequency')
    self.new_settable_attr('max_freq', 20, float, 'Maximum frequency')
    self.new_settable_attr('column_norm', False, bool,
                           'Option to apply column normalization to spectrum')

    self.new_settable_attr('show_wave_threshold', True, bool,
                           'Option to show wave threshold')

    self.new_settable_attr('dev_mode', False, bool,
                           'Option to toggle developer mode')
    self.new_settable_attr('summit', False, bool,
                           'Option to toggle summit visualization')
    self.new_settable_attr('stft', True, bool,
                           'Option to plot STFT instead of FT spectrum')
    self.new_settable_attr('welch', True, bool,
                           "Option to plot Welch's periodogram")

    self.new_settable_attr('dev_arg', '32', str, 'Developer mode argument')

    self.new_settable_attr('filter', False, bool, 'Whether to filter signal')
    self.new_settable_attr('filter_arg', '0.3,35', str,
                           'Arguments for filtering signals')

    self.new_settable_attr('ymax', None, float, 'ymax in Welch plot')

    # Set configs
    self.configs = kwargs

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

  @staticmethod
  def welch_spectrogram(x, fs, window_size, overlap, nperseg=None):
    """TODO
    Calculate spectrogram using Welch's method

    Parameters:
    - x: input signal
    - fs: sampling frequency
    - window_size: size of sliding window
    - overlap: overlap between windows
    - nperseg: segment length for Welch's method (default: window_size//8)
    """
    from scipy import signal

    if nperseg is None:
      nperseg = window_size // 8

      # Calculate step size
    step = window_size - overlap

    # Initialize time and frequency arrays
    num_windows = (len(x) - window_size) // step + 1
    frequencies, _ = signal.welch(x[:window_size], fs, nperseg=nperseg)
    times = np.arange(num_windows) * step / fs

    # Initialize spectrogram array
    spectrogram = np.zeros((len(frequencies), num_windows))

    # Calculate PSD for each window
    for i in range(num_windows):
      start = i * step
      end = start + window_size
      segment = x[start:end]
      frequencies, psd = signal.welch(segment, fs, nperseg=nperseg)
      spectrogram[:, i] = psd

    return frequencies, times, spectrogram

  def _get_spectrum(self, s, ymin=None, ymax=None):
    from scipy.signal import stft

    if self.configs.get('layer', 1) == 2:
      x = self._low_freq_signal(s)
      s = s - x
    elif self.get('filter'):
      s = self._butter_filt(s)

    # Compute the Short Time Fourier Transform (STFT)
    fs = self.explorer.selected_signal_group.digital_signals[0].sfreq

    if self.get('welch'):
      f, t, Zxx = stft(s, fs=fs, nperseg=256)
      # TODO
      # f, t, Zxx = self.welch_spectrogram(
      #   s, fs, 256, 128, 256)
    else:
      f, t, Zxx = stft(s, fs=fs, nperseg=256)

    # Plot STFT result
    spectrum = np.abs((Zxx))

    # Cut value
    if ymin is None: ymin = self.get('min_freq')
    if ymax is None: ymax = self.get('max_freq')
    h_mask = f > ymin
    f, spectrum = f[h_mask], spectrum[h_mask]
    l_mask = f < ymax
    f, spectrum = f[l_mask], spectrum[l_mask]

    if self.get('column_norm'):
      spectrum = spectrum / np.max(spectrum, axis=0, keepdims=True)

    return f, t, spectrum

  def _calc_dominate_freq_curve_v1(self, s: np.ndarray, ymin=None, ymax=None):
    f, secs, spectrum = self._get_spectrum(s, ymin, ymax)
    dom_f = np.sum(f[..., np.newaxis] * spectrum, axis=0) / np.sum(
      spectrum, axis=0)
    return f, secs, spectrum, dom_f

  def _calc_dominate_freq_curve_v2(self, s: np.ndarray):
    f, secs, spectrum = self._get_spectrum(s)
    dom_f = np.sum(f[..., np.newaxis] * spectrum, axis=0) / np.sum(
      spectrum, axis=0)
    return f, secs, spectrum, dom_f

  def _plot_spectrum(self, ax: plt.Axes):
    if self.get('stft'):
      f, t, spectrum = self._get_spectrum(self.explorer.selected_signal)

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

        # Plot sigma band
        ax.plot([0, 30], [11, 11], 'y:')
        ax.plot([0, 30], [16, 16], 'y:')

        ax2 = ax.twinx()
        ax2.set_yticks([(ymin + 4) / 2, 6, 10.5, (13 + ymax) / 2],
                       [r'$\delta$', r'$\theta$', r'$\alpha$', r'$\beta$'])
        ax2.set_ylim(ymin, ymax)
    else:
      from scipy.integrate import simps

      # Ref: https://raphaelvallat.com/bandpower.html
      s = self.explorer.selected_signal
      fs = self.explorer.selected_signal_group.digital_signals[0].sfreq

      if self.get('welch'):
        from scipy import signal

        # TODO: win_len should be determined
        # win_len is similar to bandwidth in multitaper, the win_len
        #  corresponding to the default bandwidth in multitaper is '8 * win_len'
        win_len = 8 * fs
        freqs, psd = signal.welch(s, fs, nperseg=win_len)
      else:
        from mne.time_frequency import psd_array_multitaper

        psd, freqs = psd_array_multitaper(
          s[np.newaxis, :], fs, adaptive=True, normalization='full', fmax=30,
          n_jobs=5, verbose=False)
        psd = psd.ravel()

      psd = psd * 1e12
      ax.plot(freqs, psd, color='k', lw=2)

      # Calculate total power
      freq_res = freqs[1] - freqs[0]
      total_power = simps(psd, dx=freq_res)

      for key, low, high, color in [
        ('Delta', 0.5, 4, '#4677b0'), ('Theta', 4, 8, '#599d3e'),
        ('Alpha', 8, 12, '#f1c440'), ('Beta', 12, 30, '#b05555')]:
        idx = np.logical_and(freqs >= low, freqs <= high)

        band_power = simps(psd[idx], dx=freq_res)
        label = f'{key} ({low}-{high} Hz), RP = {band_power / total_power:.3f}'

        ax.fill_between(freqs[idx], psd[idx], color=color, alpha=0.5,
                        label=label)
      ax.legend()

      ax.set_xlabel('Frequency [Hz]')
      ax.set_ylabel('PSD [$V^2$/Hz]')

      # ax.set_xlim([0, freqs.max()])
      # ax.set_xlim([0, 30])
      min_f, max_f = self.get('min_freq'), self.get('max_freq')
      ax.set_xlim([min_f, max_f])

      psd_max = psd[np.logical_and(freqs >= min_f, freqs <= max_f)].max()
      ymax = self.get('ymax', psd_max * 1.1)
      ax.set_ylim([0, ymax])

      # Show sigma band using rectangle
      ax.fill_between([11, 16], [0, 0], [ymax, ymax], color='y',
                      alpha=0.2, zorder=-999)

      ax.grid(True)

      return f'Welch' if self.get('welch') else f'Multitaper'


  def _low_freq_signal(self, s: np.ndarray):
    ks = int(self.get('dev_arg'))
    x = np.convolve(s, [1/ks] * ks, 'same')
    return x


  def pooling(self, s, size):
    # `size` should be an odd integer
    # assert size - size // 2 * 2 == 1
    p = (size - 1) // 2

    shifted_signals = [s]
    for i in range(1, p + 1):
      # i = 1, 2, ..., p
      s_l = np.concatenate([[s[0]] * i, s[:-i]])
      s_r = np.concatenate([s[i:], [s[-1]] * i])
      shifted_signals.extend([s_l, s_r])

    aligned_signals = np.stack(shifted_signals, axis=-1)
    upper = np.max(aligned_signals, axis=-1)
    lower = np.min(aligned_signals, axis=-1)

    return upper, lower


  def _get_summits(self, s: np.ndarray):
    """TODO"""
    ks = int(self.get('dev_arg'))
    x = np.convolve(s, [1/ks] * ks, 'same')

    d_x = x[1:] - x[:-1]
    sign_d_x = np.sign(d_x)
    d_sign = sign_d_x[1:] + sign_d_x[:-1]

    indices = np.argwhere(d_sign == 0)

    return indices


  def _butter_filt(self, s: np.ndarray):
    # Filter signal if required
    from scipy.signal import sosfilt
    from scipy.signal import butter

    filter_args: str = self.get('filter_arg').split(',')
    assert len(filter_args) == 2
    low, high = [float(s) for s in filter_args]

    fs = self.explorer.selected_signal_group.digital_signals[0].sfreq
    sos = butter(10, [low, high], 'bandpass', fs=fs, output='sos')
    s = sosfilt(sos, s)

    return s


  def _plot_signal(self, ax: plt.Axes):
    s: np.ndarray = self.explorer.selected_signal
    t = np.linspace(0, 30, num=len(s))

    if self.configs.get('layer', 1) == 2:
      x = self._low_freq_signal(s)
      ax.plot(t, s - x)
    else:
      if self.get('filter'): s = self._butter_filt(s)

      # Plot signal
      ax.plot(t, s)

      # Plot auxiliary lines if required
      if self.get('dev_mode'):
        x = self._low_freq_signal(s)
        ax.plot(t, x, 'r-')

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

    # (2) Plot frequency estimation
    if self.get('summit'):
      # (2.1)
      f, secs, spectrum, dom_f = self._calc_dominate_freq_curve_v1(
        self.explorer.selected_signal)

      ax2 = ax.twinx()
      ax2.plot(secs, dom_f, 'r:', linewidth=2)
      ax2.plot(secs, [4] * len(secs), '--', linewidth=2, color='orange')
      ax2.set_ylabel('Frequency Estimation')
      ax2.set_ylim([0, 12])

      # (2.2)
      upper, lower = self.pooling(
        self.explorer.selected_signal, int(self.get('dev_arg')))
      ax.plot(t, upper, ':', color='gray')
      ax.plot(t, lower, ':', color='green')

    # Return stats
    return ''

  # endregion: Plot Methods

  # region: Interfaces

  def set_developer_arg(self, v):
    """Set developer argument"""
    self.set('dev_arg', v)
  da = set_developer_arg

  def set_filter_arg(self, low='0.3', high='35'):
    self.set('filter_arg', f'{low},{high}')
  fa = set_filter_arg

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
    self.register_a_shortcut('s', lambda: self.flip('summit'),
                             'Toggle `show_summit`')

    self.register_a_shortcut('f', lambda: self.flip('filter'),
                             'Toggle `filter`')
    self.register_a_shortcut('t', lambda: self.flip('stft'),
                             'Toggle `stft`')
    self.register_a_shortcut('e', lambda: self.flip('welch'),
                             'Toggle `welch`')

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

    self.set('t1', t1, auto_refresh=False, verbose=False)
    self.set('t2', t2, auto_refresh=False, verbose=False)
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
  from roma import finder
  from roma import io

  # Set directories
  data_dir = r'../../data/'
  data_dir += 'sleepeasonx'

  prefix = ['', 'sleepedfx', 'ucddb', 'rrsh'][1]
  pattern = f'{prefix}*.sg'

  # Select .sg files
  sg_file_list = finder.walk(data_dir, pattern=pattern)[:20]

  signal_groups = []
  for path in sg_file_list:
    sg = io.load_file(path, verbose=True)
    signal_groups.append(sg)

  # Visualize signal groups
  EpochExplorer.explore(signal_groups, plot_wave=True, add_layer_2=True)


