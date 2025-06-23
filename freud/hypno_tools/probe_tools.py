from collections import OrderedDict
from hypnomics.hypnoprints.probes import ProbeLibrary



def get_extractor_dict(keys, **kwargs):
  od = OrderedDict()

  for key in keys:
    if '-' in key: name, arg = key.split('-')
    else: name, arg = key, None

    if name == 'AMP':
      _fs = kwargs.get('fs')
      _ws = float(arg)
      od[key] = lambda s, fs=_fs, ws=_ws: ProbeLibrary.amplitude(
        s, fs=fs, window_size=ws)
    elif name == 'FREQ':
      _fs = kwargs.get('fs')
      _fmax = float(arg)
      od[key] = lambda s, fs=_fs, fmax=_fmax: ProbeLibrary.frequency_stft(
        s, fs=fs, fmax=fmax)
    elif name == 'GFREQ':
      _fs = kwargs.get('fs')
      _fmax = float(arg)
      od[key] = lambda s, fs=_fs, fmax=_fmax: ProbeLibrary.frequency_st(
        s, fs=fs, fmax=fmax)
    elif name in ('P', 'RP'):
      _fs = kwargs.get('fs')
      od[key] = lambda s, fs=_fs, band=arg: ProbeLibrary.total_power(
        s, fs=fs, band=band)
    elif name == 'MAG':
      od[key] = lambda s: ProbeLibrary.mean_absolute_gradient(s)
    elif name == 'KURT':
      od[key] = lambda s: ProbeLibrary.kurtosis(s)
    elif name == 'ENTROPY':
      od[key] = lambda s: ProbeLibrary.sample_entropy(s)
    elif name == 'RPS':
      _fs = kwargs.get('fs')
      _b1, _b2, _st = arg.split('_')
      _st = {'95': '95th percentile', 'MIN': 'min',
            'AVG': 'mean', 'STD': 'std'}[_st]
      od[key] = lambda s, fs=_fs, b1=_b1, b2=_b2, st=_st: ProbeLibrary.relative_power_stats(
        s, fs=fs, band1=b1, band2=b2, stat_key=st)
    elif name == 'BKURT':
      _fs = kwargs.get('fs')
      od[key] = lambda s, fs=_fs, band=arg: ProbeLibrary.band_kurtosis(
        s, fs=fs, band=band)
    elif name == 'power_group':
      _fs = kwargs.get('fs')
      od[key] = ProbeLibrary.class_power_group(_fs)
    else: raise KeyError(f'Unknown key: {key}')

  return od



def get_probe_keys(PROBE_CONFIG, expand_group=False):
  """Set expand_group to False if the results are used to get_extractors"""
  PROBE_KEYS = []

  # (1.4.1) Part A
  if 'A' in PROBE_CONFIG: PROBE_KEYS.extend(['FREQ-20', 'AMP-1'])

  # (1.4.2) Part B
  if 'B' in PROBE_CONFIG: PROBE_KEYS.extend(
    ['P-TOTAL', 'RP-DELTA', 'RP-THETA', 'RP-ALPHA', 'RP-BETA'])

  """ 
  ['POWER-30', 'PR-DELTA_TOTAL', 'PR-THETA_TOTAL', 'PR-ALPHA_TOTAL', 
   'PR-BETA_TOTAL', 'PR-SIGMA_TOTAL', 'PR-DELTA_THETA', 
   'PR-DELTA_ALPHA', 'PR-THETA_ALPHA']
  """
  if 'b' in PROBE_CONFIG:
    if expand_group:
      from hypnomics.hypnoprints.probes.wavestats.power_probes import PowerProbes
      PROBE_KEYS.extend(PowerProbes.probe_keys)
    else: PROBE_KEYS.append('power_group')

    PROBE_KEYS.append('KURT')


  # (1.4.3) Part C (sun2017)
  if 'C' in PROBE_CONFIG:
    # (1.4.3.1)
    PROBE_KEYS.extend(['MAG', 'KURT', 'ENTROPY'])

    # (1.4.3.2)
    for b1, b2 in [('DELTA', 'TOTAL'), ('THETA', 'TOTAL'), ('ALPHA', 'TOTAL'),
                   ('DELTA', 'THETA'), ('DELTA', 'ALPHA'), ('THETA', 'ALPHA')]:
      for stat_key in ['95', 'MIN', 'AVG', 'STD']:
        PROBE_KEYS.append(f'RPS-{b1}_{b2}_{stat_key}')

    # (1.4.3.3)
    for b in ['DELTA', 'THETA', 'ALPHA', 'SIGMA']:
      PROBE_KEYS.append(f'BKURT-{b}')

  if 'c' in PROBE_CONFIG:
    PROBE_KEYS.extend(['MAG', 'KURT', 'ENTROPY'])

  # (1.4.4) Part D (a useful part of sun2017)
  if 'D' in PROBE_CONFIG:
    assert 'C' not in PROBE_CONFIG
    PROBE_KEYS.extend(['KURT'])

    for b1, b2 in [('DELTA', 'THETA'), ('DELTA', 'ALPHA'), ('THETA', 'ALPHA')]:
      PROBE_KEYS.append(f'RPS-{b1}_{b2}_AVG')

  # (1.4.5)
  if 'E' in PROBE_CONFIG: PROBE_KEYS.extend(
      [f'PR-{band}_TOTAL' for band in (
        'DELTA', 'THETA', 'ALPHA', 'BETA', 'SIGMA')])

  # (1.4.4.6)
  if 'd' in PROBE_CONFIG:
    PROBE_KEYS.extend(['POWER-30', 'PR-DELTA_TOTAL', 'PR-THETA_TOTAL',
                       'PR-ALPHA_TOTAL', 'PR-BETA_TOTAL'])

    PROBE_KEYS.append('KURT')

    for b1, b2 in [('DELTA', 'THETA'), ('DELTA', 'ALPHA'), ('THETA', 'ALPHA')]:
      PROBE_KEYS.append(f'PR-{b1}_{b2}')

  return PROBE_KEYS


def get_probe_suffix(probe_config):
  probe_keys = get_probe_keys(probe_config, expand_group=True)
  return f'{probe_config}{len(probe_keys)}'
