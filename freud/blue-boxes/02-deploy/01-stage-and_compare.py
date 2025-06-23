from freud.deploy.inference import stage_alpha,compare
from freud.talos_utils.sleep_sets.sleepedfx import SleepEDFx, SleepSet




data_dir = r'../../../data/sleepedfx'
preprocess = 'trim;iqr'
t_path = r'E:\eason\project\xai-sleep\08-FNN\01_cnn_v1\checkpoints\0607_cnn_v1(16-s16-32-s32-64-5)\0607_cnn_v1(16-s16-32-s32-64-5).py'
signal_groups = SleepEDFx.load_as_signal_groups(data_dir, preprocess=preprocess)
anno = stage_alpha(signal_groups[2], t_path)
compare(signal_groups[2],anno)