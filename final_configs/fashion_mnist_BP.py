config = {
'lr': 0.000287604813302375,
'beta1': 0.9,
'beta2': 0.99,
'epsilon': 4.7253263377092954e-08,
'out_dir': 'logs/fashion_mnist/BP',
'network_type': 'BP',
'initialization': 'xavier_normal',
'target_stepsize': 1.0,
'dataset': 'fashion_mnist',
'optimizer': 'Adam',
'optimizer_fb': 'Adam',
'momentum': 0.0,
'parallel': True,
'normalize_lr': True,
'batch_size': 128,
'forward_wd': 0.0,
'epochs_fb': 0,
'not_randomized': True,
'not_randomized_fb': True,
'extra_fb_minibatches': 0,
'epochs': 100,
'num_hidden': 5,
'size_hidden': 256,
'size_input': 784,
'size_output': 10,
'hidden_activation': 'tanh',
'output_activation': 'softmax',
'no_bias': False,
'no_cuda': False,
'random_seed': 42,
'cuda_deterministic': False,
'freeze_BPlayers': False,
'multiple_hpsearch': False,
'save_logs': False,
'save_BP_angle': False,
'save_GN_angle': False,
'save_GN_activations_angle': False,
'save_BP_activations_angle': False,
'gn_damping': 0.0,
'hpsearch': False,
'log_interval': 5,
}