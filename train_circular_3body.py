import os
import json
import ghnn

nn_path = os.path.join('..', 'NeuralNets')
if not os.path.exists(nn_path):
    os.mkdir(nn_path)

os.mkdir(os.path.join('..', 'NeuralNets', 'circular_3body'))
nn_types = ['MLP', 'SympNet', 'HenonNet', 'double_HenonNet', 'GHNN']

for nn_type in nn_types:
    nn_path = os.path.join('..', 'NeuralNets', 'circular_3body', nn_type)
    os.mkdir(nn_path)

    if nn_type == 'SympNet':
        with open(os.path.join('ghnn', 'training', 'default_G_SympNet.json')) as file_:
            settings = json.load(file_)
    else:
        with open(os.path.join('ghnn', 'training', f'default_{nn_type}.json')) as file_:
            settings = json.load(file_)

    data_path = os.path.split(os.getcwd())[0]
    data_path = os.path.join(data_path, 'Data', 'circular_3body', 'h_05_m5_training.h5.1')
    settings['data_path'] = data_path
    settings['batch_size'] = 1000

    for i in range(1, 11):
        settings['seed'] = i
        os.mkdir(os.path.join(nn_path, f'nn_{i}'))
        with open(os.path.join(nn_path, f'nn_{i}', 'settings.json'), 'w') as file_:
            json.dump(settings, file_, indent=4, separators=(',', ': '))

        wd = os.getcwd()
        os.chdir(os.path.join(nn_path, f'nn_{i}'))
        ghnn.training.train_from_folder()
        os.chdir(wd)
