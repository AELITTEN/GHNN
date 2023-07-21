"""Just something to start the training."""
import os
from ghnn.nets.helpers import net_from_dir

__all__ = ['train_from_folder']

def train_from_folder():
    """Starts the training from the current directory. Expects a 'settings.json' file."""
    my_net = net_from_dir('settings.json')

    if os.path.exists('checkpoints'):
        epoch = max([int(file[5:-5]) for file in os.listdir('checkpoints')])
        my_net.load_from_checkpoint(os.path.join('checkpoints', f'model{epoch}.json'))
        my_net.settings['initial_epoch'] = epoch

    my_net.train()

if __name__ == '__main__':
    train_from_folder()
