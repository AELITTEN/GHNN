import os
import ghnn

def generate_nn_paths_list(names):
    nn_paths_list = []
    for name in names:
        nn_paths = []
        for i in range(1,11):
            nn_paths.append(os.path.join('..', 'NeuralNets', 'circular_3body', name, f'nn_{i}'))
        nn_paths_list.append(nn_paths)
    return nn_paths_list

if __name__ == '__main__':
    store_name = 'circ_3body_all_runs.h5.1'
    num_runs = 5000
    max_time = 7
    data_path = os.path.join('..', 'Data')

    all_names = ['MLP', 'SympNet', 'HenonNet', 'double_HenonNet', 'GHNN']
    best_names = ['MLP', 'SympNet', 'double_HenonNet', 'GHNN']

    figures_path = os.path.join('..', 'Figures')
    if not os.path.exists(figures_path):
        os.mkdir(figures_path)

    nn_paths_list = generate_nn_paths_list(all_names)
    ghnn.plotting.plot_data_mae_moments(data_path, store_name, nn_paths_list, 'mean_var', test='circ_3body_training.h5.1', max_time=max_time, save_name=os.path.join(figures_path, 'circ_3body_mae.png'))

    kwargs = {'energy': True, 'momentum': False, 'mse': True, 'max_time': max_time}
    nn_paths_list = generate_nn_paths_list(best_names)
    nn_paths = [nn_path[0] for nn_path in nn_paths_list]
    ghnn.plotting.predict_run(data_path, store_name, 1, nn_paths, save_name=os.path.join(figures_path, 'circ_3body_plots.png'), **kwargs)


    """Other possible things to do:
        >>> ghnn.plotting.plot_loss_moments(nn_paths_list, 'mean_var')
        >>> ghnn.plotting.plot_data_mae_moments(data_path, store_name, nn_paths_list, 'mean_var', max_time=max_time)

        >>> ghnn.plotting.plot_loss(nn_paths)
        >>> ghnn.plotting.plot_data_mae(data_path, store_name, nn_paths, max_time=max_time)

        >>> ghnn.plotting.predict_run_rand(data_path, store_name, num_runs, nn_paths[0], **kwargs)
        >>> ghnn.plotting.predict_run_rand(data_path, store_name, num_runs, nn_paths, **kwargs)
    """
