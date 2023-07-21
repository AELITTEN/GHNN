import os
import math
import ghnn

def generate_nn_paths_list(names):
    nn_paths_list = []
    for name in names:
        nn_paths = []
        for i in range(1,11):
            nn_paths.append(os.path.join('..', 'NeuralNets', 'm025T_pendulum', name, f'nn_{i}'))
        nn_paths_list.append(nn_paths)
    return nn_paths_list

if __name__ == '__main__':
    store_name = 'all_runs.h5.1'
    num_runs = 500
    data_path = os.path.join('..', 'Data', 'm025T_pendulum')

    all_names = ['MLP', 'SympNet', 'HenonNet', 'double_HenonNet', 'GHNN']
    best_names = ['MLP', 'SympNet', 'HenonNet', 'GHNN']

    figures_path = os.path.join('..', 'Figures')
    if not os.path.exists(figures_path):
        os.mkdir(figures_path)

    nn_paths_list = generate_nn_paths_list(best_names)
    ghnn.plotting.plot_data_mae_moments(data_path, store_name, nn_paths_list, 'mean_var', test='h_01_m025T_training.h5.1', period_q=math.pi, t_in_T=True, save_name=os.path.join(figures_path, 'm025T_pendulum_mae.png'))

    kwargs = {'energy': True, 'mse': True, 'phase_space': True, 'period_q': math.pi}
    nn_paths = [nn_path[0] for nn_path in nn_paths_list]
    ghnn.plotting.predict_pendulum(data_path, store_name, 6, nn_paths, save_name=os.path.join(figures_path, 'm025T_pendulum_plots.png'), **kwargs)


    """Other possible things to do:
        >>> ghnn.plotting.plot_loss_moments(nn_paths_list, 'mean_var')
        >>> ghnn.plotting.plot_data_mae_moments(data_path, store_name, nn_paths_list, 'mean_var', period_q=math.pi, t_in_T=True)

        >>> ghnn.plotting.plot_loss(nn_paths)
        >>> ghnn.plotting.plot_data_mae(data_path, store_name, nn_paths, max_time=max_time, period_q=math.pi, t_in_T=True)

        >>> ghnn.plotting.predict_pendulum_rand(data_path, store_name, num_runs, nn_paths[0], **kwargs)
        >>> ghnn.plotting.predict_pendulum_rand(data_path, store_name, num_runs, nn_paths, **kwargs)
    """
