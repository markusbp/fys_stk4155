import argparse

def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default = 1e-5, help = 'Learning rate', type = float)
    parser.add_argument('--train_steps', default = 10, help = 'Number of training steps',type = int)
    parser.add_argument('--batch_size', default = 50, help = 'Minibatch size',type = int)
    parser.add_argument('--timesteps', default = 100, help = 'Number of timesteps', type = int)
    parser.add_argument('--nodes', default = 100, help = 'Number of recurrent nodes',type = int)
    parser.add_argument('--out_nodes', default = 2, help = 'Number of output neurons', type = int)
    parser.add_argument('--activation', default = 'relu', help = 'Recurrent activation function')
    parser.add_argument('--dropout_rate', default = 0, help = 'Dropout rate', type = float)
    parser.add_argument('--beta', default = 1, help = 'Softmax beta factor', type = float)
    parser.add_argument('--l2', default = 0, help = 'L2 weight regularization factor', type = float)
    parser.add_argument('--load', default = False, help = 'Whether to only load model, not train', type = bool)
    return parser.parse_args()

if __name__ == '__main__':
    options = get_params()
