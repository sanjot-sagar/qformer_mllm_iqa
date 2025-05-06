import torch


def print_state_dict_keys(model_path):
    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # Get the state dictionary from the checkpoint
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Print all keys in the state dictionary
    print("Keys in the state dictionary:")
    for key in state_dict.keys():
        print(key)

    # Check if 'regressor' is in the state dictionary
    if 'regressor' in state_dict:
        regressor_state = state_dict['regressor']
        if 'weight' in regressor_state:
            print("Regressor weights shape:", regressor_state['weight'].shape)
        if 'bias' in regressor_state:
            print("Regressor bias shape:", regressor_state['bias'].shape)


if __name__ == "__main__":
    # Path to the model checkpoint file (.tar file)
    model_path = "/scratch/sanjotst/mse_runs_replication/Run0012/Train/iter_1890.tar"

    # Call the function to print keys in the state dictionary
    print_state_dict_keys(model_path)
