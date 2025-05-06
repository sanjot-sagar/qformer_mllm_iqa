import scipy.io

# Specify the path to your dmos.mat file
mat_file = '/scratch/sanjotst/datasets/LIVE_IQA/dmos.mat'

# Load the MATLAB file
mat_data = scipy.io.loadmat(mat_file)

# Print the headers or keys in the MATLAB file
print("Headers in dmos.mat:")
print(list(mat_data.keys()))

# Specify the path to your dmos.mat file
mat_file = '/scratch/sanjotst/datasets/LIVE_IQA/dmos.mat'
# Load the MATLAB file
mat_data = scipy.io.loadmat(mat_file)
# Print some information from each variable
for var_name in mat_data:
    if not var_name.startswith('__'):  # Skip internal variables
        print(f"Variable: {var_name}")
        # Print shape of the variable
        print(f"Shape: {mat_data[var_name].shape}")
        # Print data type of the variable
        print(f"Data Type: {mat_data[var_name].dtype}")
        # Print first few elements if the variable is an array
        if mat_data[var_name].ndim == 2:  # Check if 2D array
            print(f"First few elements: {mat_data[var_name][:5]}")
        print("=" * 30)
