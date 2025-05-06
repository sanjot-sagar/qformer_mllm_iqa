import scipy.io

# Specify the path to your dmos.mat file and refnames_all.mat file
dmos_mat_file = '/scratch/sanjotst/datasets/LIVE_IQA/dmos.mat'
refnames_mat_file = '/scratch/sanjotst/datasets/LIVE_IQA/refnames_all.mat'

# Load the MATLAB files
dmos_data = scipy.io.loadmat(dmos_mat_file)
refnames_data = scipy.io.loadmat(refnames_mat_file)

# Extract dmos values and corresponding filenames
dmos_values = dmos_data['dmos'].flatten()  # Flatten to 1D array
orgs = dmos_data['orgs'].flatten()  # Flatten to 1D array
refnames_all = refnames_data['refnames_all'].flatten()  # Flatten to 1D array

# Iterate through and print DMOS values for distorted images
print("DMOS values for distorted images:")
for i in range(len(orgs)):
    if orgs[i] == 0:
        filename = refnames_all[i][0]  # Extract filename
        dmos_value = dmos_values[i]
        print(f"Filename: {filename}, DMOS: {dmos_value}")
