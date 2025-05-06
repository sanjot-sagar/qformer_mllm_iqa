import os
import yaml
# Step 1: Load the existing YAML file
existing_yml_path = '/home/sanjotst/llm_iqa/llm-iqa/code/baselines/utils/environment.yml'
new_yml_path = './llm6_clone.yml'
with open(existing_yml_path, 'r') as file:
    env_data = yaml.safe_load(file)
# Step 2: Modify the environment name and remove the prefix
# Replace with your desired new environment name
env_data['name'] = 'llm8'
if 'prefix' in env_data:
    del env_data['prefix']
# Save the modified YAML to a new file
with open(new_yml_path, 'w') as file:
    yaml.safe_dump(env_data, file)
# Step 3: Create the new environment using the modified YAML file
os.system(f'conda env create -f {new_yml_path}')
