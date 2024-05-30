import os
import argparse
import numpy as np
from src.config import ConfigTGCN
from src.dataprocess import generate_dataset_ms


# Set related parameters
parser = argparse.ArgumentParser()
parser.add_argument('--result_path', help="path for preprocess outputs", type=str, default='./preprocess_Result/')
args = parser.parse_args()


if __name__ == "__main__":

    # Config initialization
    config = ConfigTGCN()
    config.batch_size = 1
    # Load evaluation dataset
    dataset = generate_dataset_ms(config, training=False)
    # Sub-directory of resulting inputs
    inputs_path = os.path.join(args.result_path, "inputs")
    os.mkdir(inputs_path)
    # The dataset is an instance of Dataset object
    iterator = dataset.create_dict_iterator(output_numpy=True)
    # Generate binary dataset files
    targets_list = []
    for i, data in enumerate(iterator):
        file_name = "T-GCN_data_bs" + str(config.batch_size) + "_" + str(i) + ".bin"
        file_path = inputs_path + "/" + file_name
        data['inputs'].tofile(file_path)
        targets_list.append(data['targets'])
    np.save(args.result_path + "targets_ids.npy", targets_list)
    print("="*20, "Generate binary dataset files finished!", "="*20)
