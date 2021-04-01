import numpy as np
import yaml
from train import Train
from utils import common

if __name__ == "__main__":
    yaml_file = open("data/conf.yaml")
    conf = yaml.load(yaml_file, Loader=yaml.FullLoader)
    pose_desc_dict = common.load_all_mean_params(conf)
    param_mean = common.get_param_mean(pose_desc_dict)
    if conf['mode'] == 'train':
        model = Train(conf, pose_desc_dict, param_mean)
        model.forward()
    else:
        pass




