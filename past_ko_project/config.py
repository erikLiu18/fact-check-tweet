def interpret_config(config):
    if config["data_param"]["dataset"] == "mnist":
        config["data_param"]["num_classes"] = 10
        config["data_param"]["inplanes"] = 1
        config["data_param"]["size"] = 28
    elif config["data_param"]["dataset"] == "cifar10":
        config["data_param"]["num_classes"] = 10
        config["data_param"]["inplanes"] = 3
        config["data_param"]["size"] = 32
    return config
