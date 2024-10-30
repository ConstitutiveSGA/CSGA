import os
import re
import torch
import scientific_generative_agent


def main():
    config = _set_config()
    agent  = scientific_generative_agent.ScientificGenerativeAgent(config)
    agent.set_up()
    agent.run()
    
    # todo: plot predictions
    

def _set_config():
    return {
        "input_dir":  os.path.join("..", "input"),
        "data_file":  "treloar_uniaxial_tension.csv",
        "output_dir": os.path.join("..", "output"),
    }


if __name__ == "__main__":
    main()
