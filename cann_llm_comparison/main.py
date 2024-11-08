import os
import scientific_generative_agent


def main():
    config = _set_config()
    agent  = scientific_generative_agent.ScientificGenerativeAgent(config)
    agent.set_up()
    agent.run()
        

def _set_config():
    return {
        "input_dir":  os.path.join("..", "input"),
        "output_dir": os.path.join("..", "output"),
        "problem":    "brain_shear",
    }
    # Available problem types:
    # - "treloar_uniaxial_tension"
    # - "treloar_biaxial_tension"
    # - "treloar_shear"
    # - "synthetic_a_uniaxial_tension"
    # - "synthetic_a_biaxial_tension"
    # - "synthetic_a_shear"
    # - "synthetic_b_uniaxial_tension"
    # - "synthetic_b_biaxial_tension"
    # - "synthetic_b_shear"
    # - "brain_tension"
    # - "brain_compression"
    # - "brain_shear"


if __name__ == "__main__":
    main()
