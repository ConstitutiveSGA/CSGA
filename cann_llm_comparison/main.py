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
        "problem":    "synthetic_a",
    }
    # Available problem types:
    # - "synthetic_a"
    # - "synthetic_b"
    # - "brain"


if __name__ == "__main__":
    main()
