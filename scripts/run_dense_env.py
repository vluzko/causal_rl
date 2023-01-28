from causal_rl.environments import Torus


def main():
    env = Torus(5)
    data = env.generate_data(100)


if __name__ == "__main__":
    main()
