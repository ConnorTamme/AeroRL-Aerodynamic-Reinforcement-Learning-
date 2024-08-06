from ddqn_agent import DDQN_Agent

if __name__ == "__main__":
    ddqn_agent = DDQN_Agent(70000, 0.8, 0.0005, "Adam", useDepth=True)
    ddqn_agent.train()
