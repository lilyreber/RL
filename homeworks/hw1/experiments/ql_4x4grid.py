import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
os.environ["SUMO_HOME"] = "/usr/share/sumo"

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.agents import MyQLAgent
from sumo_rl.exploration import EpsilonGreedy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="ql", choices=["ql", "myql"])
    parser.add_argument("--algorithm", type=str, default="q", choices=["q", "sarsa"])
    args = parser.parse_args()



    alpha = 0.1
    gamma = 0.99
    decay = 1
    runs = 1 #30
    episodes = 4

    env = SumoEnvironment(
        net_file="sumo_rl/nets/4x4-Lucas/4x4.net.xml",
        route_file="sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml",
        use_gui=True,
        num_seconds=8000, #80000,
        min_green=5,
        delta_time=5,
    )

    for run in range(1, runs + 1):
        initial_states = env.reset()

        if args.agent == "ql":
            ql_agents = {
                ts: QLAgent(
                    starting_state=env.encode(initial_states[ts], ts),
                    state_space=env.observation_space,
                    action_space=env.action_space,
                    alpha=alpha,
                    gamma=gamma,
                    exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005, decay=decay),
                )
                for ts in env.ts_ids
            }
        elif args.agent == "myql":
            ql_agents = {
                ts: MyQLAgent(
                    starting_state=env.encode(initial_states[ts], ts),
                    state_space=env.observation_space,
                    action_space=env.action_space,
                    alpha=alpha,
                    gamma=gamma,
                    exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005, decay=decay),
                )
                for ts in env.ts_ids
            }

        for episode in range(1, episodes + 1):
            if episode != 1:
                initial_states = env.reset()
                for ts in initial_states.keys():
                    ql_agents[ts].state = env.encode(initial_states[ts], ts)

            infos = []
            done = {"__all__": False}
            while not done["__all__"]:
                actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

                s, r, done, info = env.step(action=actions)

                for agent_id in s.keys():
                    if args.agent == "ql":
                        ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])
                    elif args.agent == "myql":
                        ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id], algorithm=args.algorithm)
            if args.agent == "ql":
                env.save_csv(f"outputs/4x4/ql-4x4grid_run{run}", episode)
            else:
                env.save_csv(f"outputs/4x4/myql_{args.algorithm}-4x4grid_run{run}", episode)

    env.close()
