import torch
import torch.optim as optim

import policy
import deepnetwork
from model import Actor, Critic
import distribution
import customenv.multipendulum as multipendulum
import statemodifier

def main():
#    env = customgym.PythonGym('Hopper-v2', 1000)
    env = multipendulum.MultiPendulum()

#    env.env.seed(500)
#    torch.manual_seed(500)
    
    o_size = env.observation_size()[0]
    a_size = env.action_size()[0]

    actor_network = deepnetwork.CNN([o_size, 64, 64, a_size], "actor")
    actor_opt = optim.Adam(actor_network.parameters(), lr=0.003) # actor lr_rate

    actor = Actor(actor_network, actor_opt, distribution.GaussianDistribution)
    
    critic_network = deepnetwork.CNN([o_size, 64, 64, 1], "critic")
    critic_opt = optim.Adam(critic_network.parameters(), lr=0.003, # critic lr_rate
                              weight_decay=0.001) # critie lr_rate2
    critic = Critic(critic_network, critic_opt)
        
    agent = policy.PPOAgent(env, actor, critic, {'gamma' : 0.998, 'lamda' : 0.996, 'steps' : 4096, 'modifier' : statemodifier.DefaultModifier()})

    print("Start Training: PPO")
    agent.train(train_step = 15000)
    print("End Training: PPO")

    state = env.reset()
    while True:
        env.render()

        action = agent.next_action(state)
        state, _, done, _ = env.step(action)
        if done:
            print("END")
            state = env.reset()
        
if __name__ == "__main__":
    main()
