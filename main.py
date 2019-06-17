import torch
import torch.optim as optim

import policy
import deepnetwork
from model import Actor, Critic
import distribution
import customenv.multipendulum as multipendulum
import customenv.customgym as customgym
import statemodifier
import os, argparse
from time import sleep

def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--train_step', type=int, default=15000)
    parser.add_argument('--name', type=str, default='zigui')
    parser.add_argument('--env', type=str, default=None)
    parser.add_argument('--render', default=False, action="store_true")

    return parser.parse_args()

def get_env(name):
    if name is None: return None
    if name == "Hopper-v2": return customgym.PythonGym('Hopper-v2', 1000)
    if name == "multipendulum": return multipendulum.MultiPendulum()
    assert(False)

def main():
    args = argument_parse()
    env = get_env(args.env)

    if env is None: env = multipendulum.MultiPendulum()

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
        
    agent = policy.PPOAgent(env, actor, critic, {'gamma' : 0.99, 'lamda' : 0.99, 'steps' : 4096}, args.render)

    model_path = os.path.join(os.getcwd(),'save_model')

    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    if args.load_model is not None:
        ckpt = torch.load(os.path.join(os.getcwd(), args.load_model))
        agent.set_ckpt(ckpt)

    print("Start Training: PPO")
    for i in range(args.train_step):
        agent.train(train_step = 1)
        filename = 'ckpt_{}_{}_{}'.format(env.name, args.name, i)
        torch.save(agent.get_ckpt(), os.path.join(model_path, filename))
    
    print("End Training: PPO")

    state = env.reset()
    while True:
        env.render()

        action = agent.next_action(state)
        state, _, done, _, _ = env.step(action)
        sleep(0.05)
        if done:
            print("END")
            state = env.reset()
        
if __name__ == "__main__":
    main()
