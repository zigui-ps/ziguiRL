import torch
import torch.optim as optim
import torch.nn.functional as F

import policy
import deepnetwork
from model import Actor, Critic
import distribution
import customenv.multipendulum as multipendulum
#import customenv.customgym as customgym
import mybiped.biped as biped
import statemodifier
import os, argparse
from time import sleep

def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('xml_file_name', nargs='?')
    parser.add_argument('motion_file_name', nargs='?')
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--train_step', type=int, default=15000)
    parser.add_argument('--name', type=str, default='zigui')
    parser.add_argument('--env', type=str, default=None)
    parser.add_argument('--render', default=False, action="store_true")

    return parser.parse_args()

def get_env(name):
    if name is None: return None
#    if name == "Hopper-v2": return customgym.PythonGym('Hopper-v2', 1000)
    if name == "multipendulum": return multipendulum.MultiPendulum()
    if name == "biped": return bipedenv.Biped()
    assert(False)

def main():
    args = argument_parse()
    env = get_env(args.env)

    if env is None: env = biped.Biped()

#    env.env.seed(500)
#    torch.manual_seed(500)
    
    o_size = env.observation_size()[0]
    a_size = env.action_size()[0]

    actor_network = deepnetwork.CNN([o_size, 1024, 512, a_size], "actor")
    actor_opt = optim.Adam(actor_network.parameters(), lr=0.0002) # actor lr_rate

    dist_network = deepnetwork.CNN([o_size, 1024, 512, a_size], "dist", zeroInit=True)
    dist_opt = optim.Adam(dist_network.parameters(), lr = 0.0003)
    dist = distribution.NetGaussianDistribution(dist_network, dist_opt)
    
#    dist = distribution.FixedGaussianDistribution()

    actor = Actor(actor_network, actor_opt, dist)
    
    critic_network = deepnetwork.CNN([o_size, 1024, 512, 1], "critic")
    critic_opt = optim.Adam(critic_network.parameters(), lr=0.0003) # critic lr_rate
    critic = Critic(critic_network, critic_opt)
        
    agent = policy.PPOAgent(env, actor, critic, {'gamma' : 0.99, 'lamda' : 0.95, 'steps' : 20000, 'batch_size' : 1024, 'modifier' : statemodifier.ClassicModifier()}, args.render)

    model_path = os.path.join(os.getcwd(),'save_model')

    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    if args.load_model is not None:
        ckpt = torch.load(os.path.join(os.getcwd(), args.load_model))
        agent.set_ckpt(ckpt)

    print("Start Training: PPO")
    for i in range(args.train_step):
        agent.train(train_step = 1)
        file_name = 'ckpt_{}_{}_{}'.format(env.name, args.name, i)
        path = os.path.join(model_path, file_name)
        torch.save(agent.get_ckpt(), path)
        print("saved at {}".format(path))
    
    print("End Training: PPO")

    state = env.reset()
    while True:
        env.render()

        action = agent.next_action(state)
        state, _, done, _, _ = env.step(action)
#        sleep(0.05)
        if done:
            print("END")
            state = env.reset()
        
if __name__ == "__main__":
    main()
