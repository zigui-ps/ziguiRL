from statemodifier import ClassicModifier
from replaybuffer import ReplayBuffer
import torch

class AgentInterface():
    def __init__(self, env, actor, critic, args):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.args = args
        self.state_normalizer = ClassicModifier() if 'modifier' not in args else args['modifier']
        self._episodes = 0

        assert('steps' in args)
        self.steps = args['steps']
        

    def get_replay_buffer(self):
        total_score, steps, n = 0, 0, 0
        replay_buffer = ReplayBuffer()
        while steps < self.steps:
            self._episodes += 1
            n += 1
            state = self.state_normalizer.apply(self.env.reset())
            score = 0
            while True: # timelimits
                #if n == 1: 
                self.env.render()
                action = self.actor.get_action(state)
                next_state, reward, done, tl, _ = self.env.step(action)
                next_state = self.state_normalizer.apply(next_state)
                
                if tl:
                    reward += self.critic.get_values(next_state).detach()[0]
                
                score += reward
                replay_buffer.append(state, action, reward, done)
                    
                state = next_state
                total_score, steps = total_score + reward, steps + 1
                if done: break

            replay_buffer.path_finish()
        
        print("episodes: {}, score: {}".format(self._episodes, total_score / n))
        return replay_buffer

    def train(self):
        assert(False)

    def next_action(self, state):
        state = self.state_normalizer.modify(state)
        return self.actor.action(state)

class PPOAgent(AgentInterface):
    def __init__(self, env, actor, critic, args):
        super(PPOAgent, self).__init__(env, actor, critic, args)

        assert('gamma' and 'lamda' in args)
        self.gamma = args['gamma']
        self.lamda = args['lamda']

    def train(self, train_step):
        for _ in range(train_step): # train step
            self.actor.mode_eval()
            self.critic.mode_eval()

            replay_buffer = self.get_replay_buffer()
            states, actions = replay_buffer.get_tensor()

            old_policy = self.actor.log_policy(states, actions).detach()
            old_values = self.critic.get_values(states).detach()

            returns, advants = replay_buffer.get_gae(old_values, self.gamma, self.lamda) # gamma
            
            criterion = torch.nn.MSELoss()
            n = len(states)
            batch_size = 64
            
            self.actor.mode_train()
            self.critic.mode_train()
    
            for epoch in range(10):
                arr = torch.randperm(n)
                for i in range(n // batch_size):
                    batch_index = arr[batch_size * i : batch_size * (i+1)]
                    states_samples = states[batch_index]
                    returns_samples = returns[batch_index]
                    advants_samples = advants[batch_index]
                    actions_samples = actions[batch_index]
                    oldvalues_samples = old_values[batch_index]
                    oldpolicy_samples = old_policy[batch_index]

                    # surrogate function
                    new_policy = self.actor.log_policy(states_samples, actions_samples)
                    ratio = torch.exp(new_policy - oldpolicy_samples)
                    loss = ratio * advants_samples

                    # clip
                    values = self.critic.get_values(states_samples)
                    clipped_values = oldvalues_samples + \
                            torch.clamp(values - oldvalues_samples, -0.2, 0.2) # clip param
                    critic_loss1 = criterion(clipped_values, returns_samples)
                    critic_loss2 = criterion(values, returns_samples)
                    critic_loss = torch.max(critic_loss1, critic_loss2).mean()

                    clipped_ratio = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) # clip param
                    clipped_loss = clipped_ratio * advants_samples
                    actor_loss = -torch.min(loss, clipped_loss).mean()

                    # merge loss function
                    loss = actor_loss + 0.5 * critic_loss

                    self.actor.apply_loss(loss, retain_graph=True)
                    self.critic.apply_loss(loss, retain_graph=False)

class VanilaAgent(AgentInterface):
    def __init__(self, env, actor, critic, args):
        super(VanilaAgent, self).__init__(env, actor, critic, args)
        assert('gamma' and 'lamda' and 'batch_size' in args)
        self.gamma = args['gamma']

    def train(self):
        for _ in range(15000): # train step
            self.actor.mode_eval()
            self.critic.mode_eval()

            replay_buffer = self.get_replay_buffer()
            states, actions = replay_buffer.get_tensor()

            returns = replay_buffer.get_returns(self.gamma) # gamma
        
            self.actor.mode_train()
            self.critic.mode_train()
            
            log_policy = self.actor.log_policy(states, actions)
            returns = returns.unsqueeze(1)
            loss = -(returns * log_policy).mean()
            
            self.actor.apply_loss(loss)

