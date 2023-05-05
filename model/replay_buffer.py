import numpy as np
from typing import Dict, List, Tuple, Deque
from collections import deque

class ReplayBuffer:
    """
    A simple numpy replay buffer.
    和d3qn中的ReplayBuffer 类似 多了一个acts_param_buf
    obs_dim: the dimension of the input
    size: the size of ReplayBuffer or memory
    batch_size
    """

    def __init__(self, obs_dim: int, param_dim: int, tl_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.tl_buf = np.zeros([size, tl_dim], dtype=np.float32)
        self.next_tl_buf = np.zeros([size, tl_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.acts_param_buf = np.zeros([size, param_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0 # self.ptr points the position to add a new line, self.size is the length of loaded lines

    def store(
        self,
        obs: np.ndarray,
        tl_code: np.ndarray,
        act: np.ndarray, 
        act_param: float,
        rew: float, 
        next_obs: np.ndarray, 
        next_tl_code: np.ndarray,
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.tl_buf[self.ptr] = tl_code
        self.next_tl_buf[self.ptr] = next_tl_code
        self.acts_buf[self.ptr] = act
        self.acts_param_buf[self.ptr] = act_param
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False) # get a batch from loaded lines
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    tl_code=self.tl_buf[idxs],
                    next_tl_code=self.next_tl_buf[idxs],
                    act=self.acts_buf[idxs],
                    act_param = self.acts_param_buf[idxs],
                    rew=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size
    

class NSTEPReplayBuffer:
    #Replay Buffer for n-step learning
    def __init__(self, 
        obs_dim: int, 
        param_dim: int, 
        tl_dim: int, 
        size: int, 
        batch_size: int, 
        n_step: int = 1, 
        gamma: float =0.99
    ):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.tl_buf = np.zeros([size, tl_dim], dtype=np.float32)
        self.next_tl_buf = np.zeros([size, tl_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.acts_param_buf = np.zeros([size, param_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0 # self.ptr points the position to add a new line, self.size is the length of loaded lines

        #for n-step learning
        self.n_step_buffer=deque(maxlen=n_step)
        self.n_step=n_step
        self.gamma=gamma

    def store(
        self,
        obs: np.ndarray,
        tl_code: np.ndarray,
        act: np.ndarray, 
        act_param: float,
        rew: float, 
        next_obs: np.ndarray, 
        next_tl_code: np.ndarray,
        done: bool,
    ):
        transition=(obs, tl_code, act, act_param, rew, next_obs, next_tl_code, done)
        self.n_step_buffer.append(transition)

        #single step transition is not ready
        if len(self.n_step_buffer)<self.n_step:
            return ()
            
        #make a n-step transition
        # make a n-step transition
        rew, next_obs, next_tl_code, done = self._get_n_step_info(
            self.n_step_buffer, self.gamma
        )
        obs, tl_code, act, act_param = self.n_step_buffer[0][:4]

        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.tl_buf[self.ptr] = tl_code
        self.next_tl_buf[self.ptr] = next_tl_code
        self.acts_buf[self.ptr] = act
        self.acts_param_buf[self.ptr] = act_param
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False) # get a batch from loaded lines
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    tl_code=self.tl_buf[idxs],
                    next_tl_code=self.next_tl_buf[idxs],
                    act=self.acts_buf[idxs],
                    act_param = self.acts_param_buf[idxs],
                    rew=self.rews_buf[idxs],
                    done=self.done_buf[idxs])
    
    def sample_batch_from_idxs(
        self, indices: np.ndarray
    ) -> Dict[str, np.ndarray]:
        # for N-step Learning
        return dict(
            obs=self.obs_buf[indices],
            next_obs=self.next_obs_buf[indices],
            acts=self.acts_buf[indices],
            rews=self.rews_buf[indices],
            done=self.done_buf[indices])

    def _get_n_step_info(
        self, n_step_buffer: Deque, gamma: float
    ) -> Tuple[np.int64, np.ndarray, bool]:
        """Return n step rew, next_obs, next_tl_code and done."""
        # info of the last transition
        rew, next_obs, next_tl_code, done = n_step_buffer[-1][-4:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, n_tl, d = transition[-4:]

            rew = r + gamma * rew * (1 - d)
            next_obs, next_tl_code, done = (n_o, n_tl, d) if d else (next_obs, next_tl_code, done)

        return rew, next_obs, next_tl_code, done

    def __len__(self) -> int:
        return self.size
    

class RecurrentReplayBuffer:
    # Replay Buffer for Burn-in R2D2 (RECURRENT EXPERIENCE REPLAY IN DISTRIBUTED REINFORCEMENT LEARNING)
    def __init__(self, 
        obs_dim: int, 
        param_dim: int, 
        tl_dim: int, 
        size: int, 
        batch_size: int, 
        n_step: int = 1, 
        burn_in_step: int = 20,
        gamma: float =0.99
    ):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.tl_buf = np.zeros([size, tl_dim], dtype=np.float32)
        self.next_tl_buf = np.zeros([size, tl_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.acts_param_buf = np.zeros([size, param_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0 # self.ptr points the position to add a new line, self.size is the length of loaded lines

        #previous steps info for burn-in RNN
        self.burn_in_buffer=deque(maxlen=burn_in_step)
        self.burn_in_step=burn_in_step
        self.prev_obs=np.zeros([size, burn_in_step-1, obs_dim], dtype=np.float32)
        self.prev_tl_code=np.zeros([size, burn_in_step-1, tl_dim], dtype=np.float32)
        self.prev_acts=np.zeros([size, burn_in_step-1, 1], dtype=np.float32)
        self.prev_acts_param=np.zeros([size, burn_in_step-1, param_dim], dtype=np.float32)

        #for n-step learning
        self.n_step_buffer=deque(maxlen=n_step)
        self.n_step=n_step

        self.gamma=gamma

    def store(
        self,
        obs: np.ndarray,
        tl_code: np.ndarray,
        act: np.ndarray, 
        act_param: float,
        rew: float, 
        next_obs: np.ndarray, 
        next_tl_code: np.ndarray,
        done: bool,
    ):
        transition=(obs, tl_code, act, act_param, rew, next_obs, next_tl_code, done)
        self.n_step_buffer.append(transition)
        self.burn_in_buffer.append(transition)

        #single step transition is not ready
        if len(self.n_step_buffer)<self.n_step or len(self.burn_in_buffer)<self.burn_in_step:
            return ()

        for i, trans in enumerate(self.burn_in_buffer):
            if i == self.burn_in_step - 1:
                break
            self.prev_obs[self.ptr, i]=trans[0]
            self.prev_tl_code[self.ptr, i]=trans[1]
            self.prev_acts[self.ptr, i]=trans[2]
            self.prev_acts_param[self.ptr, i]=trans[3]
            
        #make a n-step transition
        #make a n-step transition
        rew, next_obs, next_tl_code, done = self._get_n_step_info(
            self.n_step_buffer, self.gamma
        )
        obs, tl_code, act, act_param = self.n_step_buffer[0][:4]

        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.tl_buf[self.ptr] = tl_code
        self.next_tl_buf[self.ptr] = next_tl_code
        self.acts_buf[self.ptr] = act
        self.acts_param_buf[self.ptr] = act_param
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False) # get a batch from loaded lines
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    tl_code=self.tl_buf[idxs],
                    next_tl_code=self.next_tl_buf[idxs],
                    act=self.acts_buf[idxs],
                    act_param = self.acts_param_buf[idxs],
                    rew=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    prev_obs=self.prev_obs[idxs],
                    prev_tl_code=self.prev_tl_code[idxs],
                    prev_acts=self.prev_acts[idxs],
                    prev_acts_param=self.prev_acts_param[idxs])

    def _get_n_step_info(
        self, n_step_buffer: Deque, gamma: float
    ) -> Tuple[np.int64, np.ndarray, bool]:
        """Return n step rew, next_obs, next_tl_code and done."""
        # info of the last transition
        rew, next_obs, next_tl_code, done = n_step_buffer[-1][-4:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, n_tl, d = transition[-4:]

            rew = r + gamma * rew * (1 - d)
            next_obs, next_tl_code, done = (n_o, n_tl, d) if d else (next_obs, next_tl_code, done)

        return rew, next_obs, next_tl_code, done

    def __len__(self) -> int:
        return self.size