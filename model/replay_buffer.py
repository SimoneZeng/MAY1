import numpy as np
from typing import Dict, List, Tuple

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
    
