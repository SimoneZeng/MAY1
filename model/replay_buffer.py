import operator, random
import numpy as np
from typing import Dict, List, Tuple, Deque, Callable
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

        return self.n_step_buffer[0]

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
    

class SegmentTree:
    """ Create SegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    Attributes:
        capacity (int)
        tree (list)
        operation (function)

    """

    def __init__(self, capacity: int, operation: Callable, init_value: float):
        """Initialization.

        Args:
            capacity (int)
            operation (function)
            init_value (float)

        """
        assert (
            capacity > 0 and capacity & (capacity - 1) == 0
        ), "capacity must be positive and a power of 2."
        self.capacity = capacity
        self.tree = [init_value for _ in range(2 * capacity)]
        self.operation = operation

    def _operate_helper(
        self, start: int, end: int, node: int, node_start: int, node_end: int
    ) -> float:
        """Returns result of operation in segment."""
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._operate_helper(start, mid, 2 * node, node_start, mid),
                    self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )

    def operate(self, start: int = 0, end: int = 0) -> float:
        """Returns result of applying `self.operation`."""
        if end <= 0:
            end += self.capacity
        end -= 1

        return self._operate_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float):
        """Set value in tree."""
        idx += self.capacity
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        """Get real value in leaf node of tree."""
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity + idx]


class SumSegmentTree(SegmentTree):
    """ Create SumSegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    """

    def __init__(self, capacity: int):
        """Initialization.

        Args:
            capacity (int)

        """
        super(SumSegmentTree, self).__init__(
            capacity=capacity, operation=operator.add, init_value=0.0
        )

    def sum(self, start: int = 0, end: int = 0) -> float:
        """Returns arr[start] + ... + arr[end]."""
        return super(SumSegmentTree, self).operate(start, end)

    def retrieve(self, upperbound: float) -> int:
        """Find the highest index `i` about upper bound in the tree"""
        # TODO: Check assert case and fix bug
        assert 0 <= upperbound <= self.sum() + 1e-5, "upperbound: {}".format(upperbound)

        idx = 1

        while idx < self.capacity:  # while non-leaf
            left = 2 * idx
            right = left + 1
            if self.tree[left] > upperbound:
                idx = 2 * idx
            else:
                upperbound -= self.tree[left]
                idx = right
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    """ Create SegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    """

    def __init__(self, capacity: int):
        """Initialization.

        Args:
            capacity (int)

        """
        super(MinSegmentTree, self).__init__(
            capacity=capacity, operation=min, init_value=float("inf")
        )

    def min(self, start: int = 0, end: int = 0) -> float:
        """Returns min(arr[start], ...,  arr[end])."""
        return super(MinSegmentTree, self).operate(start, end)
    

class PrioritizedReplayBuffer(NSTEPReplayBuffer):
    """Prioritized Replay buffer.
    
    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        
    """
    
    def __init__(
        self, 
        obs_dim: int, 
        param_dim: int,
        tl_dim: int,
        size: int, 
        batch_size: int = 32, 
        alpha: float = 0.6,
        n_step: int = 1, 
        gamma: float = 0.99,
    ):
        """Initialization."""
        assert alpha >= 0
        
        super(PrioritizedReplayBuffer, self).__init__(
            obs_dim, param_dim, tl_dim, size, batch_size, n_step, gamma
        )
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
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
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        """Store experience and priority."""
        transition = super().store(obs, tl_code, act, act_param, rew, next_obs, next_tl_code, done)
        
        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size
        
        return transition

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0
        
        indices = self._sample_proportional()
        
        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        tl_codes = self.tl_buf[indices]
        next_tl_codes = self.next_tl_buf[indices]
        acts = self.acts_buf[indices]
        act_params = self.acts_param_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        
        return dict(
            obs=obs,
            next_obs=next_obs,
            tl_code=tl_codes,
            next_tl_code=next_tl_codes,
            act=acts,
            act_param=act_params,
            rew=rews,
            done=done,
            weights=weights,
            indices=indices,
        )
        
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight