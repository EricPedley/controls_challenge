from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTEXT_LENGTH, MAX_ACC_DELTA, CONTROL_START_IDX, STEER_RANGE, State, FuturePlan, COST_END_IDX, LAT_ACCEL_COST_MULTIPLIER , DEL_T, FUTURE_PLAN_STEPS, VOCAB_SIZE, LataccelTokenizer, LATACCEL_RANGE
import numpy as np
from gymnasium.vector import VectorEnv
from gymnasium.spaces import Box
from controllers import BaseController
from pathlib import Path
import torch
import pandas as pd
import onnxruntime as ort

class BatchedTinyPhysicsModel:
    """Batched version of TinyPhysicsModel that uses PyTorch tensors and GPU"""
    def __init__(self, model_path: str, device='cuda'):
        self.tokenizer = LataccelTokenizer()
        self.device = device
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        provider = 'CUDAExecutionProvider' if device == 'cuda' and torch.cuda.is_available() else 'CPUExecutionProvider'
        
        with open(model_path, "rb") as f:
            self.ort_session = ort.InferenceSession(f.read(), options, [provider])
    
    def softmax(self, x, axis=-1):
        x_max = torch.max(x, dim=axis, keepdim=True)[0]
        e_x = torch.exp(x - x_max)
        return e_x / torch.sum(e_x, dim=axis, keepdim=True)
    
    def predict_batched(self, states: torch.Tensor, tokens: torch.Tensor, temperature=1.0):
        """
        Batched prediction using IO Binding to keep tensors on GPU
        Args:
            states: (batch_size, context_length, 4) - [action, roll_lataccel, v_ego, a_ego]
            tokens: (batch_size, context_length) - tokenized past predictions
        Returns:
            sampled tokens: (batch_size,) - sampled lataccel tokens
        """
        if self.device == 'cuda':
            # Use IO Binding to avoid CPU roundtrips
            io_binding = self.ort_session.io_binding()
            
            # Bind inputs - tensors stay on GPU
            states_contiguous = states.contiguous()
            tokens_contiguous = tokens.contiguous()
            io_binding.bind_input(
                name='states',
                device_type='cuda',
                device_id=0,
                element_type=np.float32,
                shape=tuple(states_contiguous.shape),
                buffer_ptr=states_contiguous.data_ptr()
            )
            io_binding.bind_input(
                name='tokens',
                device_type='cuda',
                device_id=0,
                element_type=np.int64,
                shape=tuple(tokens_contiguous.shape),
                buffer_ptr=tokens_contiguous.data_ptr()
            )
            
            # Bind output - allocate on GPU
            output_shape = (states.shape[0], states.shape[1], VOCAB_SIZE)
            output_tensor = torch.empty(output_shape, dtype=torch.float32, device=self.device)
            io_binding.bind_output(
                name='output',
                device_type='cuda',
                device_id=0,
                element_type=np.float32,
                shape=tuple(output_tensor.shape),
                buffer_ptr=output_tensor.data_ptr()
            )
            
            # Run inference - everything stays on GPU
            self.ort_session.run_with_iobinding(io_binding)
            
            # Output is already a torch tensor on GPU
            probs = self.softmax(output_tensor / temperature, axis=-1)
        else:
            # Fallback to CPU path
            states_np = states.cpu().numpy().astype(np.float32)
            tokens_np = tokens.cpu().numpy().astype(np.int64)
            res = self.ort_session.run(None, {'states': states_np, 'tokens': tokens_np})[0]
            res_torch = torch.from_numpy(res).to(self.device)
            probs = self.softmax(res_torch / temperature, axis=-1)
        
        # Sample from the last timestep for each batch
        # probs shape: (batch_size, context_length, vocab_size)
        last_probs = probs[:, -1, :]  # (batch_size, vocab_size)
        
        # Sample for each environment
        samples = torch.multinomial(last_probs, num_samples=1).squeeze(-1)  # (batch_size,)
        return samples

class TinyPhysicsEnv(VectorEnv):
    def __init__(self, num_envs=1, device='cuda'):
        self.num_agents = 1
        self.num_envs = num_envs
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        self.sim_model = BatchedTinyPhysicsModel('models/tinyphysics.onnx', device=self.device)
        self.reward_threshold = -1000
        
        self.policy_history_len = 4
        self.lookahead_len = 4
        state_low = [-1, 0, -0.1]
        state_high = [1, 60, 0.1]
        actions_low = [-2]
        actions_high = [2]
        preds_low = [-2]
        preds_high = [2]
        self.all_data_paths = list(Path('data').glob('*.csv'))
        
        self.observation_space = Box(
            np.concatenate([
                np.tile(state_low, self.policy_history_len),
                np.tile(actions_low, self.policy_history_len),
                np.tile(preds_low, self.policy_history_len),
                np.tile(preds_low, self.policy_history_len),
                np.repeat(preds_low + state_low, self.lookahead_len)  # future plan
            ]),
            np.concatenate([
                np.tile(state_high, self.policy_history_len),
                np.tile(actions_high, self.policy_history_len),
                np.tile(preds_high, self.policy_history_len),
                np.tile(preds_high, self.policy_history_len),
                np.repeat(preds_high + state_high, self.lookahead_len)  # future plan
            ]),
        )
        self.action_space = Box(-2, 2, (1,))
        
        # Load all data upfront as tensors
        self.data_tensors = [None] * self.num_envs  # Will hold dict of tensors per env
        self.data_lengths = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # History buffers (cyclic) - shape: (num_envs, context_length, ...)
        self.state_history = torch.zeros(self.num_envs, CONTEXT_LENGTH, 3, device=self.device)  # [roll_lataccel, v_ego, a_ego]
        self.action_history = torch.zeros(self.num_envs, CONTEXT_LENGTH, device=self.device)
        self.current_lataccel_history = torch.zeros(self.num_envs, CONTEXT_LENGTH, device=self.device)
        self.target_lataccel_history = torch.zeros(self.num_envs, CONTEXT_LENGTH, device=self.device)
        
        # Current state
        self.step_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.current_lataccel = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.previous_pred = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.cyclic_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # Current position in cyclic buffer


        self.tokenizer_bins = torch.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE,device=self.device)
        
        self.reset()
    
    def _load_data_to_tensors(self, data_path: str):
        """Load CSV data and convert to tensors on device"""
        df = pd.read_csv(data_path)
        ACC_G = 9.81
        
        data_dict = {
            'roll_lataccel': torch.tensor(np.sin(df['roll'].values) * ACC_G, dtype=torch.float32, device=self.device),
            'v_ego': torch.tensor(df['vEgo'].values, dtype=torch.float32, device=self.device),
            'a_ego': torch.tensor(df['aEgo'].values, dtype=torch.float32, device=self.device),
            'target_lataccel': torch.tensor(df['targetLateralAcceleration'].values, dtype=torch.float32, device=self.device),
            'steer_command': torch.tensor(-df['steerCommand'].values, dtype=torch.float32, device=self.device)  # right-positive
        }
        return data_dict, len(df)
    
    def _get_cyclic_history_view(self, buffer, env_idx):
        """Get properly ordered view of cyclic buffer for a specific env"""
        # Returns the last CONTEXT_LENGTH items in the right order
        idx = self.cyclic_idx[env_idx].item()
        if buffer.dim() == 2:  # 1D history per env (actions, lataccels)
            rolled = torch.roll(buffer[env_idx], shifts=-idx, dims=0)
            return rolled
        else:  # 2D history per env (states)
            rolled = torch.roll(buffer[env_idx], shifts=-idx, dims=0)
            return rolled
    
    def step(self, actions: torch.Tensor):
        """Vectorized step for all environments in parallel"""
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self.device)
        
        assert actions.shape[0] == self.num_envs
        if actions.dim() == 2:
            actions = actions.squeeze(-1)  # (num_envs,)
        
        for env_index, step_index in enumerate(self.step_indices):
            if step_index >= self.data_tensors[env_index]['roll_lataccel'].shape[0]:
                self._reset_index(env_index)
            
        
        # Get current states and targets for all envs
        states = torch.stack([
            torch.stack([
                self.data_tensors[i]['roll_lataccel'][self.step_indices[i]],
                self.data_tensors[i]['v_ego'][self.step_indices[i]],
                self.data_tensors[i]['a_ego'][self.step_indices[i]]
            ], dim=0)
            for i in range(self.num_envs)
        ])  # (num_envs, 3)
        
        targets = torch.stack([
            self.data_tensors[i]['target_lataccel'][self.step_indices[i]]
            for i in range(self.num_envs)
        ])  # (num_envs,)
        
        # Update cyclic buffers
        write_idx = self.cyclic_idx % CONTEXT_LENGTH
        for i in range(self.num_envs):
            self.state_history[i, write_idx[i]] = states[i]
            self.target_lataccel_history[i, write_idx[i]] = targets[i]
        
        # Determine actions (use logged actions before CONTROL_START_IDX)
        before_control = self.step_indices < CONTROL_START_IDX
        logged_actions = torch.stack([
            self.data_tensors[i]['steer_command'][self.step_indices[i]]
            for i in range(self.num_envs)
        ])
        actions = torch.where(before_control, logged_actions, actions)
        actions = torch.clamp(actions, STEER_RANGE[0], STEER_RANGE[1])
        
        # Update action history
        for i in range(self.num_envs):
            self.action_history[i, write_idx[i]] = actions[i]
        
        # Prepare batched input for model
        # Get ordered history for each env
        states_batch = torch.stack([self._get_cyclic_history_view(self.state_history, i) for i in range(self.num_envs)])
        actions_batch = torch.stack([self._get_cyclic_history_view(self.action_history, i) for i in range(self.num_envs)])
        past_preds_batch = torch.stack([self._get_cyclic_history_view(self.current_lataccel_history, i) for i in range(self.num_envs)])
        
        # Combine into model input: (num_envs, context_length, 4) = [action, roll_lataccel, v_ego, a_ego]
        model_states = torch.cat([
            actions_batch.unsqueeze(-1),  # (num_envs, context_length, 1)
            states_batch  # (num_envs, context_length, 3)
        ], dim=-1)  # (num_envs, context_length, 4)
        
        # Tokenize past predictions
        past_preds_np = past_preds_batch.cpu().numpy()
        tokens = self.sim_model.tokenizer.encode(past_preds_np)  # (num_envs, context_length)
        tokens_torch = torch.from_numpy(tokens).to(self.device)

        past_preds_np = past_preds_batch#.cpu().numpy()
        clipped = torch.clip(past_preds_np, LATACCEL_RANGE[0], LATACCEL_RANGE[1])
        tokens_torch = torch.searchsorted(self.tokenizer_bins, clipped, side='left')#.reshape_as(clipped)
        
        # Batched model prediction
        pred_tokens = self.sim_model.predict_batched(model_states, tokens_torch, temperature=0.8)
        pred = self.tokenizer_bins[pred_tokens]
        
        # Clip predictions by MAX_ACC_DELTA
        pred = torch.clamp(pred, self.current_lataccel - MAX_ACC_DELTA, self.current_lataccel + MAX_ACC_DELTA)
        
        # Update current lataccel (use pred after CONTROL_START_IDX, else use target)
        self.current_lataccel = torch.where(before_control, targets, pred)
        
        # Update current lataccel history
        for i in range(self.num_envs):
            self.current_lataccel_history[i, write_idx[i]] = self.current_lataccel[i]
        
        # Increment step indices and cyclic idx
        self.step_indices += 1
        self.cyclic_idx += 1
        
        # Build observations
        obs_list = []
        for i in range(self.num_envs):
            # Get last policy_history_len items from cyclic buffers
            state_hist = self._get_cyclic_history_view(self.state_history, i)[-self.policy_history_len:]
            action_hist = self._get_cyclic_history_view(self.action_history, i)[-self.policy_history_len:]
            current_lataccel_hist = self._get_cyclic_history_view(self.current_lataccel_history, i)[-self.policy_history_len:]
            target_lataccel_hist = self._get_cyclic_history_view(self.target_lataccel_history, i)[-self.policy_history_len:]
            
            # Get future plan
            start_idx = self.step_indices[i]
            end_idx = min(start_idx + FUTURE_PLAN_STEPS, self.data_lengths[i])
            plan_len = end_idx - start_idx
            
            future_plan = torch.zeros(4, FUTURE_PLAN_STEPS, device=self.device)
            if plan_len > 0:
                future_plan[0, :plan_len] = self.data_tensors[i]['target_lataccel'][start_idx:end_idx]
                future_plan[1, :plan_len] = self.data_tensors[i]['roll_lataccel'][start_idx:end_idx]
                future_plan[2, :plan_len] = self.data_tensors[i]['v_ego'][start_idx:end_idx]
                future_plan[3, :plan_len] = self.data_tensors[i]['a_ego'][start_idx:end_idx]
            
            obs = torch.cat([
                state_hist.flatten(),
                action_hist,
                current_lataccel_hist,
                target_lataccel_hist,
                future_plan[:, :self.lookahead_len].flatten()
            ])
            obs_list.append(obs)
        
        obs = torch.stack(obs_list)#.cpu().numpy()  # (num_envs, obs_dim)
        
        # Compute rewards
        lat_accel_cost = (targets - pred) ** 2
        jerk_cost = ((pred - self.previous_pred) / DEL_T) ** 2
        self.previous_pred = pred.clone()
        total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
        
        rewards = torch.where(self.step_indices > CONTROL_START_IDX, -total_cost, torch.zeros_like(total_cost))

        terminated = rewards < self.reward_threshold
        for idx_to_reset in torch.nonzero(terminated):
            self._reset_index(idx_to_reset)
        
        # Check termination
        truncated = (self.step_indices >= self.data_lengths)#.cpu().numpy()
        infos = [{} for _ in range(self.num_envs)]
        
        return obs, rewards.reshape((self.num_envs, -1)), terminated.reshape((self.num_envs, -1)), truncated.reshape((self.num_envs, -1)), infos
    
    def _reset_index(self, env_index: int, seed=None):
        """Reset a specific environment"""
        assert env_index < self.num_envs
        if seed is not None:
            np.random.seed(seed + env_index)
        
        data_path = np.random.choice(self.all_data_paths)
        self.data_tensors[env_index], data_len = self._load_data_to_tensors(str(data_path))
        self.data_lengths[env_index] = data_len
        
        # Reset to CONTEXT_LENGTH
        self.step_indices[env_index] = CONTEXT_LENGTH
        self.cyclic_idx[env_index] = CONTEXT_LENGTH
        
        # Initialize history from data
        for t in range(CONTEXT_LENGTH):
            self.state_history[env_index, t, 0] = self.data_tensors[env_index]['roll_lataccel'][t]
            self.state_history[env_index, t, 1] = self.data_tensors[env_index]['v_ego'][t]
            self.state_history[env_index, t, 2] = self.data_tensors[env_index]['a_ego'][t]
            self.action_history[env_index, t] = self.data_tensors[env_index]['steer_command'][t]
            self.current_lataccel_history[env_index, t] = self.data_tensors[env_index]['target_lataccel'][t]
            self.target_lataccel_history[env_index, t] = self.data_tensors[env_index]['target_lataccel'][t]
        
        self.current_lataccel[env_index] = self.data_tensors[env_index]['target_lataccel'][CONTEXT_LENGTH - 1].type_as(self.current_lataccel) # TODO: figure out why the type cast here is even necessary
        self.previous_pred[env_index] = self.current_lataccel[env_index]
    
    def reset(self, seed=None, options=None):
        """Reset all environments"""
        if seed is not None:
            np.random.seed(seed)
        
        for i in range(self.num_envs):
            self._reset_index(i, seed=seed)
        
        # Build initial observations
        obs_list = []
        for i in range(self.num_envs):
            # Get last policy_history_len items
            state_hist = self.state_history[i, -self.policy_history_len:]
            action_hist = self.action_history[i, -self.policy_history_len:]
            current_lataccel_hist = self.current_lataccel_history[i, -self.policy_history_len:]
            target_lataccel_hist = self.target_lataccel_history[i, -self.policy_history_len:]
            
            # Get future plan
            start_idx = self.step_indices[i]
            end_idx = min(start_idx + FUTURE_PLAN_STEPS, self.data_lengths[i])
            plan_len = end_idx - start_idx
            
            future_plan = torch.zeros(4, FUTURE_PLAN_STEPS, device=self.device)
            if plan_len > 0:
                future_plan[0, :plan_len] = self.data_tensors[i]['target_lataccel'][start_idx:end_idx]
                future_plan[1, :plan_len] = self.data_tensors[i]['roll_lataccel'][start_idx:end_idx]
                future_plan[2, :plan_len] = self.data_tensors[i]['v_ego'][start_idx:end_idx]
                future_plan[3, :plan_len] = self.data_tensors[i]['a_ego'][start_idx:end_idx]
            
            obs = torch.cat([
                state_hist.flatten(),
                action_hist,
                current_lataccel_hist,
                target_lataccel_hist,
                future_plan[:, :self.lookahead_len].flatten()
            ])
            obs_list.append(obs)
        
        obs = torch.stack(obs_list)#.cpu().numpy()
        infos = [{} for _ in range(self.num_envs)]
        
        return obs, infos




