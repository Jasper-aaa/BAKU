import random
import numpy as np
import pickle as pkl
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset,IterableDataset
import numpy as np
import cv2


def flexible_grid_image(frames):
    """
    frames: 按时间顺序给的历史帧 [ ..., f(i-3), f(i-2), f(i-1) ]
            不包括当前帧 f(i)，最后一格固定放 f(i-1)
    return: grid 图 (H, W, C)
    """

    n = len(frames)
    H, W, C = frames[0].shape

    if n == 1:
        return frames[0]

    elif n == 2:
        # 两张图横拼
        cell_w = W // 2
        resized = [cv2.resize(f, (cell_w, H)) for f in frames]
        return np.concatenate(resized, axis=1)

    elif n == 3:
        # 3帧：补一帧，但右下角强制是 f(i-1)
        f1, f2, f3 = frames
        frames_fixed = [f1, f2, f3, f3]  # 补最后一帧，但右下角还是 f(i-1)

        cell_h, cell_w = H // 2, W // 2
        resized = [cv2.resize(f, (cell_w, cell_h)) for f in frames_fixed]
        top = np.concatenate(resized[:2], axis=1)
        bot = np.concatenate(resized[2:], axis=1)
        return np.concatenate([top, bot], axis=0)

    elif n == 4:
        # 4帧：直接做 2×2，保证最后一格是 f(i-1)
        f1, f2, f3, f4 = frames
        frames_fixed = [f1, f2, f3, f4]  # f4 == f(i-1)

        cell_h, cell_w = H // 2, W // 2
        resized = [cv2.resize(f, (cell_w, cell_h)) for f in frames_fixed]
        top = np.concatenate(resized[:2], axis=1)
        bot = np.concatenate(resized[2:], axis=1)
        return np.concatenate([top, bot], axis=0)

    else:
        raise ValueError("Only support up to 4 history frames.")

class BCDataset(IterableDataset):
    def __init__(
        self,
        path,
        suite,
        scenes,
        tasks,
        num_demos_per_task,
        obs_type,
        history,
        history_len,
        prompt,
        temporal_agg,
        num_queries,
        img_size,
        store_actions=False,
        his_grid_aug=False,
    ):
        self._obs_type = obs_type
        self._prompt = prompt
        self._history = history
        self._history_len = history_len if history else 1
        self.img_size = img_size

        # temporal_aggregation
        self._temporal_agg = temporal_agg
        self._num_queries = num_queries

        # Convert task_names, which is a list, to a dictionary
        tasks = {task_name: scene[task_name] for scene in tasks for task_name in scene}

        # Get relevant task names
        task_name = []
        for scene in scenes:
            task_name.extend([task_name for task_name in tasks[scene]])

        # get data paths
        self._paths = []
        # for suite in suites:
        self._paths.extend(list((Path(path) / suite).glob("*")))

        if task_name is not None:
            paths = {}
            idx2name = {}
            for path in self._paths:
                task = str(path).split(".")[0].split("/")[-1]
                if task in task_name:
                    # get idx of task in task_name
                    idx = task_name.index(task)
                    paths[idx] = path
                    idx2name[idx] = task
            del self._paths
            self._paths = paths

        # store actions
        if store_actions:
            self.actions = []

        # read data
        self._episodes = {}
        self._max_episode_len = 0
        self._max_state_dim = 0
        self._num_samples = 0
        for _path_idx in self._paths:
            print(f"Loading {str(self._paths[_path_idx])}")
            # read
            data = pkl.load(open(str(self._paths[_path_idx]), "rb"))
            observations = (
                data["observations"] if self._obs_type == "pixels" else data["states"]
            )
            actions = data["actions"]
            task_emb = data["task_emb"]
            # store
            self._episodes[_path_idx] = []
            for i in range(min(num_demos_per_task, len(observations))):
                episode = dict(
                    observation=observations[i],
                    action=actions[i],
                    task_emb=task_emb,
                )
                self._episodes[_path_idx].append(episode)
                self._max_episode_len = max(
                    self._max_episode_len,
                    (
                        len(observations[i])
                        if not isinstance(observations[i], dict)
                        else len(observations[i]["pixels"])
                    ),
                )
                # if obs_type == 'features':
                self._max_state_dim = max(
                    self._max_state_dim, data["states"][i].shape[-1]
                )
                self._num_samples += (
                    len(observations[i])
                    if self._obs_type == "features"
                    else len(observations[i]["pixels"])
                )

                # store actions
                if store_actions:
                    self.actions.append(actions[i])

        self.stats = {
            "actions": {
                "min": 0,
                "max": 1,
            },
            "proprioceptive": {
                "min": 0,
                "max": 1,
            },
        }
        self.preprocess = {
            "actions": lambda x: (x - self.stats["actions"]["min"])
            / (self.stats["actions"]["max"] - self.stats["actions"]["min"] + 1e-5),
            "proprioceptive": lambda x: (x - self.stats["proprioceptive"]["min"])
            / (
                self.stats["proprioceptive"]["max"]
                - self.stats["proprioceptive"]["min"]
                + 1e-5
            ),
        }

        # augmentation
        self.aug = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )

        # Samples from envs
        self.envs_till_idx = len(self._episodes)

        # ImageGrid Augmentation 
        self._his_grid_aug = his_grid_aug
    def _sample_episode(self, env_idx=None):
        idx = random.randint(0, self.envs_till_idx - 1) if env_idx is None else env_idx
        episode = random.choice(self._episodes[idx])
        return (episode, idx) if env_idx is None else episode

    def _sample(self):
        # prepare for grid image process in film layer
        # history_len = self._history_len
        expected_len = self._history_len +1 

        episodes, env_idx = self._sample_episode()
        observations = episodes["observation"]
        actions = episodes["action"]
        task_emb = episodes["task_emb"]
        
        sample_idx = np.random.randint(
            0, len(observations["pixels"])
        )
        # for history sampling
        previous_idx = sample_idx-self._history_len if sample_idx-self._history_len > 0 else 0
        
        # for time embedding
        frame_indices = list(range(previous_idx, sample_idx + 1))

        # sampled actions
        if self._temporal_agg:
            # we just predict the future num_queries steps
            sampled_action = np.zeros(
                (self._num_queries, actions.shape[-1])
            )
            end = min(len(actions),sample_idx+self._num_queries)
            sampled_action[:end - sample_idx] = actions[sample_idx:end]
        else:
            sampled_action = actions[sample_idx]

        
        
        curr_pixel = self.aug(observations['pixels'][
                sample_idx
            ]).unsqueeze(0)
        
        curr_pixel_egocentric = self.aug(observations["pixels_egocentric"][
                sample_idx
            ]).unsqueeze(0)
        
        # if not using history
        if not self._history:
            sampled_proprioceptive_state =  np.concatenate(
                [
                    observations["joint_states"][
                        sample_idx:sample_idx+1
                    ],
                    observations["gripper_states"][
                        sample_idx:sample_idx+1
                    ],
                ],
                axis=-1,
            )
            expected_len = len(curr_pixel.shape[0])
            return{
                "pixels": curr_pixel,
                "pixels_egocentric": curr_pixel_egocentric,
                "proprioceptive": self.preprocess["proprioceptive"](
                    sampled_proprioceptive_state
                ),
                "actions": self.preprocess["actions"](sampled_action),
                "task_emb": task_emb,
                "image_len": expected_len, 
                'frame_indices': None,
                'current_index': None,
            }
        
        his_pixels = observations['pixels'][
                previous_idx:sample_idx
            ]
        
        his_pixels_egocentric = observations['pixels_egocentric'][
                previous_idx:sample_idx
            ]
        # if self._his_grid_aug:
        #     his_grid_pixels = []
        #     his_grid_pixels_egocentric = []
        #     # make grid
        #     for k in range(0,history_len,4):
        #         # every 4 franes one grid
        #         group_his_pixel = his_pixels[k:k+4]
        #         group_his_pixel_egocentric = his_pixels_egocentric[k:k+4]
        #         # avoid sample index = 0
        #         if len(group_his_pixel) < 1 :
        #             break
        #         # grid augmentation
        #         grid_pixel = flexible_grid_image(
        #             group_his_pixel
        #         )
        #         grid_pixel_egocentric = flexible_grid_image(
        #             group_his_pixel_egocentric
        #         )

        #         his_grid_pixels.append(grid_pixel)
        #         his_grid_pixels_egocentric.append(grid_pixel_egocentric)

            
        #     his_pixel = torch.stack([self.aug(grid_pixel) for grid_pixel in his_grid_pixels]) 
        #     his_pixel_egocentric = torch.stack([self.aug(grid_pixel_ego) for grid_pixel_ego in his_grid_pixels_egocentric])

        # else:
        sampled_proprioceptive_state = np.concatenate(
                [
                    observations["joint_states"][
                        previous_idx : sample_idx+1
                    ],
                    observations["gripper_states"][
                        previous_idx: sample_idx+1
                    ],
                ],
                axis=-1,
            )
        curr_state = np.concatenate(
                [
                    observations["joint_states"][sample_idx],
                    observations["gripper_states"][sample_idx],
                ],
                axis=-1,
            )
        
        if len(his_pixels) > 0:
            his_pixel = torch.stack([self.aug(pixel) for pixel in his_pixels]) 
            his_pixel_egocentric = torch.stack([self.aug(pixel_ego) for pixel_ego in his_pixels_egocentric])
            
            sampled_pixel = torch.cat([his_pixel,curr_pixel],dim=0)
            sampled_pixel_egocentric = torch.cat([his_pixel_egocentric,curr_pixel_egocentric],dim=0)
        else:
            sampled_pixel = curr_pixel
            sampled_pixel_egocentric = curr_pixel_egocentric
        # padding 
        sampled_proprioceptive_state = self.pad_to_length_np(sampled_proprioceptive_state, expected_len, curr_state)
        sampled_pixel = self.pad_to_length(sampled_pixel, expected_len, curr_pixel)
        sampled_pixel_ego = self.pad_to_length(sampled_pixel_egocentric, expected_len, curr_pixel_egocentric)

        # frame_indices 也要 pad（用最早的index补齐）
        if len(frame_indices) < expected_len:
            pad_num = expected_len - len(frame_indices)
            frame_indices = [frame_indices[0]] * pad_num + frame_indices

        frame_indices = torch.tensor(frame_indices, dtype=torch.long)   

        assert len(frame_indices) == len(sampled_pixel)
        return {
                "pixels": sampled_pixel,
                "pixels_egocentric": sampled_pixel_ego,
                "proprioceptive": self.preprocess["proprioceptive"](
                    sampled_proprioceptive_state
                ),
                "actions": self.preprocess["actions"](sampled_action),
                "task_emb": task_emb,
                "image_len": expected_len, 
                "frame_indices": frame_indices,     
                "current_index": torch.tensor(sample_idx), 
            }
    
    def sample_test(self,env_idx,step=None):
        episode = self._sample_episode(env_idx)
        observations = episode["observation"]
        actions = episode["action"]
        task_emb = episode["task_emb"]

        if self._obs_type == "pixels":
            pixels_shape = observations["pixels"].shape
        
        prompt_pixel = None
        prompt_pixel_egocentric = None
        prompt_proprioceptive_state = None
        prompt_action = None
        return {
                "prompt_pixels": prompt_pixel,
                "prompt_pixels_egocentric": prompt_pixel_egocentric,
                "prompt_proprioceptive": (
                    self.preprocess["proprioceptive"](prompt_proprioceptive_state)
                    if prompt_proprioceptive_state is not None
                    else None
                ),
                "prompt_actions": (
                    self.preprocess["actions"](prompt_action)
                    if prompt_action is not None
                    else None
                ),
                "task_emb": task_emb,
            }
    def __iter__(self):
        while True:
            yield self._sample()

    def __len__(self):
        return self._num_samples


    def pad_to_length(self, tensor, target_len, pad_value, dim=0):
        """Pad tensor to target_len along given dim with pad_value"""
        if tensor.shape[dim] >= target_len:
            return tensor
        pad_num = target_len - tensor.shape[dim]
        pad_tensor = pad_value.repeat(pad_num, *[1 for _ in tensor.shape[1:]])
        return torch.cat([tensor,pad_tensor], dim=dim)

    def pad_to_length_np(self, array, target_len, pad_value, axis=0):
        """Pad numpy array along axis with pad_value"""
        if array.shape[axis] >= target_len:
            return array
        pad_num = target_len - array.shape[axis]
        pad_block = np.repeat(np.expand_dims(pad_value, axis=axis), pad_num, axis=axis)
        return np.concatenate([array,pad_block], axis=axis)
        

            