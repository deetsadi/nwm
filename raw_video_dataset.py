from torch.utils.data import Dataset as TorchDataset
import json
import math
import os
import cv2
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

_UINT8_MAX_F = float(torch.iinfo(torch.uint8).max)
class RawVideoDataset(TorchDataset):
    def __init__(self, data_dir, image_size, context_frames=8, max_target=20, n_output=1, stride=4, skip_frame=0, n_goals=1, k_min=29, k_max=89, with_actions = True):
        super().__init__()
        # mp.set_start_method('spawn')

        self.data_dir = Path(data_dir)
        self.image_size = image_size

        self.context_frames = context_frames
        self.max_target_range = max_target
        
        # Load main metadata
        with open(self.data_dir / "metadata.json") as f:
            metadata = json.load(f)
            self.num_shards = metadata["num_shards"]
            self.query = metadata["query"]
            self.hz = metadata["hz"]
            self.num_images = metadata["num_images"]
        
        # Load shard-specific metadata
        self.shard_sizes = []
        for shard in range(self.num_shards):
            with open(self.data_dir / f"metadata/metadata_{shard}.json") as f:
                shard_metadata = json.load(f)
                self.shard_sizes.append(shard_metadata["shard_num_frames"])
        
        # Calculate cumulative shard sizes for index mapping
        self.cumulative_sizes = np.cumsum([0] + self.shard_sizes)
        assert self.cumulative_sizes[-1] == self.num_images, "Metadata mismatch in total number of frames"
        
        # Store video paths instead of keeping captures open
        self.video_paths = [
            str(self.data_dir / f"videos/video_{shard}.mp4")
            for shard in range(self.num_shards)
        ]
        
        # Store action paths, and open them up if with_actions
        self.action_paths = [
            str(self.data_dir / f"robot_states/states_{shard}.bin")
            for shard in range(self.num_shards)
        ]
        self.with_actions = with_actions
        if self.with_actions:
            self.action_shards = []
            for i,path in enumerate(self.action_paths):
                action_shard = np.memmap(path, dtype=np.float32, mode="r", shape=(self.shard_sizes[i], 25))
                self.action_shards.append(
                    action_shard
                )
            self.action_shards = np.concatenate(self.action_shards, 0)
            self.action_mean = np.mean(self.action_shards, 0)
            self.action_std = np.std(self.action_shards, 0)
            self.action_shards = (self.action_shards - self.action_mean) / self.action_std
        
        # Store action paths, and open them up if with_actions
        self.segment_paths = [
            str(self.data_dir / f"segment_indices/segment_idx_{shard}.bin")
            for shard in range(self.num_shards)
        ]
        
        segment_shards = []
        for i,path in enumerate(self.segment_paths):
            segment_shard = np.memmap(path, dtype=np.int32, mode="r", shape=(self.shard_sizes[i], 1))
            segment_shards.append(segment_shard)
        self.segment_shards = np.concatenate(segment_shards).squeeze(-1)
        
        self.n_goals = n_goals
        self.n_input, self.n_output, self.stride, self.skip_frame = context_frames, n_output, stride, skip_frame
        if self.n_goals > 1:
            self.k_min = k_min
            self.k_max = k_max
            self.video_len = (self.n_output + self.n_input + self.k_max + self.n_output)
        else:
            # Number of frames between the first and last frames of a video sequence (excluding one endpoint frame)
            self.video_len = (self.n_output + self.n_input + self.skip_frame) # * self.skip_frame 
            print (f"Video length: {self.video_len}")
        
        start_indices = np.arange(0, self.num_images - self.video_len, self.stride)
        end_indices = start_indices + self.video_len
        
        start_segment_ids = self.segment_shards[start_indices] # 
        end_segment_ids = self.segment_shards[end_indices] # 

        # Verify all video files exist and are readable
        for path in self.video_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Video file not found: {path}")
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                cap.release()
                raise IOError(f"Could not open video file: {path}")
            cap.release()
        
        # Verify all action files exist and are readable
        for path in self.action_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Video file not found: {path}")
        
        self.valid_start_inds = start_indices[start_segment_ids == end_segment_ids]


    def get_index_info(self, idx):
        shard_idx = np.searchsorted(self.cumulative_sizes[1:], idx, side='right')
        frame_idx = idx - self.cumulative_sizes[shard_idx]
        return shard_idx, frame_idx 
    
    def extract_frames_opencv(self, start_shard, end_shard, start_frame, end_frame):
        if start_shard == end_shard:
            # if the shards are same then we can get all frames from the same video
            video_path = self.video_paths[start_shard]
            start_cap = cv2.VideoCapture(video_path)
            start_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Jump to start frame
            
            frames = []
            for frame_idx in range(start_frame, end_frame):
                ret, frame = start_cap.read()
                if not ret:
                    break
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
                frame = cv2.resize(frame, (self.image_size, self.image_size))
                frames.append(frame)            
            start_cap.release()
            return np.array(frames)  # Shape: (num_frames, H, W, 3)
        else:
            # if the shards are different then we need to get the video frames from different videos
            frames = []
            start_cap = cv2.VideoCapture(self.video_paths[start_shard])
            start_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for frame_idx in range(start_frame, self.shard_sizes[start_shard]):
                ret, frame = start_cap.read()
                if not ret:
                    break
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
                frame = cv2.resize(frame, (self.image_size, self.image_size))
                frames.append(frame)
            start_cap.release()

            end_cap = cv2.VideoCapture(self.video_paths[end_shard])
            end_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for frame_idx in range(0, end_frame):
                ret, frame = end_cap.read()
                if not ret:
                    break
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
                frame = cv2.resize(frame, (self.image_size, self.image_size))
                frames.append(frame)
            
            end_cap.release()
            return np.array(frames)  

    def __len__(self):
        return len(self.valid_start_inds)

    def __getitem__(self, idx):
        if self.n_goals <= 1:
            return self.get_single_goal_frame(idx)
        else:
            return self.get_multiple_goal_frames(idx)
    
    def get_single_goal_frame(self, idx):
        if idx < 0 or idx >= len(self.valid_start_inds):
            raise IndexError(f"Index {idx} is out of bounds for dataset with {self.num_images} images")

        start_idx = self.valid_start_inds[idx]
        end_idx = start_idx + self.video_len
        start_shard_idx, start_frame_idx = self.get_index_info(start_idx)
        end_shard_idx, end_frame_idx = self.get_index_info(end_idx)
        
        assert self.segment_shards[start_idx] == self.segment_shards[end_idx] 
        shard_idx = start_shard_idx
        
        frames = self.extract_frames_opencv(
            start_shard_idx,
            end_shard_idx,
            start_frame_idx, 
            end_frame_idx
        )
        
        frames = np.moveaxis(frames, 3, 0)
        frames = frames/_UINT8_MAX_F * 2.0 - 1.0

        assert frames.shape[1] == self.n_input + self.n_output + self.skip_frame
        
        past_frames = frames[:, :self.n_input]
        future_frame = frames[:, self.n_input + self.skip_frame:]
        assert future_frame.shape[1] == self.n_output

        ret = {
            "past_frames": past_frames.astype(np.float32), 
            "future_frames": future_frame.astype(np.float32)
            }

        if self.with_actions:
            ret["past_actions"] = self.action_shards[start_idx:start_idx+self.n_input].astype(np.float32)
            ret["future_actions"] = self.action_shards[start_idx + self.n_input:start_idx + self.n_input + self.skip_frame + self.n_output].astype(np.float32)
            actions = torch.tensor(np.concatenate([ret["past_actions"][-1:], ret["future_actions"][:-1]], axis=0))
            # ret["pooled_actions"] = torch.sum(actions, dim=0) / self.action_mean
            ret["actions"] = actions
        
        return (
            torch.cat([torch.tensor(past_frames.astype(np.float32)), torch.tensor(future_frame.astype(np.float32))], dim=1),
            # ret["pooled_actions"], # only returning the last action
            ret["actions"],
            # torch.tensor(ret["past_actions"][-1]), # only returning the last action
            # torch.cat([torch.tensor(ret["past_actions"]), torch.tensor(ret["future_actions"])], dim=0),
            torch.tensor(np.array(self.skip_frame + self.n_output).astype(np.float32)),
        )

    # def build_index(self):
    #     self.index_tuples = []
    #     print ("Building index...")
    #     for segment_id in tqdm(np.unique(self.segment_shards)):
    #         # Get all indices for this segment
    #         segment_indices = np.where(self.segment_shards == segment_id)[0]
    #         if len(segment_indices) == 0:
    #             continue
            
    #         shard_idx, frame_idx = self.get_index_info(segment_indices[0])
    #         segment_length = len(segment_indices)

    #         for i in range(segment_length - self.context_frames):
    #             context_start_global_idx = segment_indices[i]
    #             context_end_global_idx = segment_indices[i + self.context_frames]

    #             target_min_idx = context_end_global_idx + self.k_min
    #             target_max_idx = min(context_end_global_idx + self.k_max, segment_indices[-1])

    #             if target_max_idx >= target_min_idx:
    #                 # Record (shard index, start frame index, global target min idx, global target max idx)
    #                 self.index_tuples.append((
    #                     *self.get_index_info(context_start_global_idx),
    #                     target_min_idx,
    #                     target_max_idx
    #                 ))

    def get_multiple_goal_frames(self, idx):
        if idx < 0 or idx >= len(self.valid_start_inds):
            raise IndexError(f"Index {idx} is out of bounds for dataset with {self.num_images} images")

        start_idx = self.valid_start_inds[idx]
        start_shard_idx, start_frame_idx = self.get_index_info(start_idx)

        # Sample multiple goal offsets
        goal_offsets = np.random.randint(self.k_min - self.n_input, self.k_max - self.n_input, size=self.n_goals)

        goal_frames_list = []
        future_actions_list = []

        # Extract past frames once
        end_idx_for_past = start_idx + self.n_input + self.k_max + self.n_output
        end_shard_idx_past, end_frame_idx_past = self.get_index_info(end_idx_for_past)

        frames_full = self.extract_frames_opencv(
            start_shard_idx,
            end_shard_idx_past,
            start_frame_idx,
            end_frame_idx_past
        )
        frames_full = np.moveaxis(frames_full, 3, 0)
        frames_full = frames_full / _UINT8_MAX_F * 2.0 - 1.0

        past_frames = frames_full[:, :self.n_input]
        past_frames_tensor = torch.tensor(past_frames.astype(np.float32))
        
        for goal_offset in goal_offsets:
            goal_start = self.n_input + goal_offset
            goal_end = goal_start + self.n_output
            future_frame = frames_full[:, goal_start:goal_end]
            # Ensure shape is correct
            if future_frame.shape[1] != self.n_output:
                continue  # skip invalid samples

            goal_frames_list.append(torch.tensor(future_frame.astype(np.float32)))

            if self.with_actions:
                future_actions = [self.action_shards[self.n_input:goal_start].astype(np.float32)]
                future_actions_list.append(future_actions)

        if not goal_frames_list:
            raise ValueError("No valid goal frames found for this index.")

        goal_frames_tensor = torch.stack(goal_frames_list, dim=1)  # [C, num_goals, T, H, W]
        assert [x for x in goal_frames_tensor.shape] == [3, self.n_goals, 1, self.image_size, self.image_size]

        ret = {
            "past_frames": past_frames_tensor,
            "future_frames": goal_frames_tensor,
        }

        if self.with_actions:
            past_actions = self.action_shards[start_idx : start_idx + self.n_input].astype(np.float32)
            padding_primitive = [-100 for _ in range(25)]
            action_seq = []
            for goal_seq in future_actions_list:
                for future_actions in goal_seq:
                    future_actions = np.array(future_actions)
                    actual_length = len(future_actions)  # last past + future
                    desired_length = 1 + self.k_max - self.n_input
                    num_pads = desired_length - actual_length

                    padding = np.array([padding_primitive] * num_pads, dtype=np.float32)
                    # print (padding.shape, future_actions.shape, self.n_input, self.k_max)
                    full_action_seq = np.concatenate([future_actions, padding], axis=0)

                    action_seq.append([full_action_seq])

            actions = torch.tensor(np.concatenate(action_seq, axis=0).astype(np.float32))
            ret["actions"] = actions

        all_frames = torch.cat([
                past_frames_tensor.unsqueeze(1).expand(-1, goal_frames_tensor.shape[1], -1, -1, -1),
                goal_frames_tensor
            ], dim=2)  # [C, num_goals, T_past + T_future, H, W]
        return (
            all_frames.reshape((all_frames.shape[1], all_frames.shape[0], all_frames.shape[2], all_frames.shape[3], all_frames.shape[4])),
            ret.get("actions", None),
            torch.tensor(goal_offsets.astype(np.float32)),
        )

    def get_frame_info(self, idx):
        """Helper method to debug frame locations"""
        shard_idx = np.searchsorted(self.cumulative_sizes[1:], idx, side='right')
        frame_idx = idx - self.cumulative_sizes[shard_idx]
        return {
            "global_index": idx,
            "shard_index": shard_idx,
            "frame_index": frame_idx,
            "video_path": self.video_paths[shard_idx]
        }


if __name__ == "__main__":
    ds = RawVideoDataset("/home/aditya/nwm/data/train_v2.0_raw", 256, skip_frame=8, n_goals=2, with_actions=True, k_min=10, k_max=15)
    print (len(ds))
    for i in tqdm(range(len(ds))):
        d = ds[i]
        # print (d[1][0])
