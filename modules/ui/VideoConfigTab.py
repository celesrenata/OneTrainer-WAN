from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui import components
from modules.util.ui.UIState import UIState

import customtkinter as ctk


class VideoConfigTab:
    """UI component for video-specific training configuration parameters."""

    def __init__(self, master, train_config: TrainConfig, ui_state: UIState):
        super().__init__()

        self.master = master
        self.train_config = train_config
        self.ui_state = ui_state

        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1)

        self.scroll_frame = None

        self.refresh_ui()

    def refresh_ui(self):
        if self.scroll_frame:
            self.scroll_frame.destroy()

        self.scroll_frame = ctk.CTkScrollableFrame(self.master, fg_color="transparent")
        self.scroll_frame.grid(row=0, column=0, sticky="nsew")
        self.scroll_frame.grid_columnconfigure(0, weight=1)

        # Only show video config for video models
        if self.train_config.model_type.is_hunyuan_video() or self.train_config.model_type.is_wan() or self.train_config.model_type.is_hi_dream():
            self.__create_video_config_frame()

    def __create_video_config_frame(self):
        frame = ctk.CTkFrame(master=self.scroll_frame, corner_radius=5)
        frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)

        row = 0

        # Video Data Processing Parameters
        components.label(frame, row, 0, "Max Frames",
                         tooltip="Maximum number of frames to use for training. Higher values require more memory.")
        components.entry(frame, row, 1, self.ui_state, "video_config.max_frames")
        row += 1

        components.label(frame, row, 0, "Frame Sample Strategy",
                         tooltip="Strategy for sampling frames from videos: uniform (evenly spaced), random (randomly selected), keyframe (prefer keyframes)")
        components.options(frame, row, 1, ["uniform", "random", "keyframe"], self.ui_state, "video_config.frame_sample_strategy")
        row += 1

        components.label(frame, row, 0, "Target FPS",
                         tooltip="Target frames per second for video processing")
        components.entry(frame, row, 1, self.ui_state, "video_config.target_fps")
        row += 1

        components.label(frame, row, 0, "Max Duration (seconds)",
                         tooltip="Maximum duration of video clips to process")
        components.entry(frame, row, 1, self.ui_state, "video_config.max_duration")
        row += 1

        # Temporal Consistency Parameters
        components.label(frame, row, 0, "Temporal Consistency Weight",
                         tooltip="Weight for temporal consistency loss. Higher values enforce more consistency between frames.")
        components.entry(frame, row, 1, self.ui_state, "video_config.temporal_consistency_weight")
        row += 1

        components.label(frame, row, 0, "Use Temporal Attention",
                         tooltip="Enable temporal attention mechanisms for better frame-to-frame consistency")
        components.switch(frame, row, 1, self.ui_state, "video_config.use_temporal_attention")
        row += 1

        # Memory Management Parameters
        components.label(frame, row, 0, "Spatial Compression Ratio",
                         tooltip="Compression ratio for spatial dimensions to reduce memory usage")
        components.entry(frame, row, 1, self.ui_state, "video_config.spatial_compression_ratio")
        row += 1

        components.label(frame, row, 0, "Temporal Compression Ratio",
                         tooltip="Compression ratio for temporal dimension to reduce memory usage")
        components.entry(frame, row, 1, self.ui_state, "video_config.temporal_compression_ratio")
        row += 1

        components.label(frame, row, 0, "Video Batch Size Multiplier",
                         tooltip="Multiplier for batch size when processing video. Lower values reduce memory usage.")
        components.entry(frame, row, 1, self.ui_state, "video_config.video_batch_size_multiplier")
        row += 1

        # Video-specific Training Parameters
        components.label(frame, row, 0, "Frame Dropout Probability",
                         tooltip="Probability of dropping frames during training for regularization")
        components.entry(frame, row, 1, self.ui_state, "video_config.frame_dropout_probability")
        row += 1

        components.label(frame, row, 0, "Temporal Augmentation",
                         tooltip="Enable temporal augmentations like frame shuffling and temporal cropping")
        components.switch(frame, row, 1, self.ui_state, "video_config.temporal_augmentation")
        row += 1