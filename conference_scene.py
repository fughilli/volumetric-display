import os
import random

import numpy as np

from artnet import Raster, Scene, load_scene


class ConferenceScene(Scene):
    """
    Master scene that smoothly transitions between multiple conference-themed scenes.
    Randomly selects scenes and blends between them.
    """

    def __init__(self, **kwargs):
        properties = kwargs.get("properties")
        if not properties:
            raise ValueError("ConferenceScene requires a 'properties' object.")

        self.properties = properties
        self.width = properties.width
        self.height = properties.height
        self.length = properties.length

        # Scene configuration - paths to scene files
        self.scene_dir = os.path.dirname(os.path.abspath(__file__))
        self.scene_configs = [
            # ("Data Center Layers", "data_center_layers_scene.py"),
            # ("Helix Redundancy", "helix_redundancy_scene.py"),
            # ("Spherical Waves", "spherical_waves_scene.py"),
            ("Rain Computation", "rain_computation_scene.py"),
            ("Neurons Firing (GPU)", "neurons_scene_gpu.py"),  # GPU-accelerated version
            # ("Neurons Firing (CPU)", "neurons_scene.py"),  # Original CPU version
            # ("Waves", "wave_scene.py"),
        ]

        # Timing
        self.scene_duration = 20.0  # seconds per scene
        self.transition_duration = 3.0  # seconds for blend transition

        # State
        self.current_scene = None
        self.next_scene = None
        self.current_scene_name = ""
        self.next_scene_name = ""
        self.scene_start_time = 0.0
        self.transition_start_time = None
        self.last_scene_idx = -1

        # Temporary rasters for blending
        self.raster1 = None
        self.raster2 = None

        # Initialize first scene
        self._initialize_scene(0.0)

        print("ConferenceScene initialized")
        print(f"Available scenes: {[name for name, _ in self.scene_configs]}")

    def _create_scene_instance(self, scene_path):
        """Create a new instance of a scene by loading it dynamically"""
        full_path = os.path.join(self.scene_dir, scene_path)
        return load_scene(full_path, properties=self.properties)

    def _select_next_scene(self):
        """Select a random scene different from current"""
        available_indices = [i for i in range(len(self.scene_configs)) if i != self.last_scene_idx]
        next_idx = random.choice(available_indices)
        self.last_scene_idx = next_idx
        return next_idx

    def _initialize_scene(self, time: float):
        """Initialize the first scene"""
        idx = self._select_next_scene()
        scene_name, scene_path = self.scene_configs[idx]

        self.current_scene = self._create_scene_instance(scene_path)
        self.current_scene_name = scene_name
        self.scene_start_time = time
        self.transition_start_time = None

        print(f"Starting scene: {scene_name}")

    def _start_transition(self, time: float):
        """Begin transition to next scene"""
        idx = self._select_next_scene()
        next_scene_name, next_scene_path = self.scene_configs[idx]

        self.next_scene = self._create_scene_instance(next_scene_path)
        self.next_scene_name = next_scene_name
        self.transition_start_time = time

        print(f"Transitioning from '{self.current_scene_name}' to '{self.next_scene_name}'")

    def _complete_transition(self):
        """Complete transition, make next scene current"""
        self.current_scene = self.next_scene
        self.current_scene_name = self.next_scene_name
        self.next_scene = None
        self.next_scene_name = ""
        self.scene_start_time = self.transition_start_time + self.transition_duration
        self.transition_start_time = None

        print(f"Transition complete. Now showing: {self.current_scene_name}")

    def render(self, raster: Raster, time: float):
        """Render with scene transitions"""
        # Initialize temporary rasters if needed
        if self.raster1 is None:
            self.raster1 = np.zeros_like(raster.data)
            self.raster2 = np.zeros_like(raster.data)

        # Calculate time in current scene
        scene_time = time - self.scene_start_time

        # Check if we need to start transition
        if (
            self.transition_start_time is None
            and scene_time >= self.scene_duration - self.transition_duration
        ):
            self._start_transition(time)

        # Check if we're in transition
        if self.transition_start_time is not None:
            transition_time = time - self.transition_start_time

            if transition_time >= self.transition_duration:
                # Transition complete
                self._complete_transition()
                # Render new current scene normally
                self.current_scene.render(raster, time)
            else:
                # In transition - blend both scenes
                blend_factor = transition_time / self.transition_duration

                # Render current scene to raster1
                temp_raster1 = Raster(self.width, self.height, self.length)
                temp_raster1.data = self.raster1
                self.current_scene.render(temp_raster1, time)

                # Render next scene to raster2
                temp_raster2 = Raster(self.width, self.height, self.length)
                temp_raster2.data = self.raster2
                self.next_scene.render(temp_raster2, time)

                # Blend with smooth curve (ease in/out)
                # Use smoothstep for nicer transition
                smooth_blend = blend_factor * blend_factor * (3.0 - 2.0 * blend_factor)

                # Blend into output raster
                raster.data = (
                    (1.0 - smooth_blend) * temp_raster1.data + smooth_blend * temp_raster2.data
                ).astype(np.uint8)

        else:
            # Normal rendering - no transition
            self.current_scene.render(raster, time)
