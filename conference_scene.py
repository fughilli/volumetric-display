import logging
import os
import random
import traceback

import numpy as np

from artnet import Raster, Scene, load_scene

logger = logging.getLogger(__name__)


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
            ("Data Center Layers", "data_center_layers_scene_gpu.py"),
            ("Helix Redundancy", "helix_redundancy_scene_gpu.py"),
            ("Spherical Waves", "spherical_waves_scene_gpu.py"),
            # ("Rain Computation", "rain_computation_scene.py"),
            ("Neurons Firing", "neurons_scene_gpu.py"),
            ("Waves", "wave_scene.py"),
        ]

        # Timing
        self.scene_duration = 30.0  # seconds per scene
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
        try:
            return load_scene(full_path, properties=self.properties)
        except Exception as e:
            logger.error(f"Failed to create scene instance from {scene_path}: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise

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

        try:
            self.next_scene = self._create_scene_instance(next_scene_path)
            self.next_scene_name = next_scene_name
            self.transition_start_time = time

            print(f"Transitioning from '{self.current_scene_name}' to '{self.next_scene_name}'")
        except Exception as e:
            logger.error(f"Failed to start transition to '{next_scene_name}': {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            # Try to skip to next scene immediately
            self._force_next_scene(time)

    def _complete_transition(self):
        """Complete transition, make next scene current"""
        self.current_scene = self.next_scene
        self.current_scene_name = self.next_scene_name
        self.next_scene = None
        self.next_scene_name = ""
        self.scene_start_time = self.transition_start_time + self.transition_duration
        self.transition_start_time = None

        print(f"Transition complete. Now showing: {self.current_scene_name}")

    def _force_next_scene(self, time: float):
        """Force transition to next scene (used when current scene crashes)"""
        # Clear current scene
        self.current_scene = None
        self.current_scene_name = ""

        # Try to initialize a new scene
        try:
            self._initialize_scene(time)
        except Exception as e:
            logger.error(f"Failed to initialize replacement scene: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            # Scene will remain None, which is handled in render()

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
                try:
                    if self.current_scene:
                        self.current_scene.render(raster, time)
                except Exception as e:
                    logger.error(f"Scene '{self.current_scene_name}' crashed during render: {e}")
                    logger.error(f"Traceback:\n{traceback.format_exc()}")
                    # Force transition to next scene
                    self._force_next_scene(time)
                    # Try to render the new scene
                    if self.current_scene:
                        try:
                            self.current_scene.render(raster, time)
                        except Exception as e2:
                            logger.error(f"Replacement scene also crashed: {e2}")
                            raster.data.fill(0)
            else:
                # In transition - blend both scenes
                blend_factor = transition_time / self.transition_duration

                # Render current scene to raster1
                temp_raster1 = Raster(self.width, self.height, self.length)
                temp_raster1.data = self.raster1
                try:
                    if self.current_scene:
                        self.current_scene.render(temp_raster1, time)
                    else:
                        temp_raster1.data.fill(0)
                except Exception as e:
                    logger.error(
                        f"Scene '{self.current_scene_name}' crashed during transition render: {e}"
                    )
                    logger.error(f"Traceback:\n{traceback.format_exc()}")
                    temp_raster1.data.fill(0)
                    # Force transition to next scene
                    self._force_next_scene(time)

                # Render next scene to raster2
                temp_raster2 = Raster(self.width, self.height, self.length)
                temp_raster2.data = self.raster2
                try:
                    if self.next_scene:
                        self.next_scene.render(temp_raster2, time)
                    else:
                        temp_raster2.data.fill(0)
                except Exception as e:
                    logger.error(
                        f"Next scene '{self.next_scene_name}' crashed during transition render: {e}"
                    )
                    logger.error(f"Traceback:\n{traceback.format_exc()}")
                    temp_raster2.data.fill(0)
                    # Try to get a new next scene
                    try:
                        self._start_transition(time)
                        if self.next_scene:
                            try:
                                self.next_scene.render(temp_raster2, time)
                            except Exception:
                                temp_raster2.data.fill(0)
                    except Exception:
                        temp_raster2.data.fill(0)

                # Blend with smooth curve (ease in/out)
                # Use smoothstep for nicer transition
                smooth_blend = blend_factor * blend_factor * (3.0 - 2.0 * blend_factor)

                # Blend into output raster
                raster.data = (
                    (1.0 - smooth_blend) * temp_raster1.data + smooth_blend * temp_raster2.data
                ).astype(np.uint8)

        else:
            # Normal rendering - no transition
            try:
                if self.current_scene:
                    self.current_scene.render(raster, time)
                else:
                    raster.data.fill(0)
            except Exception as e:
                logger.error(f"Scene '{self.current_scene_name}' crashed during render: {e}")
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                # Force transition to next scene
                self._force_next_scene(time)
                # Try to render the new scene
                if self.current_scene:
                    try:
                        self.current_scene.render(raster, time)
                    except Exception as e2:
                        logger.error(f"Replacement scene also crashed: {e2}")
                        raster.data.fill(0)
