"""
JAXwing - A minimalist flight simulator built with JAX and custom ray tracing.
"""

from .physics import Aircraft, init_state, update_physics
from .raytracer import Ray, Scene, generate_rays, render_frame
from .world import Terrain, Sky, create_scene, get_height
from .main import Simulator, run_simulation

__all__ = [
    "Aircraft", "init_state", "update_physics", 
    "Ray", "Scene", "generate_rays", "render_frame",
    "Terrain", "Sky", "create_scene", "get_height",
    "Simulator", "run_simulation"
]
