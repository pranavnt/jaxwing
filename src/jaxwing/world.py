"""
Environment and scene representation module.
"""
from typing import Dict, Tuple, Any, Optional

import jax
import jax.numpy as jnp
from jax import random


class Terrain:
    """
    Represents the ground with height map.
    """
    def __init__(self, size: int = 256, height_scale: float = 50.0) -> None:
        """
        Initialize terrain with a procedural height map.

        Parameters
        ----------
        size : int, optional
            Size of the height map (size x size), reduced from 1024 to 256 for better performance
        height_scale : float, optional
            Maximum height of terrain, by default 50.0
        """
        self.size = size
        self.height_scale = height_scale
        self.map_scale = 2000.0  # World units per height map dimension

        # Generate procedural terrain
        self.height_map = generate_terrain(size, height_scale)

        # Precompute terrain colors for efficient lookup
        self.color_map = generate_terrain_colors(self.height_map, height_scale)


class Sky:
    """
    Handles atmosphere and lighting.
    """
    def __init__(self, sunset_mode: bool = False) -> None:
        """
        Initialize sky with basic properties.

        Parameters
        ----------
        sunset_mode : bool, optional
            Whether to use sunset colors, by default False
        """
        if sunset_mode:
            # Sunset configuration
            self.color = jnp.array([0.2, 0.1, 0.3])  # Deep blue-purple sky
            self.sun_direction = jnp.array([0.2, 0.3, -0.1])  # Low sun angle
            self.sun_direction = self.sun_direction / jnp.linalg.norm(self.sun_direction)
            self.sun_color = jnp.array([1.0, 0.6, 0.3])  # Orange-red sunlight
            self.ambient_light = 0.2  # Lower ambient light for sunset
        else:
            # Default daytime configuration
            self.color = jnp.array([0.5, 0.7, 1.0])
            self.sun_direction = jnp.array([0.5, 0.8, -0.2])
            self.sun_direction = self.sun_direction / jnp.linalg.norm(self.sun_direction)
            self.sun_color = jnp.array([1.0, 0.95, 0.9])  # Slightly warm sunlight
            self.ambient_light = 0.3  # Amount of ambient light (0-1)


@jax.jit
def perlin_noise(x: jnp.ndarray, y: jnp.ndarray, seed: int = 0) -> jnp.ndarray:
    """
    Simple Perlin-like noise implementation with JAX.

    Parameters
    ----------
    x : jnp.ndarray
        X coordinates
    y : jnp.ndarray
        Y coordinates
    seed : int, optional
        Random seed, by default 0

    Returns
    -------
    jnp.ndarray
        Noise values at given coordinates
    """
    # Integer grid coordinates
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)

    # Fractional part
    x -= x0
    y -= y0

    # Hash function (using integer wrapping)
    key = random.PRNGKey(seed)
    hash_table = random.uniform(key, (256,)) * 2.0 * jnp.pi

    # Function to get pseudo-random gradient vectors
    def gradient(h: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        h = h % 256
        angle = hash_table[h]
        return jnp.cos(angle), jnp.sin(angle)

    # Hash coordinates
    h00 = (x0 + y0 * 59) % 256
    h10 = (x0 + 1 + y0 * 59) % 256
    h01 = (x0 + (y0 + 1) * 59) % 256
    h11 = (x0 + 1 + (y0 + 1) * 59) % 256

    # Gradient vectors
    g00x, g00y = gradient(h00)
    g10x, g10y = gradient(h10)
    g01x, g01y = gradient(h01)
    g11x, g11y = gradient(h11)

    # Dot products
    d00 = g00x * x + g00y * y
    d10 = g10x * (x - 1) + g10y * y
    d01 = g01x * x + g01y * (y - 1)
    d11 = g11x * (x - 1) + g11y * (y - 1)

    # Smooth interpolation function
    def smooth(t: jnp.ndarray) -> jnp.ndarray:
        return t * t * t * (t * (t * 6 - 15) + 10)

    # Calculate interpolation weights
    sx = smooth(x)
    sy = smooth(y)

    # Interpolate dot products
    x0_interp = d00 + sx * (d10 - d00)
    x1_interp = d01 + sx * (d11 - d01)
    return x0_interp + sy * (x1_interp - x0_interp)


def generate_terrain(size: int, height_scale: float, seed: int = 42) -> jnp.ndarray:
    """
    Generate a procedural terrain height map using fractal noise.

    Parameters
    ----------
    size : int
        Size of the height map (size x size)
    height_scale : float
        Maximum height of terrain
    seed : int, optional
        Random seed, by default 42

    Returns
    -------
    jnp.ndarray
        2D array of height values
    """
    # Create coordinate grid
    x, y = jnp.meshgrid(jnp.linspace(0, 4, size), jnp.linspace(0, 4, size))

    # Generate fractal noise by summing octaves
    height_map = jnp.zeros((size, size))

    # Define octave parameters
    octaves = 6
    persistence = 0.5
    lacunarity = 2.0

    amplitude = 1.0
    frequency = 1.0

    # Sum up noise octaves
    for i in range(octaves):
        noise = perlin_noise(x * frequency, y * frequency, seed + i)
        height_map += noise * amplitude

        # Update parameters for next octave
        amplitude *= persistence
        frequency *= lacunarity

    # Normalize and scale height values
    height_map = (height_map - jnp.min(height_map)) / (jnp.max(height_map) - jnp.min(height_map))
    height_map = height_map * height_scale

    return height_map


def generate_terrain_colors(height_map: jnp.ndarray, height_scale: float) -> jnp.ndarray:
    """
    Generate terrain colors based on height with procedural texture details.

    Parameters
    ----------
    height_map : jnp.ndarray
        Terrain height map
    height_scale : float
        Maximum height of terrain

    Returns
    -------
    jnp.ndarray
        RGB color values for terrain
    """
    size = height_map.shape[0]
    colors = jnp.zeros((size, size, 3))

    # Normalize heights to 0-1 range for color mapping
    norm_heights = height_map / height_scale

    # Create coordinate grids for texture generation
    y_coords, x_coords = jnp.mgrid[0:size, 0:size] / size

    # Generate noise patterns for texture detail
    detail_noise1 = perlin_noise(x_coords * 8, y_coords * 8, seed=123)
    detail_noise2 = perlin_noise(x_coords * 16, y_coords * 16, seed=456)
    detail_noise3 = perlin_noise(x_coords * 32, y_coords * 32, seed=789)

    # Combine noise at different scales for more natural look
    detail_map = (detail_noise1 * 0.6 + detail_noise2 * 0.3 + detail_noise3 * 0.1)
    detail_map = (detail_map + 1.0) * 0.5  # Normalize to 0-1

    # Water level
    water_level = 0.1
    water_color = jnp.array([0.0, 0.1, 0.4])
    water_shallow = jnp.array([0.1, 0.3, 0.5])

    # Beach
    beach_level = 0.12
    beach_color = jnp.array([0.76, 0.7, 0.5])
    beach_detail = jnp.array([0.7, 0.65, 0.45])  # Darker sand for detail

    # Grass
    grass_level = 0.35
    grass_color = jnp.array([0.1, 0.5, 0.1])
    grass_detail = jnp.array([0.05, 0.4, 0.05])  # Darker grass for detail

    # Forest
    forest_level = 0.6
    forest_color = jnp.array([0.0, 0.4, 0.0])
    forest_detail = jnp.array([0.0, 0.3, 0.0])  # Darker forest for detail

    # Rock
    rock_level = 0.8
    rock_color = jnp.array([0.5, 0.5, 0.5])
    rock_detail = jnp.array([0.4, 0.4, 0.4])  # Darker rock for detail

    # Snow
    snow_color = jnp.array([1.0, 1.0, 1.0])
    snow_detail = jnp.array([0.9, 0.9, 0.95])  # Blue-ish snow shadow for detail

    # Function to blend colors
    def blend(t: jnp.ndarray, color1: jnp.ndarray, color2: jnp.ndarray) -> jnp.ndarray:
        # Make sure t has the right shape for broadcasting
        t_expanded = t[..., jnp.newaxis]  # Add channel dimension for broadcasting
        return color1 * (1 - t_expanded) + color2 * t_expanded

    # Function to apply texture detail using the detail noise
    def apply_detail(base_color: jnp.ndarray, detail_color: jnp.ndarray, detail_noise: jnp.ndarray, strength: float = 0.3) -> jnp.ndarray:
        # Scale the noise by strength
        scaled_noise = detail_noise * strength
        # Add channel dimension for broadcasting
        scaled_noise_expanded = scaled_noise[..., jnp.newaxis]
        return base_color * (1 - scaled_noise_expanded) + detail_color * scaled_noise_expanded

    # Generate colors based on height
    def color_at_height(h: jnp.ndarray, detail: jnp.ndarray) -> jnp.ndarray:
        water_mask = h < water_level
        beach_mask = (h >= water_level) & (h < beach_level)
        grass_mask = (h >= beach_level) & (h < grass_level)
        forest_mask = (h >= grass_level) & (h < forest_level)
        rock_mask = (h >= forest_level) & (h < rock_level)
        snow_mask = h >= rock_level

        # Calculate transition factors with UV-based detail
        water_t = jnp.clip((h - water_level * 0.5) / (water_level * 0.5), 0, 1)
        # Water noise needs special handling because it's a separate noise pattern
        water_noise = detail * 0.1  # Use the existing detail map, scaled down
        water_blend_factor = jnp.clip(water_t, 0, 1)  # Ensure it's in 0-1 range
        water_color_blend = blend(water_blend_factor, water_color, water_shallow)

        # Beach with UV detail
        beach_t = jnp.clip((h - beach_level) / (grass_level - beach_level) * 3, 0, 1)
        beach_color_blend = blend(beach_t, beach_color, grass_color)
        beach_color_blend = apply_detail(beach_color_blend, beach_detail, detail)

        # Grass with UV detail
        grass_t = jnp.clip((h - grass_level) / (forest_level - grass_level), 0, 1)
        grass_color_blend = blend(grass_t, grass_color, forest_color)
        grass_color_blend = apply_detail(grass_color_blend, grass_detail, detail)

        # Forest with UV detail
        forest_t = jnp.clip((h - forest_level) / (rock_level - forest_level), 0, 1)
        forest_color_blend = blend(forest_t, forest_color, rock_color)
        forest_color_blend = apply_detail(forest_color_blend, forest_detail, detail)

        # Rock with UV detail
        rock_t = jnp.clip((h - rock_level) / ((1.0 - rock_level) * 0.5), 0, 1)
        rock_color_blend = blend(rock_t, rock_color, snow_color)
        rock_color_blend = apply_detail(rock_color_blend, rock_detail, detail)

        # Snow with UV detail
        snow_with_detail = apply_detail(snow_color, snow_detail, detail, 0.2)

        # Combine all colors
        color = jnp.zeros((size, size, 3))
        color = jnp.where(water_mask[:, :, None], water_color_blend, color)
        color = jnp.where(beach_mask[:, :, None], beach_color_blend, color)
        color = jnp.where(grass_mask[:, :, None], grass_color_blend, color)
        color = jnp.where(forest_mask[:, :, None], forest_color_blend, color)
        color = jnp.where(rock_mask[:, :, None], rock_color_blend, color)
        color = jnp.where(snow_mask[:, :, None], snow_with_detail, color)

        return color

    # Create the color map with texture detail
    colors = color_at_height(norm_heights, detail_map)

    return colors


def get_height(position: jnp.ndarray, terrain: Optional[Terrain] = None) -> float:
    """
    Returns terrain height at position using bilinear interpolation.

    Parameters
    ----------
    position : jnp.ndarray
        Position [x, y]
    terrain : Optional[Terrain], optional
        Terrain object, by default None

    Returns
    -------
    float
        Height at the given position
    """
    # Default implementation for when terrain is not provided
    if terrain is None:
        return 0.0

    # Convert world position to height map coordinates
    size = terrain.size
    map_scale = terrain.map_scale

    # Calculate map coordinates (with wrapping for infinite terrain)
    map_x = ((position[0] % map_scale) / map_scale) * size
    map_y = ((position[1] % map_scale) / map_scale) * size

    # Get integer and fractional parts
    x0 = jnp.floor(map_x).astype(jnp.int32) % size
    y0 = jnp.floor(map_y).astype(jnp.int32) % size
    x1 = (x0 + 1) % size
    y1 = (y0 + 1) % size

    fx = map_x - x0
    fy = map_y - y0

    # Bilinear interpolation of height values
    h00 = terrain.height_map[y0, x0]
    h10 = terrain.height_map[y0, x1]
    h01 = terrain.height_map[y1, x0]
    h11 = terrain.height_map[y1, x1]

    h0 = h00 * (1 - fx) + h10 * fx
    h1 = h01 * (1 - fx) + h11 * fx

    return h0 * (1 - fy) + h1 * fy


def get_terrain_color(position: jnp.ndarray, terrain: Terrain) -> jnp.ndarray:
    """
    Returns terrain color at position using bilinear interpolation.

    Parameters
    ----------
    position : jnp.ndarray
        Position [x, y]
    terrain : Terrain
        Terrain object

    Returns
    -------
    jnp.ndarray
        RGB color at the given position
    """
    # Convert world position to color map coordinates
    size = terrain.size
    map_scale = terrain.map_scale

    # Calculate map coordinates (with wrapping for infinite terrain)
    map_x = ((position[0] % map_scale) / map_scale) * size
    map_y = ((position[1] % map_scale) / map_scale) * size

    # Get integer and fractional parts
    x0 = jnp.floor(map_x).astype(jnp.int32) % size
    y0 = jnp.floor(map_y).astype(jnp.int32) % size
    x1 = (x0 + 1) % size
    y1 = (y0 + 1) % size

    fx = map_x - x0
    fy = map_y - y0

    # Bilinear interpolation of color values
    c00 = terrain.color_map[y0, x0]
    c10 = terrain.color_map[y0, x1]
    c01 = terrain.color_map[y1, x0]
    c11 = terrain.color_map[y1, x1]

    c0 = c00 * (1 - fx) + c10 * fx
    c1 = c01 * (1 - fx) + c11 * fx

    return c0 * (1 - fy) + c1 * fy


def get_normal(position: jnp.ndarray, terrain: Optional[Terrain] = None) -> jnp.ndarray:
    """
    Calculates terrain surface normal at the given position.

    Parameters
    ----------
    position : jnp.ndarray
        Position [x, y]
    terrain : Optional[Terrain], optional
        Terrain object, by default None

    Returns
    -------
    jnp.ndarray
        Surface normal [nx, ny, nz]
    """
    # Default implementation for when terrain is not provided
    if terrain is None:
        return jnp.array([0.0, 0.0, 1.0])


    # Sample heights at nearby points
    delta = 1.0  # Small distance for gradient calculation

    h_center = get_height(position, terrain)
    h_right = get_height(position + jnp.array([delta, 0.0]), terrain)
    h_up = get_height(position + jnp.array([0.0, delta]), terrain)

    # Calculate partial derivatives
    dx = (h_right - h_center) / delta
    dy = (h_up - h_center) / delta

    # Normal is perpendicular to the tangent vectors
    normal = jnp.array([-dx, -dy, 1.0])
    return normal / jnp.linalg.norm(normal)  # Ensure normalized


class Building:
    """
    Simple landmark building for distance reference.
    """
    def __init__(self, position: jnp.ndarray, height: float, width: float, color: jnp.ndarray):
        """
        Initialize a building.

        Parameters
        ----------
        position : jnp.ndarray
            [x, y] position of the building center
        height : float
            Building height
        width : float
            Building width (square base)
        color : jnp.ndarray
            RGB color of the building
        """
        self.position = position
        self.height = height
        self.width = width
        self.color = color


class AircraftModel:
    """
    Simple 3D model of the aircraft for cockpit view.
    """
    def __init__(self):
        """
        Initialize the aircraft model with default parameters.
        """
        # Aircraft nose dimensions
        self.length = 5.0  # Length of nose cone
        self.width = 2.0   # Width at the base of the nose
        self.height = 1.0  # Height of the nose
        self.color = jnp.array([0.3, 0.3, 0.3])  # Dark gray color for the aircraft nose


def create_scene() -> Tuple[Terrain, Sky]:
    """
    Builds the complete scene.

    Returns
    -------
    Tuple[Terrain, Sky]
        The terrain and sky objects
    """
    terrain = Terrain()
    sky = Sky(sunset_mode=False)
    return terrain, sky
