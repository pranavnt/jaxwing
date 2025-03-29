"""
Custom ray tracing renderer built from scratch.
"""
from functools import partial
from typing import Dict, List, Tuple, Any, Optional
import os
import pickle
import hashlib

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from .world import get_height, get_normal, get_terrain_color


class Ray:
    """
    Representation of a ray with origin and direction.
    """
    def __init__(self, origin: jnp.ndarray, direction: jnp.ndarray) -> None:
        """
        Initialize a ray with origin and direction.

        Parameters
        ----------
        origin : jnp.ndarray
            The origin point of the ray [x, y, z]
        direction : jnp.ndarray
            The normalized direction vector of the ray [dx, dy, dz]
        """
        self.origin = origin
        self.direction = direction / jnp.linalg.norm(direction)  # Normalize


class Scene:
    """
    Container for renderable objects in the scene.
    """
    def __init__(self, terrain: Any, sky: Any) -> None:
        """
        Initialize a scene with terrain and sky.

        Parameters
        ----------
        terrain : Any
            The terrain object
        sky : Any
            The sky object
        """
        self.terrain = terrain
        self.sky = sky

        # Aircraft model for cockpit view
        from .world import AircraftModel
        self.aircraft = AircraftModel()


@partial(jax.jit, static_argnums=(2, 3, 4))
def generate_rays_jit(camera_pos: jnp.ndarray, camera_dir: jnp.ndarray,
                     width: int, height: int, fov: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create primary rays from camera for rendering, using JAX for acceleration.

    Parameters
    ----------
    camera_pos : jnp.ndarray
        Camera position [x, y, z]
    camera_dir : jnp.ndarray
        Camera direction [dx, dy, dz]
    width : int
        Image width in pixels
    height : int
        Image height in pixels
    fov : float
        Field of view in radians

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        Origins and directions for all rays
    """
    aspect_ratio = width / height

    # Create coordinate grid for all pixels
    y, x = jnp.mgrid[0:height, 0:width]
    x = x.reshape(-1)
    y = y.reshape(-1)

    # Map pixel coordinates to [-1, 1] range with correct aspect ratio
    pixel_x = (2 * (x + 0.5) / width - 1) * aspect_ratio * jnp.tan(fov / 2)
    pixel_y = (1 - 2 * (y + 0.5) / height) * jnp.tan(fov / 2)

    # Create normalized direction vectors
    # For Phase 3, we'll use proper camera orientation
    camera_dir = camera_dir / jnp.linalg.norm(camera_dir)

    # Define camera coordinate system
    camera_forward = camera_dir
    camera_right = jnp.array([camera_dir[1], -camera_dir[0], 0.0])

    # If camera is looking straight up or down, use a different right vector
    camera_right = jnp.where(
        jnp.abs(jnp.dot(camera_forward, jnp.array([0.0, 0.0, 1.0]))) > 0.99,
        jnp.array([1.0, 0.0, 0.0]),
        camera_right
    )

    camera_right = camera_right / jnp.linalg.norm(camera_right)
    camera_up = jnp.cross(camera_right, camera_forward)

    # Create ray directions using camera basis
    ray_dirs = (camera_forward.reshape(1, 3) +
                pixel_x.reshape(-1, 1) * camera_right.reshape(1, 3) +
                pixel_y.reshape(-1, 1) * camera_up.reshape(1, 3))

    # Normalize all directions
    ray_dirs = ray_dirs / jnp.linalg.norm(ray_dirs, axis=1, keepdims=True)

    # Create ray origins (all same as camera position)
    ray_origins = jnp.tile(camera_pos, (len(ray_dirs), 1))

    return ray_origins, ray_dirs


def generate_rays(camera_pos: jnp.ndarray, camera_dir: jnp.ndarray,
                 width: int, height: int, fov: float) -> List[Ray]:
    """
    Create primary rays from camera for rendering (legacy version).

    Parameters
    ----------
    camera_pos : jnp.ndarray
        Camera position [x, y, z]
    camera_dir : jnp.ndarray
        Camera direction [dx, dy, dz]
    width : int
        Image width in pixels
    height : int
        Image height in pixels
    fov : float
        Field of view in radians

    Returns
    -------
    List[Ray]
        List of rays for each pixel
    """
    ray_origins, ray_dirs = generate_rays_jit(camera_pos, camera_dir, width, height, fov)

    rays = []
    for i in range(len(ray_origins)):
        rays.append(Ray(ray_origins[i], ray_dirs[i]))

    return rays




# A simplified ray-marcher that's JAX-compatible
@partial(jax.jit, static_argnames=['terrain'])
def intersect_terrain_ray(ray_origin: jnp.ndarray, ray_dir: jnp.ndarray,
                         terrain: Any, max_dist: float = 10000.0) -> Tuple[bool, jnp.ndarray, float]:
    """
    Find intersection between a ray and the terrain using simplified ray marching.

    Parameters
    ----------
    ray_origin : jnp.ndarray
        Origin of the ray [x, y, z]
    ray_dir : jnp.ndarray
        Direction of the ray [dx, dy, dz]
    terrain : Any
        Terrain object
    max_dist : float, optional
        Maximum ray distance, by default 10000.0

    Returns
    -------
    Tuple[bool, jnp.ndarray, float]
        (hit, position, distance)
    """
    # Ultra-simplified terrain intersection for better performance
    # Use much coarser steps (16 instead of 64)
    # Ultra-simplified with just 4 steps for speed
    steps = 4
    step_size = max_dist / steps

    # Pre-compute just a few positions
    ts = jnp.array([250.0, 500.0, 750.0, 1000.0])  # Fixed steps
    positions = ray_origin[jnp.newaxis, :] + ray_dir[jnp.newaxis, :] * ts[:, jnp.newaxis]

    # Get heights at these positions
    heights = jnp.array([get_height(p[:2], terrain) for p in positions])

    # Check for intersections
    intersections = positions[:, 2] < heights

    # Find the first intersection using JAX-friendly code without conditionals
    # Convert boolean array to 0.0/1.0 for multiplication
    intersect_mask = intersections.astype(jnp.float32)

    # Check if any intersections exist
    has_intersection = jnp.max(intersect_mask) > 0

    # Get first intersection index (or 0 if none)
    # This uses a trick where we multiply the index by the mask
    # so only valid intersections contribute
    indices = jnp.arange(len(ts))
    valid_indices = indices * intersect_mask
    first_idx = jnp.argmax(valid_indices)

    # Select the appropriate distance
    t = jnp.where(has_intersection, ts[first_idx], max_dist)

    # Calculate position
    hit_pos = ray_origin + ray_dir * t

    # Return results
    return has_intersection, hit_pos, t


@jax.jit
def intersect_sky_ray(ray_origin: jnp.ndarray, ray_dir: jnp.ndarray) -> Tuple[bool, jnp.ndarray, float]:
    """
    Find intersection between a ray and the sky dome.

    Parameters
    ----------
    ray_origin : jnp.ndarray
        Origin of the ray [x, y, z]
    ray_dir : jnp.ndarray
        Direction of the ray [dx, dy, dz]

    Returns
    -------
    Tuple[bool, jnp.ndarray, float]
        (hit, position, distance)
    """
    # Simple sky intersection - just return sky color based on ray direction
    # For Phase 3, we use a simplified atmospheric model

    # Sky is always hit if terrain isn't
    sky_distance = 5000.0  # Far away
    sky_point = ray_origin + ray_dir * sky_distance

    return True, sky_point, sky_distance


@partial(jax.jit, static_argnames=['scene'])
def shade_point(pos: jnp.ndarray, normal: jnp.ndarray, view_dir: jnp.ndarray,
              distance: float, scene: Scene) -> jnp.ndarray:
    """
    Calculate lighting and shading for a point on the terrain.

    Parameters
    ----------
    pos : jnp.ndarray
        Intersection point [x, y, z]
    normal : jnp.ndarray
        Surface normal at intersection point
    view_dir : jnp.ndarray
        Direction from the point to the viewer
    distance : float
        Distance from the viewer to the point
    scene : Scene
        Scene containing terrain and sky information

    Returns
    -------
    jnp.ndarray
        RGB color for the shaded point
    """
    # Get base terrain color
    base_color = get_terrain_color(pos[:2], scene.terrain)

    # Sun direction and color from sky
    sun_dir = scene.sky.sun_direction
    sun_color = scene.sky.sun_color
    ambient_light = scene.sky.ambient_light

    # Calculate diffuse lighting (Lambert)
    diffuse_strength = jnp.maximum(0.0, jnp.dot(normal, sun_dir))

    # Calculate specular lighting (Blinn-Phong)
    half_vector = (sun_dir + view_dir) / jnp.linalg.norm(sun_dir + view_dir)
    specular_strength = jnp.power(jnp.maximum(0.0, jnp.dot(normal, half_vector)), 32.0)

    # Cast shadow ray to determine if point is in shadow
    shadow_factor = cast_shadow_ray(pos, sun_dir, scene)

    # Combine lighting components
    diffuse = diffuse_strength * sun_color * (1.0 - shadow_factor)  # Reduce diffuse in shadow
    specular = specular_strength * sun_color * 0.2 * (1.0 - shadow_factor)  # No specular in shadow
    ambient = ambient_light * scene.sky.color  # Ambient light is unaffected by shadows

    # Combine lighting with base color
    color = base_color * (ambient + diffuse) + specular

    # Height-dependent atmospheric fog with exponential distance falloff
    # Fog density: d = 1 - e^(-dist/fog_dist)
    # Height factor: h = clip((z-min_h)/(max_h-min_h), 0, 1)
    fog_base_distance = 2000.0
    fog_min_height = 50.0
    fog_max_height = 800.0

    height_factor = jnp.clip((pos[2] - fog_min_height) / (fog_max_height - fog_min_height), 0.0, 1.0)
    fog_distance = fog_base_distance * (1.0 + height_factor * 1.5)

    sun_dir = scene.sky.sun_direction
    up_vector = jnp.array([0.0, 0.0, 1.0])
    sun_altitude = jnp.dot(sun_dir, up_vector)
    sunset_factor = jnp.maximum(0.0, 1.0 - sun_altitude * 5.0)

    fog_density = 1.0 - jnp.exp(-distance / fog_distance)

    # Adjust fog color based on sun position (warmer at sunset)
    base_fog_color = scene.sky.color
    sunset_fog_color = jnp.array([0.9, 0.6, 0.5])  # Warm sunset fog
    fog_color = base_fog_color * (1.0 - sunset_factor) + sunset_fog_color * sunset_factor

    # Apply altitude-dependent fog color variation (blue-ish at higher altitudes)
    high_altitude_fog = jnp.array([0.7, 0.8, 1.0])  # Slight blue tint for high altitude
    fog_color = fog_color * (1.0 - height_factor * 0.5) + high_altitude_fog * height_factor * 0.5

    # Apply fog with height-adjusted density
    fog_factor = fog_density * (1.0 - height_factor * 0.7)  # Reduce fog at higher altitudes
    color = color * (1.0 - fog_factor) + fog_color * fog_factor

    # Distance-based desaturation for very far objects (aerial perspective)
    extreme_distance = 4000.0
    desaturation_factor = jnp.minimum(1.0, distance / extreme_distance)

    # Distance-based desaturation using luminance blending
    # Luminance (Y) = 0.299R + 0.587G + 0.114B (Rec. 601 standard)
    luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    luminance_color = jnp.array([luminance, luminance, luminance])

    distance_threshold = extreme_distance * 0.5
    blend_factor = jnp.maximum(0.0, (distance - distance_threshold) / distance_threshold)
    desaturation_amount = blend_factor * desaturation_factor * 0.5

    color = color * (1.0 - desaturation_amount) + luminance_color * desaturation_amount

    # Clamp values to [0, 1] range
    return jnp.clip(color, 0.0, 1.0)


@partial(jax.jit, static_argnames=['scene'])
def shade_sky(ray_dir: jnp.ndarray, scene: Scene) -> jnp.ndarray:
    """
    Shade a sky ray with improved atmospheric scattering.

    Parameters
    ----------
    ray_dir : jnp.ndarray
        Ray direction
    scene : Scene
        The scene containing sky information

    Returns
    -------
    jnp.ndarray
        RGB color for the sky
    """
    # Base sky colors
    zenith_color = scene.sky.color
    horizon_color = jnp.array([0.95, 0.5, 0.2])  # Orange-red sunset glow
    ground_haze = jnp.array([0.7, 0.7, 0.8])     # Bluish atmospheric haze

    # Reference vectors
    up_vector = jnp.array([0.0, 0.0, 1.0])
    sun_dir = scene.sky.sun_direction

    # Calculate sky gradients
    up_angle = jnp.abs(jnp.dot(ray_dir, up_vector))
    sun_angle = jnp.dot(ray_dir, sun_dir)

    # Improved Preetham sky model approximation
    # Higher exponents create sharper transition between zenith and horizon
    zenith_factor = jnp.power(up_angle, 0.5)
    horizon_factor = jnp.power(1.0 - up_angle, 3.0)

    # Sun-dependent coloring (sky changes based on sun position)
    sun_altitude = jnp.dot(sun_dir, up_vector)
    sunset_factor = jnp.maximum(0.0, 1.0 - sun_altitude * 5.0)  # Stronger during sunset

    # Blend the base sky gradient
    sky_color = zenith_color * zenith_factor + horizon_color * horizon_factor * sunset_factor

    # Add ground haze at low angles (more pronounced at sunset)
    haze_strength = jnp.maximum(0.0, 0.2 - up_angle) * 5.0 * sunset_factor
    sky_color = sky_color * (1.0 - haze_strength) + ground_haze * haze_strength

    # Improved sun with multiple halos and physically-based scattering
    # Multi-component sun with halos using exponential falloff:
    # Sun disk: e^((cos(θ)-1)*150) * 5.0
    # Medium halo: e^((cos(θ)-0.97)*20) * 0.8
    # Outer halo: e^((cos(θ)-0.8)*5) * 0.3 * sunset
    sun_disk = jnp.exp((sun_angle - 1.0) * 150.0) * 5.0
    sun_medium_halo = jnp.exp((sun_angle - 0.97) * 20.0) * 0.8
    sun_outer_halo = jnp.exp((sun_angle - 0.8) * 5.0) * 0.3 * sunset_factor
    atmospheric_scatter = jnp.maximum(0.0, sun_angle) * 0.1 * sunset_factor

    sun_effect = (
        scene.sky.sun_color * sun_disk +
        horizon_color * sun_medium_halo +
        horizon_color * sun_outer_halo +
        horizon_color * atmospheric_scatter
    )

    # Procedural clouds using harmonic product of sin waves:
    # C = sin(x·s) · sin(y·s + x·2) · sin(z·s·0.5 + y) · d
    # where s=15.0 (scale), d=0.08 (density)
    cloud_scale = 15.0
    cloud_density = 0.08
    cloud_noise = (
        jnp.sin(ray_dir[0] * cloud_scale) *
        jnp.sin(ray_dir[1] * cloud_scale + ray_dir[0] * 2.0) *
        jnp.sin(ray_dir[2] * cloud_scale * 0.5 + ray_dir[1])
    ) * cloud_density

    # Clouds masked by altitude: mask = max(0,cos(θ)) · (1-cos(θ))
    cloud_mask = jnp.maximum(0.0, up_angle) * (1.0 - up_angle)
    cloud_effect = jnp.array([1.0, 1.0, 1.0]) * cloud_noise * cloud_mask

    # Combine all effects
    sky_color = sky_color + sun_effect + cloud_effect

    return jnp.clip(sky_color, 0.0, 1.0)




@partial(jax.jit, static_argnames=['scene'])
def cast_shadow_ray(position: jnp.ndarray, light_dir: jnp.ndarray, scene: Scene) -> float:
    """
    Cast a shadow ray from a position towards a light source to determine
    if the point is in shadow.

    Parameters
    ----------
    position : jnp.ndarray
        The world position to cast the shadow ray from
    light_dir : jnp.ndarray
        Direction to the light source
    scene : Scene
        The scene containing terrain information

    Returns
    -------
    float
        Shadow factor (0.0 = fully lit, 1.0 = completely in shadow)
    """
    # Add a small offset to avoid self-intersection (shadow acne)
    epsilon = 0.1
    shadow_origin = position + light_dir * epsilon

    # Check for terrain intersection using the terrain ray function
    hit, _, shadow_distance = intersect_terrain_ray(shadow_origin, light_dir, scene.terrain)

    # Basic shadow implementation - completely in shadow or not
    shadow_factor = jnp.where(hit, 0.85, 0.0)  # 85% darkness if hit

    # Apply distance-based shadow softening (shadows get softer with distance)
    max_shadow_distance = 2000.0
    distance_factor = jnp.minimum(1.0, shadow_distance / max_shadow_distance)

    # Soften shadow based on distance
    shadow_factor = shadow_factor * (1.0 - distance_factor * 0.5)

    return shadow_factor




@partial(jax.jit, static_argnames=['scene', 'in_cockpit_view', 'aircraft_params'])
def trace_single_ray(ray_origin: jnp.ndarray, ray_dir: jnp.ndarray, scene: Scene,
                   in_cockpit_view: bool = False, aircraft_params: Optional[Dict] = None) -> jnp.ndarray:
    """
    Trace a single ray through the scene.

    Parameters
    ----------
    ray_origin : jnp.ndarray
        Origin of the ray
    ray_dir : jnp.ndarray
        Direction of the ray
    scene : Scene
        The scene to trace against
    in_cockpit_view : bool, optional
        Whether the camera is in cockpit view, by default False
    aircraft_params : Optional[Dict], optional
        Aircraft parameters for rendering nose in cockpit view, by default None

    Returns
    -------
    jnp.ndarray
        RGB color for the ray
    """
    # Test terrain intersection
    terrain_hit, terrain_pos, terrain_dist = intersect_terrain_ray(ray_origin, ray_dir, scene.terrain)

    # Get terrain normal and view direction
    normal = get_normal(terrain_pos[:2], scene.terrain)
    view_dir = -ray_dir  # Direction from point to viewer

    # Calculate terrain color if there's a hit
    terrain_color = shade_point(terrain_pos, normal, view_dir, terrain_dist, scene)

    # Building hit detection - hardcoded for JAX compatibility
    # Since JAX doesn't work well with dynamic objects, we'll just create a simple box directly
    
    # Define a building at (0,0)
    building_pos = jnp.array([0.0, 0.0])  # Center of building
    building_height = 500.0               # Tall building (500m)
    building_width = 100.0                # Width of building
    # Use a more moderate red color that won't appear too intense with lighting
    building_color = jnp.array([0.7, 0.2, 0.2])  # Darker reddish color
    
    # Calculate box bounds
    half_width = building_width / 2.0
    min_bound = jnp.array([building_pos[0] - half_width, building_pos[1] - half_width, 0.0])
    max_bound = jnp.array([building_pos[0] + half_width, building_pos[1] + half_width, building_height])
    
    # Ray-box intersection test (inlined for JAX compatibility)
    safe_rcp_x = jnp.where(jnp.abs(ray_dir[0]) < 1e-6, 1e9 * jnp.sign(ray_dir[0] + 1e-10), 1.0 / ray_dir[0])
    safe_rcp_y = jnp.where(jnp.abs(ray_dir[1]) < 1e-6, 1e9 * jnp.sign(ray_dir[1] + 1e-10), 1.0 / ray_dir[1])
    safe_rcp_z = jnp.where(jnp.abs(ray_dir[2]) < 1e-6, 1e9 * jnp.sign(ray_dir[2] + 1e-10), 1.0 / ray_dir[2])
    
    # Compute slab intersections
    t1 = (min_bound[0] - ray_origin[0]) * safe_rcp_x
    t2 = (max_bound[0] - ray_origin[0]) * safe_rcp_x
    t3 = (min_bound[1] - ray_origin[1]) * safe_rcp_y
    t4 = (max_bound[1] - ray_origin[1]) * safe_rcp_y
    t5 = (min_bound[2] - ray_origin[2]) * safe_rcp_z
    t6 = (max_bound[2] - ray_origin[2]) * safe_rcp_z
    
    # Ensure t1 <= t2, t3 <= t4, t5 <= t6
    tx_min = jnp.minimum(t1, t2)
    tx_max = jnp.maximum(t1, t2)
    ty_min = jnp.minimum(t3, t4)
    ty_max = jnp.maximum(t3, t4)
    tz_min = jnp.minimum(t5, t6)
    tz_max = jnp.maximum(t5, t6)
    
    # Find largest entry and smallest exit
    tmin = jnp.maximum(jnp.maximum(tx_min, ty_min), tz_min)
    tmax = jnp.minimum(jnp.minimum(tx_max, ty_max), tz_max)
    
    # Check if ray misses box
    b_hit = jnp.logical_and(tmax >= 0, tmin <= tmax)
    
    # Compute hit distance and position
    b_dist = jnp.where(b_hit, tmin, 10000.0)
    b_pos = ray_origin + ray_dir * b_dist
    
    # Initialize building hit information
    building_hit = b_hit
    building_pos = b_pos  # Always use the position from intersection function
    building_dist = b_dist  # Always use the distance from intersection function
    # Use the hardcoded color we defined above
    building_color = building_color

    # Initialize aircraft nose hit variables
    # Disable cockpit ray tracing for now - we'll handle this in the main.py file instead
    # This avoids a lot of conditional logic that's not JAX-compatible
    aircraft_hit = False
    aircraft_pos = ray_origin
    aircraft_dist = 10000.0
    aircraft_color = jnp.zeros(3)

    # Check what's closest - terrain, building, or aircraft nose
    terrain_closest = terrain_hit & ~(building_hit & (building_dist < terrain_dist)) & ~(aircraft_hit & (aircraft_dist < terrain_dist))
    building_closest = building_hit & (building_dist < terrain_dist) & ~(aircraft_hit & (aircraft_dist < building_dist))
    aircraft_closest = aircraft_hit & ((aircraft_dist < terrain_dist) | ~terrain_hit) & ((aircraft_dist < building_dist) | ~building_hit)

    # Calculate sky color
    sky_color = shade_sky(ray_dir, scene)

    # Add basic shading for building (using JAX-friendly operations)
    # Compute a simplified normal for the building using axis-aligned faces
    # Determine which face was hit based on position
    building_center = jnp.array([0.0, 0.0, building_height/2])
    hit_relative = building_pos - building_center
    
    # Look at which component has the largest magnitude
    abs_x = jnp.abs(hit_relative[0])
    abs_y = jnp.abs(hit_relative[1])
    abs_z = jnp.abs(hit_relative[2])
    
    # Default normal pointing outward from center
    hit_normal = hit_relative / jnp.maximum(jnp.linalg.norm(hit_relative), 1e-6)
    
    # Create simple axis-aligned normals for each face
    x_normal = jnp.array([jnp.sign(hit_relative[0]), 0.0, 0.0])
    y_normal = jnp.array([0.0, jnp.sign(hit_relative[1]), 0.0])
    z_normal = jnp.array([0.0, 0.0, jnp.sign(hit_relative[2])])
    
    # Select the normal based on which axis has the largest component
    hit_normal = jnp.where(abs_x > abs_y, 
                         jnp.where(abs_x > abs_z, x_normal, z_normal),
                         jnp.where(abs_y > abs_z, y_normal, z_normal))
    
    # Apply simple lighting (ambient + diffuse)
    light_dir = scene.sky.sun_direction
    diffuse = jnp.maximum(0.2, jnp.dot(hit_normal, light_dir))
    ambient = 0.3
    
    # Apply lighting to building
    lit_building_color = building_color * (ambient + diffuse * 0.7)
    
    # Apply distance fog
    fog_factor = 1.0 - jnp.exp(-building_dist / 2000.0)
    fog_color = scene.sky.color
    shaded_building_color = lit_building_color * (1.0 - fog_factor) + fog_color * fog_factor
    
    # Only apply shading if the building was hit (using conditional assignment)
    building_color = jnp.where(building_hit, shaded_building_color, building_color)
    
    # Select final color based on what was hit and what's closest
    color = jnp.where(terrain_closest, terrain_color,
                     jnp.where(building_closest, building_color,
                              jnp.where(aircraft_closest, aircraft_color, sky_color)))

    return color


# Vectorized trace function that processes all rays in parallel with JAX
@partial(jax.jit, static_argnames=['scene', 'in_cockpit_view', 'aircraft_params'])
def trace_rays_batched(ray_origins: jnp.ndarray, ray_dirs: jnp.ndarray,
                     scene: Scene, in_cockpit_view: bool = False,
                     aircraft_params: Optional[Dict] = None) -> jnp.ndarray:
    """
    JAX-optimized ray tracing using vmap for parallelization.

    Parameters
    ----------
    ray_origins : jnp.ndarray
        Origins of all rays
    ray_dirs : jnp.ndarray
        Directions of all rays
    scene : Scene
        The scene to trace against
    in_cockpit_view : bool, optional
        Whether the camera is in cockpit view, by default False
    aircraft_params : Optional[Dict], optional
        Aircraft parameters for cockpit view, by default None

    Returns
    -------
    jnp.ndarray
        Colors for all rays
    """
    n_rays = ray_origins.shape[0]

    # Define a function that traces a single ray that we can vectorize
    def trace_ray(origin, direction):
        return trace_single_ray(origin, direction, scene, in_cockpit_view, aircraft_params)

    # Use JAX's vmap to vectorize the function across all rays
    # This is much faster than looping through rays one by one
    ray_colors = jax.vmap(trace_ray)(ray_origins, ray_dirs)

    # If we're in cockpit view, we need to draw the aircraft nose directly
    # This is handled in the main.py file now for better performance

    return ray_colors


def trace_rays(rays: List[Ray], scene: Scene, in_cockpit_view: bool = False,
             aircraft_params: Optional[Dict] = None) -> jnp.ndarray:
    """
    Main ray tracing function (upgraded for Phase 3).

    Parameters
    ----------
    rays : List[Ray]
        List of rays to trace
    scene : Scene
        The scene to trace against
    in_cockpit_view : bool, optional
        Whether the camera is in cockpit view, by default False
    aircraft_params : Optional[Dict], optional
        Aircraft parameters for cockpit view, by default None

    Returns
    -------
    jnp.ndarray
        Color values for each ray
    """
    # Convert rays list to arrays for efficient processing
    ray_origins = jnp.array([ray.origin for ray in rays])
    ray_dirs = jnp.array([ray.direction for ray in rays])

    # Use batched tracing for better memory efficiency
    colors = trace_rays_batched(ray_origins, ray_dirs, scene, in_cockpit_view=in_cockpit_view,
                               aircraft_params=aircraft_params)

    return colors


@partial(jax.jit, static_argnames=['scene', 'width', 'height', 'fov', 'in_cockpit_view', 'aircraft_state'])
def render_frame(camera_pos: jnp.ndarray, camera_dir: jnp.ndarray,
                scene: Scene, width: int, height: int, fov: float = jnp.pi/3,
                in_cockpit_view: bool = False, aircraft_state: Optional[Dict] = None) -> np.ndarray:
    """
    Produce the final image by tracing rays through the scene.
    Now fully JIT-compiled and optimized for Phase 4.

    Parameters
    ----------
    camera_pos : jnp.ndarray
        Camera position [x, y, z]
    camera_dir : jnp.ndarray
        Camera direction [dx, dy, dz]
    scene : Scene
        The scene to render
    width : int
        Image width in pixels
    height : int
        Image height in pixels
    fov : float, optional
        Field of view in radians, by default pi/3
    in_cockpit_view : bool, optional
        Whether the camera is in cockpit view, by default False
    aircraft_state : Optional[Dict], optional
        Aircraft state for cockpit view, by default None

    Returns
    -------
    np.ndarray
        The rendered image as an RGB array with shape (height, width, 3)
    """

    ray_origins, ray_dirs = generate_rays_jit(camera_pos, camera_dir, width, height, fov)

    aircraft_params = None
    if in_cockpit_view and aircraft_state is not None:
        aircraft_params = {
            'position': aircraft_state.get('position', camera_pos),
            'forward': aircraft_state.get('forward', camera_dir),
            'up': aircraft_state.get('up', jnp.array([0.0, 0.0, 1.0]))
        }

    colors = trace_rays_batched(ray_origins, ray_dirs, scene, in_cockpit_view=in_cockpit_view,
                              aircraft_params=aircraft_params)

    image_reshaped = colors.reshape(height, width, 3)
    return (image_reshaped * 255).astype(jnp.uint8)