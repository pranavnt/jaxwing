"""
Aircraft physics module with JAX-accelerated flight dynamics.
"""
from functools import partial
from typing import Dict, Tuple, Any, Optional

import jax
import jax.numpy as jnp
from jax import lax

from .world import get_height


class Aircraft:
    """
    Represents an aircraft with state and properties.
    """
    def __init__(self, state: jnp.ndarray, properties: Dict[str, Any]) -> None:
        """
        Initialize an aircraft with state and properties.

        Parameters
        ----------
        state : jnp.ndarray
            The initial state vector of the aircraft
        properties : Dict[str, Any]
            Aircraft properties such as mass, wing area, etc.
        """
        self.state = state
        self.properties = properties

        # Add status flags for Phase 3
        self.is_stalled = False
        self.is_crashed = False
        self.status_message = ""

        # Add telemetry data for Phase 3
        self.telemetry = {
            "altitude": 0.0,
            "ground_speed": 0.0,
            "vertical_speed": 0.0,
            "angle_of_attack": 0.0,
            "bank_angle": 0.0,
            "g_forces": 1.0,
            "stall_warning": False,
            "terrain_height": 0.0,
            "terrain_clearance": 0.0
        }


def init_state() -> jnp.ndarray:
    """
    Initialize the aircraft state.

    Returns
    -------
    jnp.ndarray
        The initial state vector containing:
        [x, y, z, velocity_x, velocity_y, velocity_z, roll, pitch, yaw,
         angular_velocity_x, angular_velocity_y, angular_velocity_z]
    """
    # Initialize position, velocity, orientation, and angular velocity
    # For Phase 3, we add angular velocity components
    return jnp.array([
        0.0, -500.0, 100.0,  # Position [x, y, z]
        0.0, 30.0, 0.0,      # Velocity [vx, vy, vz]
        0.0, 0.0, 0.0,       # Orientation [roll, pitch, yaw]
        0.0, 0.0, 0.0        # Angular velocity [wx, wy, wz]
    ])


@jax.jit
def rotation_matrix_from_euler(roll: float, pitch: float, yaw: float) -> jnp.ndarray:
    """
    Create a rotation matrix from Euler angles.

    Parameters
    ----------
    roll : float
        Roll angle in radians
    pitch : float
        Pitch angle in radians
    yaw : float
        Yaw angle in radians

    Returns
    -------
    jnp.ndarray
        3x3 rotation matrix
    """
    # Compute trigonometric values
    cr, sr = jnp.cos(roll), jnp.sin(roll)
    cp, sp = jnp.cos(pitch), jnp.sin(pitch)
    cy, sy = jnp.cos(yaw), jnp.sin(yaw)

    # Construct rotation matrix - ZYX convention (yaw, pitch, roll)
    # This transforms from body frame to world frame
    rot = jnp.zeros((3, 3))

    # First row
    rot = rot.at[0, 0].set(cy * cp)
    rot = rot.at[0, 1].set(cy * sp * sr - sy * cr)
    rot = rot.at[0, 2].set(cy * sp * cr + sy * sr)

    # Second row
    rot = rot.at[1, 0].set(sy * cp)
    rot = rot.at[1, 1].set(sy * sp * sr + cy * cr)
    rot = rot.at[1, 2].set(sy * sp * cr - cy * sr)

    # Third row
    rot = rot.at[2, 0].set(-sp)
    rot = rot.at[2, 1].set(cp * sr)
    rot = rot.at[2, 2].set(cp * cr)

    return rot


@jax.jit
def get_velocity_in_body_frame(velocity: jnp.ndarray, orientation: jnp.ndarray) -> jnp.ndarray:
    """
    Transform velocity from world to body frame.

    Parameters
    ----------
    velocity : jnp.ndarray
        Velocity vector in world frame [vx, vy, vz]
    orientation : jnp.ndarray
        Orientation angles [roll, pitch, yaw]

    Returns
    -------
    jnp.ndarray
        Velocity in body frame [vx_body, vy_body, vz_body]
    """
    # Get rotation matrix from world to body (transpose of world from body)
    roll, pitch, yaw = orientation
    R = rotation_matrix_from_euler(roll, pitch, yaw)
    R_world_to_body = jnp.transpose(R)

    # Transform velocity
    return jnp.matmul(R_world_to_body, velocity)


@jax.jit
def calculate_aerodynamic_angles(velocity_body: jnp.ndarray) -> Tuple[float, float]:
    """
    Calculate angle of attack and sideslip angle.

    Parameters
    ----------
    velocity_body : jnp.ndarray
        Velocity in body frame [vx_body, vy_body, vz_body]

    Returns
    -------
    Tuple[float, float]
        (angle_of_attack, sideslip_angle) in radians
    """
    # Extract components
    vx, vy, vz = velocity_body

    # Calculate airspeed
    airspeed = jnp.linalg.norm(velocity_body)

    # Handle zero or very small airspeed
    safe_airspeed = jnp.maximum(airspeed, 0.01)

    # Calculate angles
    alpha = jnp.arctan2(vz, vx)  # Angle of attack
    beta = jnp.arcsin(vy / safe_airspeed)  # Sideslip angle

    return alpha, beta


def generate_wind(position: jnp.ndarray, time_sec: float = 0.0) -> jnp.ndarray:
    """
    Generate a constant wind vector.

    Parameters
    ----------
    position : jnp.ndarray
        Position in world coordinates [x, y, z]
    time_sec : float, optional
        Current simulation time in seconds, by default 0.0

    Returns
    -------
    jnp.ndarray
        Wind vector in world coordinates [wx, wy, wz]
    """
    # Constant wind direction and speed
    wind_direction = jnp.array([0.7, 0.3, 0.0])
    wind_direction = wind_direction / jnp.linalg.norm(wind_direction)
    wind_speed = 5.0  # m/s

    # Simple constant wind vector
    wind_vector = wind_direction * wind_speed

    return wind_vector


@jax.jit
def calculate_forces(state: jnp.ndarray, controls: jnp.ndarray, properties: Dict[str, Any],
                    time_sec: float = 0.0) -> jnp.ndarray:
    """
    Calculate forces acting on the aircraft with wind effects.

    Parameters
    ----------
    state : jnp.ndarray
        Current aircraft state
    controls : jnp.ndarray
        Control inputs [throttle, elevator, aileron, rudder]
    properties : Dict[str, Any]
        Aircraft properties
    time_sec : float, optional
        Current simulation time in seconds, by default 0.0

    Returns
    -------
    jnp.ndarray
        Forces and moments in body frame [Fx, Fy, Fz, Mx, My, Mz]
    """
    # Extract relevant state components
    position = state[:3]
    velocity_world = state[3:6]
    orientation = state[6:9]
    angular_velocity = state[9:]

    # Generate wind for this position and time
    wind_vector = generate_wind(position, time_sec)

    throttle_raw, elevator_raw, aileron_raw, rudder_raw = controls

    throttle = throttle_raw
    elevator = jnp.sign(elevator_raw) * (elevator_raw**2)
    aileron = jnp.sign(aileron_raw) * (aileron_raw**2)
    rudder = rudder_raw

    # Calculate airspeed by subtracting wind from ground speed
    true_airspeed_world = velocity_world - wind_vector

    # Convert velocity to body frame
    velocity_body = get_velocity_in_body_frame(true_airspeed_world, orientation)

    # Calculate airspeed, angle of attack, and sideslip
    airspeed = jnp.linalg.norm(velocity_body)
    alpha, beta = calculate_aerodynamic_angles(velocity_body)

    # Ensure safe airspeed value for calculations
    safe_airspeed = jnp.maximum(airspeed, 0.01)

    # Enhanced atmospheric model with temperature gradient
    air_density = 1.225  # kg/m^3 at sea level
    altitude = position[2]

    # International Standard Atmosphere model: T = T₀ - L·h
    # where T₀ = 288.15K, L = 0.0065 K/m (lapse rate)
    temperature_sea_level = 288.15
    lapse_rate = 0.0065
    temperature = temperature_sea_level - lapse_rate * altitude

    # Density ratio ρ/ρ₀ = (T/T₀)^(g/RL-1) where exponent ≈ 5.2561
    pressure_ratio = jnp.power(temperature / temperature_sea_level, 5.2561)
    air_density = air_density * pressure_ratio

    # Calculate dynamic pressure
    q = 0.5 * air_density * safe_airspeed**2

    # Extract aircraft properties
    wing_area = properties["wing_area"]
    wingspan = properties["wingspan"]
    mass = properties["mass"]
    engine_power = properties["engine_power"]

    # Lift coefficient with stall modeling
    # Pre-stall: CL = CL₀ + CLα·α
    # Post-stall: CL = CL₀ + sin(2α)·0.5
    # Combined with sigmoid blending: CL = CL_linear·(1-σ) + CL_stalled·σ 
    cl_0 = 0.3
    cl_alpha = 5.0
    stall_alpha = 0.3  # ~17 degrees
    
    cl_linear = cl_0 + cl_alpha * alpha
    cl_stalled = cl_0 + jnp.sin(2 * alpha) * 0.5
    
    cl_blend_factor = jax.nn.sigmoid((alpha - stall_alpha) * 20)
    cl = cl_linear * (1 - cl_blend_factor) + cl_stalled * cl_blend_factor

    # Drag coefficient using parabolic drag polar: CD = CD₀ + CL²/(πeAR)
    # where e is Oswald efficiency factor, AR is aspect ratio
    cd_0 = properties["drag_coefficient"]
    aspect_ratio = wingspan**2 / wing_area
    oswald_efficiency = 0.8
    induced_drag_factor = 1 / (jnp.pi * aspect_ratio * oswald_efficiency)
    cd = cd_0 + cl**2 * induced_drag_factor

    # Side force coefficient
    cy = -beta * 0.5  # Simple linear model for sideslip effect

    # Control surface effects on force coefficients
    cl_elevator = elevator * 0.4
    cd_elevator = jnp.abs(elevator) * 0.05
    cy_rudder = rudder * 0.4

    # Apply control effects
    cl += cl_elevator
    cd += cd_elevator
    cy += cy_rudder

    # Calculate forces in body frame
    # X-axis: negative drag, along body x-axis
    # Y-axis: side force, along body y-axis
    # Z-axis: negative lift, along body z-axis
    F_drag = -cd * q * wing_area
    F_side = cy * q * wing_area
    F_lift = -cl * q * wing_area

    # Thrust force (along body x-axis)
    F_thrust = throttle * engine_power

    # Total forces in body frame
    F_body = jnp.array([
        F_drag + F_thrust,  # X-axis force
        F_side,            # Y-axis force
        F_lift             # Z-axis force
    ])

    # Moment coefficients from control surfaces
    cm_pitch = -elevator * 1.5  # Elevator effect on pitch
    cl_roll = aileron * 0.2     # Aileron effect on roll
    cn_yaw = rudder * 0.3       # Rudder effect on yaw

    # Stability derivatives (simplified)
    cm_alpha = -0.5  # Pitch stability (negative means stable)
    cl_beta = -0.1   # Roll due to sideslip
    cn_beta = 0.1    # Yaw stability

    # Apply stability effects
    cm_pitch += cm_alpha * alpha
    cl_roll += cl_beta * beta
    cn_yaw += cn_beta * beta

    # Damping coefficients (resistance to angular velocity)
    cm_q = -20.0  # Pitch damping
    cl_p = -10.0  # Roll damping
    cn_r = -15.0  # Yaw damping

    # Apply damping based on angular velocity
    p, q, r = angular_velocity
    cm_pitch += cm_q * q * wingspan / (2 * safe_airspeed)
    cl_roll += cl_p * p * wingspan / (2 * safe_airspeed)
    cn_yaw += cn_r * r * wingspan / (2 * safe_airspeed)

    # Calculate moments
    mean_chord = wing_area / wingspan
    M_pitch = cm_pitch * q * wing_area * mean_chord
    M_roll = cl_roll * q * wing_area * wingspan
    M_yaw = cn_yaw * q * wing_area * wingspan

    # Total moments in body frame
    M_body = jnp.array([M_roll, M_pitch, M_yaw])

    # Combine forces and moments
    forces_moments = jnp.concatenate([F_body, M_body])

    return forces_moments


@jax.jit
def apply_controls(controls: jnp.ndarray) -> jnp.ndarray:
    """
    Translate user inputs to control surface effects, with limits.

    Parameters
    ----------
    controls : jnp.ndarray
        Control inputs [throttle, elevator, aileron, rudder]

    Returns
    -------
    jnp.ndarray
        Processed control values
    """
    # Apply limits to controls
    throttle, elevator, aileron, rudder = controls

    # Ensure controls are within limits
    throttle = jnp.clip(throttle, 0.0, 1.0)
    elevator = jnp.clip(elevator, -1.0, 1.0)
    aileron = jnp.clip(aileron, -1.0, 1.0)
    rudder = jnp.clip(rudder, -1.0, 1.0)

    elevator = elevator * 1.8
    aileron = aileron * 1.8
    rudder = rudder * 1.5

    def apply_expo(value, expo=0.2):
        return jnp.sign(value) * (jnp.abs(value) ** (1.0 - expo))

    elevator = apply_expo(elevator)
    aileron = apply_expo(aileron)

    elevator = jnp.clip(elevator, -1.0, 1.0)
    aileron = jnp.clip(aileron, -1.0, 1.0)
    rudder = jnp.clip(rudder, -1.0, 1.0)
    rudder = apply_expo(rudder)

    return jnp.array([throttle, elevator, aileron, rudder])


@partial(jax.jit, static_argnums=(3,4))
def update_physics(state: jnp.ndarray, controls: jnp.ndarray, dt: float,
                 properties: Dict[str, Any], terrain=None, time_sec: float = 0.0) -> jnp.ndarray:
    """
    Update the aircraft state based on forces and controls with advanced JAX optimization.

    Parameters
    ----------
    state : jnp.ndarray
        Current aircraft state
    controls : jnp.ndarray
        Control inputs [throttle, elevator, aileron, rudder]
    dt : float
        Time step in seconds
    properties : Dict[str, Any]
        Aircraft properties
    terrain : Any, optional
        Terrain object for ground collision detection
    time_sec : float, optional
        Current simulation time in seconds, used for wind effects

    Returns
    -------
    jnp.ndarray
        Updated aircraft state
    """
    # Extract state components
    position = state[:3]
    velocity_world = state[3:6]
    orientation = state[6:9]
    angular_velocity = state[9:12]

    # Calculate forces and moments in body frame with wind effects
    forces_moments = calculate_forces(state, controls, properties, time_sec)
    forces_body = forces_moments[:3]
    moments_body = forces_moments[3:]

    # Convert forces from body to world frame
    roll, pitch, yaw = orientation
    rotation = rotation_matrix_from_euler(roll, pitch, yaw)
    forces_world = jnp.matmul(rotation, forces_body)

    # Calculate accelerations (F = ma)
    mass = properties["mass"]
    linear_accel = forces_world / mass

    # Add gravity in world frame
    gravity = jnp.array([0.0, 0.0, -9.81])
    linear_accel = linear_accel + gravity

    # Update velocity and position
    new_velocity = velocity_world + linear_accel * dt
    new_position = position + velocity_world * dt + 0.5 * linear_accel * dt**2

    # Calculate angular accelerations (M = I*alpha)
    # Simplified inertia model (diagonal inertia tensor)
    Ixx = properties.get("inertia_xx", mass * 5.0)  # Substitute default if not provided
    Iyy = properties.get("inertia_yy", mass * 8.0)
    Izz = properties.get("inertia_zz", mass * 12.0)

    inertia_inv = jnp.array([1.0/Ixx, 1.0/Iyy, 1.0/Izz])
    angular_accel = moments_body * inertia_inv

    # Update angular velocity and orientation
    new_angular_velocity = angular_velocity + angular_accel * dt

    # Update orientation using angular velocity
    # For small timesteps, approximation works reasonably well
    roll_rate, pitch_rate, yaw_rate = new_angular_velocity

    # Convert body angular rates [p,q,r] to Euler rates [ϕ̇,θ̇,ψ̇] using:
    # ϕ̇ = p + sin(ϕ)tan(θ)q + cos(ϕ)tan(θ)r
    # θ̇ = cos(ϕ)q - sin(ϕ)r
    # ψ̇ = sin(ϕ)/cos(θ)q + cos(ϕ)/cos(θ)r
    sin_roll, cos_roll = jnp.sin(roll), jnp.cos(roll)
    sin_pitch, cos_pitch = jnp.sin(pitch), jnp.cos(pitch)

    roll_dot = roll_rate + sin_roll * jnp.tan(pitch) * pitch_rate + cos_roll * jnp.tan(pitch) * yaw_rate
    pitch_dot = cos_roll * pitch_rate - sin_roll * yaw_rate
    yaw_dot = (sin_roll / cos_pitch) * pitch_rate + (cos_roll / cos_pitch) * yaw_rate

    # Integrate orientation
    new_roll = roll + roll_dot * dt
    new_pitch = pitch + pitch_dot * dt
    new_yaw = yaw + yaw_dot * dt

    # Limit pitch to avoid gimbal lock
    new_pitch = jnp.clip(new_pitch, -jnp.pi/2 + 0.01, jnp.pi/2 - 0.01)

    # Wrap angles to stay within reasonable range
    new_roll = (new_roll + jnp.pi) % (2 * jnp.pi) - jnp.pi
    new_yaw = (new_yaw + jnp.pi) % (2 * jnp.pi) - jnp.pi

    new_orientation = jnp.array([new_roll, new_pitch, new_yaw])

    # Handle ground collision
    ground_height = 0.0
    if terrain is not None:
        # Using cond instead of if to maintain JIT compatibility
        ground_height = get_height(new_position[:2], terrain)

    # Check if below terrain
    is_colliding = new_position[2] < ground_height + 0.1

    # Handle collision
    def handle_collision(state_tuple):
        pos, vel, ori, ang_vel = state_tuple

        # Place aircraft just above terrain
        new_pos = pos.at[2].set(ground_height + 0.1)

        # Kill vertical velocity plus some damping on other components
        collision_damping = 0.2  # Energy lost in collision
        new_vel = vel * collision_damping
        new_vel = new_vel.at[2].set(jnp.maximum(0.0, new_vel[2]))

        return (new_pos, new_vel, ori, ang_vel)

    def normal_update(state_tuple):
        return state_tuple

    # Apply collision handling conditionally
    new_position, new_velocity, new_orientation, new_angular_velocity = lax.cond(
        is_colliding,
        handle_collision,
        normal_update,
        (new_position, new_velocity, new_orientation, new_angular_velocity)
    )

    # Combine the updated components into a new state
    new_state = jnp.concatenate([
        new_position,
        new_velocity,
        new_orientation,
        new_angular_velocity
    ])

    return new_state




def update_telemetry(aircraft: Aircraft, terrain=None) -> None:
    """
    Update telemetry data for the aircraft.

    Parameters
    ----------
    aircraft : Aircraft
        The aircraft object to update
    terrain : Any, optional
        Terrain object for ground height calculation
    """
    # Extract state
    state = aircraft.state
    position = state[:3]
    velocity_world = state[3:6]
    orientation = state[6:9]
    roll, pitch, yaw = orientation

    # Convert velocity to body frame
    velocity_body = get_velocity_in_body_frame(velocity_world, orientation)

    # Calculate derived values
    speed = jnp.linalg.norm(velocity_world)
    vertical_speed = velocity_world[2]
    alpha, beta = calculate_aerodynamic_angles(velocity_body)

    # Bank angle (simplified from roll, actual bank also depends on heading)
    bank_angle = roll  # Simplification

    # Calculate G-forces (1G = 9.81 m/s²)
    # Simplified - just using vertical acceleration component
    g_forces = 1.0  # To be enhanced in later phases

    # Get terrain height and clearance if terrain is available
    terrain_height = 0.0
    if terrain is not None:
        terrain_height = get_height(position[:2], terrain)

    terrain_clearance = position[2] - terrain_height

    # Stall warning (based on angle of attack)
    stall_warning = alpha > 0.25  # About 15 degrees, before full stall

    # Update telemetry dictionary
    aircraft.telemetry = {
        "altitude": position[2],
        "ground_speed": speed,
        "vertical_speed": vertical_speed,
        "angle_of_attack": alpha,
        "bank_angle": bank_angle,
        "g_forces": g_forces,
        "stall_warning": stall_warning,
        "terrain_height": terrain_height,
        "terrain_clearance": terrain_clearance
    }

    # Update status flags
    aircraft.is_stalled = alpha > 0.3  # About 17 degrees
    aircraft.is_crashed = terrain_clearance < 0.1 and speed < 5.0

    # Update status message
    if aircraft.is_crashed:
        aircraft.status_message = "CRASHED"
    elif aircraft.is_stalled:
        aircraft.status_message = "STALLED"
    elif stall_warning:
        aircraft.status_message = "STALL WARNING"
    elif terrain_clearance < 50.0:
        aircraft.status_message = "LOW TERRAIN"
    else:
        aircraft.status_message = ""

update_physics_jit = jax.jit(update_physics, static_argnums=(3, 4))
