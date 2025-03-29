import time
from collections import deque
from enum import Enum
from typing import Dict, Tuple, Any, List

import jax.numpy as jnp
import numpy as np
import pygame

from .physics import (
    Aircraft, init_state, update_physics, apply_controls,
    update_telemetry, rotation_matrix_from_euler
)
from .raytracer import Scene, render_frame
from .world import create_scene


class CameraView(Enum):
    COCKPIT = "cockpit"


class PerformanceMonitor:
    def __init__(self) -> None:
        self.metrics = {
            "fps": 60.0
        }


class SimulatorConfig:
    def __init__(self) -> None:
        self.width = 800
        self.height = 600
        self.render_quality = 0.35
        self.target_fps = 60
        self.vsync = True
        self.show_hud = True
        self.show_debug = True
        self.show_performance = True

        self.aircraft_properties = {
            "mass": 1000.0,
            "wingspan": 10.0,
            "wing_area": 16.0,
            "engine_power": 4000.0,
            "drag_coefficient": 0.025,
            "lift_coefficient": 0.3,
            "inertia_xx": 2000.0,
            "inertia_yy": 4000.0,
            "inertia_zz": 5000.0
        }
        self.start_throttle = 0.6

        self.physics_update_rate = 60
        self.physics_substeps = 3

        self.control_sensitivity = {
            "elevator": 1.0,
            "aileron": 1.0,
            "rudder": 1.0
        }
        self.control_expo = 0.2

        self.fog_density = 1.0
        self.time_of_day = 12.0


class Simulator:
    """
    Central class managing the entire simulation.
    """
    def __init__(self, config: SimulatorConfig = None, width: int = 800, height: int = 600,
                 start_throttle: float = 0.6) -> None:
        """
        Initialize the simulator with given configuration.

        Parameters
        ----------
        config : SimulatorConfig, optional
            Configuration options, by default None (creates default config)
        width : int, optional
            Window width in pixels, by default 800 (overridden by config if provided)
        height : int, optional
            Window height in pixels, by default 600 (overridden by config if provided)
        start_throttle : float, optional
            Starting throttle position, by default 0.6 (overridden by config if provided)
        """
        # Initialize config
        self.config = config if config is not None else SimulatorConfig()

        # Override config with parameters if provided
        if width != 800 or height != 600:
            self.config.width = width
            self.config.height = height
        if start_throttle != 0.6:
            self.config.start_throttle = start_throttle

        # Setup window dimensions from config
        self.width = self.config.width
        self.height = self.config.height
        self.running = False
        self.screen = None
        self.clock = None

        # Simulation time tracking for frame rate independence
        self.simulation_time = 0.0  # Total elapsed simulation time

        # Initialize aircraft, scene, and camera using config
        aircraft_properties = self.config.aircraft_properties

        self.aircraft = Aircraft(init_state(), aircraft_properties)
        terrain, sky = create_scene()
        self.scene = Scene(terrain, sky)

        # Control inputs [throttle, elevator, aileron, rudder]
        self.controls = jnp.array([start_throttle, 0.0, 0.0, 0.0])  # Start with higher throttle

        # Camera properties - always use cockpit view for simplicity
        self.camera_view = CameraView.COCKPIT
        self.camera_offset = jnp.array([0.0, 0.5, 1.2])  # Cockpit position
        self.camera_pos = jnp.zeros(3)
        self.camera_dir = jnp.array([0.0, 1.0, 0.0])  # Looking forward

        # Cockpit view data
        self.cockpit_data = {
            'position': jnp.zeros(3),
            'forward': jnp.array([0.0, 1.0, 0.0]),
            'up': jnp.array([0.0, 0.0, 1.0]),
            'right': jnp.array([1.0, 0.0, 0.0])
        }


        self.physics_update_rate = 20
        self.physics_dt = 1.0 / self.physics_update_rate
        self.physics_accumulator = 0.0
        self.skip_physics_frames = 0

        self.performance = PerformanceMonitor()
        self.render_quality = 0.35

    def initialize(self) -> None:
        """
        Set up pygame and initialize the display.
        """
        pygame.init()

        # Set up display with vsync option from config
        pygame_flags = pygame.DOUBLEBUF
        if self.config.vsync:
            pygame_flags |= pygame.HWSURFACE

        self.screen = pygame.display.set_mode((self.width, self.height), pygame_flags)
        pygame.display.set_caption("JAX Flight Simulator")
        self.clock = pygame.time.Clock()
        self.running = True

        # Create fonts
        pygame.font.init()
        self.fonts = {
            "small": pygame.font.Font(None, 16),
            "normal": pygame.font.Font(None, 24),
            "large": pygame.font.Font(None, 32),
            "title": pygame.font.Font(None, 48)
        }

        # Print control instructions once at startup
        print("\nJAXwing Flight Simulator Controls:")
        print("----------------------------------")
        print("W/S: Pitch down/up (elevator)")
        print("A/D: Roll left/right (bank to turn - primary turning method)")
        print("Q/E: Yaw left/right (rudder - helps with turning)")
        print("Space: Increase throttle")
        print("Shift: Decrease throttle")
        print("R: Reset aircraft")
        print("[ ]: Adjust render quality")
        print("ESC: Quit")
        print("\nFlight Tips:")
        print("- To turn effectively, use ROLL (A/D) to bank in the turn direction")
        print("- Use rudder (Q/E) to assist turns and prevent skidding")
        print("- Maintain airspeed with throttle during turns")

    def handle_input(self) -> None:
        """
        Process user controls.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    self.aircraft.state = init_state()
                    self.aircraft.is_crashed = False
                    self.aircraft.is_stalled = False
                    self.aircraft.status_message = ""

        keys = pygame.key.get_pressed()
        throttle, elevator, aileron, rudder = self.controls

        if keys[pygame.K_SPACE]:
            throttle = min(throttle + 0.01, 1.0)
        if keys[pygame.K_LSHIFT]:
            throttle = max(throttle - 0.01, 0.0)

        if keys[pygame.K_s]:
            elevator = min(elevator + 0.05, 1.0)
        elif keys[pygame.K_w]:
            elevator = max(elevator - 0.05, -1.0)
        else:
            elevator *= 0.9

        if keys[pygame.K_a]:
            aileron = min(aileron + 0.05, 1.0)
        elif keys[pygame.K_d]:
            aileron = max(aileron - 0.05, -1.0)
        else:
            aileron *= 0.95

        if keys[pygame.K_q]:
            rudder = min(rudder + 0.05, 1.0)
        elif keys[pygame.K_e]:
            rudder = max(rudder - 0.05, -1.0)
        else:
            rudder *= 0.95

        self.controls = jnp.array([throttle, elevator, aileron, rudder])

    def update_camera(self) -> None:
        """
        Update camera position and direction based on aircraft position.
        """
        aircraft_pos = self.aircraft.state[:3]
        aircraft_orientation = self.aircraft.state[6:9]
        roll, pitch, yaw = aircraft_orientation

        # Get rotation matrix from aircraft orientation
        rotation_matrix = rotation_matrix_from_euler(roll, pitch, yaw)

        # Calculate aircraft's forward direction
        forward_dir = jnp.matmul(rotation_matrix, jnp.array([0.0, 1.0, 0.0]))
        forward_dir = forward_dir / jnp.linalg.norm(forward_dir)

        # Calculate aircraft's up direction
        up_dir = jnp.matmul(rotation_matrix, jnp.array([0.0, 0.0, 1.0]))

        # Calculate aircraft's right direction
        right_dir = jnp.cross(forward_dir, up_dir)
        right_dir = right_dir / jnp.linalg.norm(right_dir)

        # Position camera at aircraft position with offset for cockpit view
        cockpit_offset = jnp.matmul(rotation_matrix, jnp.array([0.0, 0.5, 1.2]))
        self.camera_pos = aircraft_pos + cockpit_offset
        self.camera_dir = forward_dir

        # Store cockpit view data for aircraft nose rendering
        self.cockpit_data = {
            'position': aircraft_pos,
            'forward': forward_dir,
            'up': up_dir,
            'right': right_dir
        }

    def update(self, dt: float, simulation_time: float = 0.0) -> None:
        """
        Advance the simulation state with time-dependent effects.

        Parameters
        ----------
        dt : float
            Time step in seconds
        simulation_time : float, optional
            Current simulation time in seconds, used for wind effects and other
            time-dependent features
        """
        # For Phase 3, implement fixed timestep physics with multiple substeps
        # This improves stability and accuracy of physics simulation
        self.physics_accumulator += dt

        # Skip some physics frames to improve performance
        self.skip_physics_frames = (self.skip_physics_frames + 1) % 3  # Only do physics every 3 frames

        if self.skip_physics_frames == 0:
            # Perform just one physics step regardless of accumulator
            # This ensures we don't get caught in catchup loops
            if self.physics_accumulator >= self.physics_dt:
                # Apply controls
                processed_controls = apply_controls(self.controls)

                # Use extremely simplified physics for speed
                # Just do basic position and rotation updates without complex forces
                position = self.aircraft.state[:3]
                velocity = self.aircraft.state[3:6]
                orientation = self.aircraft.state[6:9]

                # Get rotation matrix
                roll, pitch, yaw = orientation
                R = rotation_matrix_from_euler(roll, pitch, yaw)

                # Very basic control effects
                throttle, elevator, aileron, rudder = processed_controls

                # Apply simple attitude changes based on controls
                roll_rate = aileron * 2.0  # Simple roll rate
                pitch_rate = elevator * 1.0  # Simple pitch rate
                yaw_rate = rudder * 0.5    # Simple yaw rate

                # Update orientation
                new_roll = roll + roll_rate * self.physics_dt
                new_pitch = pitch + pitch_rate * self.physics_dt
                new_yaw = yaw + yaw_rate * self.physics_dt
                
                # Update rotation matrix with new orientation
                new_R = rotation_matrix_from_euler(new_roll, new_pitch, new_yaw)
                
                # Calculate speed (magnitude of velocity)
                speed = jnp.linalg.norm(velocity)
                speed = jnp.maximum(speed, 0.1)  # Avoid division by zero
                
                # Get the direction vectors based on the NEW orientation
                forward_vec = jnp.array([0, 1, 0])
                forward_world = jnp.matmul(new_R, forward_vec)
                
                right_vec = jnp.array([1, 0, 0])
                right_world = jnp.matmul(new_R, right_vec)
                
                up_vec = jnp.array([0, 0, 1])
                up_world = jnp.matmul(new_R, up_vec)
                
                # Implement proper turning physics:
                # 1. Rudder effect - changes yaw and induces slight roll
                # 2. Roll/bank effect - redirects lift to create horizontal turning force
                # 3. Maintain speed while changing direction vector
                
                # Calculate rudder-induced yaw effect
                rudder_turn_force = right_world * (rudder * 2.0)  # Increased rudder effectiveness
                
                # Calculate bank-induced turn effect (more significant)
                # When rolled, lift force creates a turning moment (the main turning mechanism for aircraft)
                bank_force = jnp.sin(new_roll) * 4.0  # Further increased effectiveness
                bank_turn_force = right_world * bank_force
                
                # Acceleration components
                throttle_accel = forward_world * (throttle * 5.0)  # Forward acceleration from throttle
                turn_accel = rudder_turn_force + bank_turn_force   # Combined turning acceleration
                
                # Apply acceleration to velocity
                accel = throttle_accel + turn_accel
                new_velocity_raw = velocity + accel * self.physics_dt
                
                # Option 1: Complete velocity replacement - most direct turning
                # This fully aligns velocity with the aircraft's new orientation while preserving speed
                new_velocity_aligned = forward_world * speed * (1.0 + throttle * 0.1)
                
                # Option 2: Blend between current velocity and aligned velocity for smooth transition
                # Higher speeds make it harder to turn sharply (realistic flight dynamics)
                max_turn_rate = 1.0 - jnp.clip(speed / 100.0, 0.0, 0.8)  # Max turn rate decreases with speed
                
                # Calculate combined turn force magnitude (from both bank and rudder)
                combined_turn_force = jnp.linalg.norm(bank_turn_force) + jnp.linalg.norm(rudder_turn_force)
                turn_strength = combined_turn_force / 4.0  # Normalized turning intensity
                
                # Calculate how much to align with forward vector (higher = more direct turning)
                # Increased significantly to make turning more responsive
                blend_factor = max_turn_rate * turn_strength * 0.8
                blend_factor = jnp.minimum(blend_factor, 0.4)  # Cap maximum turn rate for stability
                
                # Blend between current velocity direction and aligned velocity
                new_velocity = new_velocity_raw * (1.0 - blend_factor) + new_velocity_aligned * blend_factor
                
                # Apply movement based on updated velocity (average of old and new velocity for better accuracy)
                avg_velocity = (velocity + new_velocity) * 0.5
                new_position = position + avg_velocity * self.physics_dt

                # Limits
                new_pitch = jnp.clip(new_pitch, -1.0, 1.0)

                # Update state
                self.aircraft.state = self.aircraft.state.at[:3].set(new_position)
                self.aircraft.state = self.aircraft.state.at[3:6].set(new_velocity)
                self.aircraft.state = self.aircraft.state.at[6:9].set(jnp.array([new_roll, new_pitch, new_yaw]))

                # Update telemetry with simplified calculations
                self.aircraft.telemetry["altitude"] = new_position[2]
                self.aircraft.telemetry["ground_speed"] = jnp.linalg.norm(new_velocity)
                self.aircraft.telemetry["vertical_speed"] = new_velocity[2]

                # Reset accumulator
                self.physics_accumulator = 0

        # Update camera based on current view mode
        self.update_camera()

    def render(self) -> None:
        """
        Render the current state to the screen.
        """
        # Calculate effective width and height based on render quality
        effective_width = int(self.width * self.render_quality)
        effective_height = int(self.height * self.render_quality)

        # Always in cockpit view, with rendering handled in pygame
        is_cockpit_view = True
        aircraft_state = None

        jax_image = render_frame(
            self.camera_pos,
            self.camera_dir,
            self.scene,
            effective_width,
            effective_height,
            in_cockpit_view=False,
            aircraft_state=None
        )

        # Convert JAX array to numpy
        image = np.array(jax_image)

        # Simple upscaling with repeat for lower resolution renders
        if self.render_quality < 1.0:
            image = np.repeat(np.repeat(image, int(1/self.render_quality), axis=0), int(1/self.render_quality), axis=1)
            image = image[:self.height, :self.width]

        # Convert to pygame surface and display
        try:
            pygame_surface = pygame.surfarray.make_surface(
                np.transpose(image, (1, 0, 2))
            )
        except Exception as e:
            print(f"Error creating surface: {e}")
            # Fallback method
            pygame_surface = pygame.Surface((self.width, self.height))
            for y in range(min(self.height, image.shape[0])):
                for x in range(min(self.width, image.shape[1])):
                    pygame_surface.set_at((x, y), tuple(image[y, x]))

        self.screen.blit(pygame_surface, (0, 0))

        # Draw cockpit overlay directly with Pygame
        # Create a transparent surface for cockpit elements
        cockpit_overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        dark_gray = (60, 60, 60, 240)
        light_gray = (150, 150, 150, 240)
        black = (20, 20, 20, 255)

        # Cockpit dimensions proportional to screen size
        nose_bottom_width = int(self.width * 0.5)
        cockpit_height = int(self.height * 0.33)

        nose_points = [
            (self.width // 2, self.height - cockpit_height // 2),  # Top center
            (self.width // 2 - nose_bottom_width // 2, self.height),  # Bottom left
            (self.width // 2 + nose_bottom_width // 2, self.height)   # Bottom right
        ]

        # Draw the main aircraft nose shape
        pygame.draw.polygon(cockpit_overlay, dark_gray, nose_points)

        # Add multiple highlight lines for more detailed cockpit
        highlight_start = (self.width // 2, self.height - cockpit_height // 2)
        highlight_end = (self.width // 2, self.height)
        pygame.draw.line(cockpit_overlay, light_gray, highlight_start, highlight_end, 3)

        # Add side panels for more realistic cockpit
        left_panel_x = self.width // 2 - nose_bottom_width // 4
        right_panel_x = self.width // 2 + nose_bottom_width // 4
        panel_top = self.height - cockpit_height // 2
        pygame.draw.line(cockpit_overlay, light_gray,
                        (left_panel_x, panel_top),
                        (left_panel_x - nose_bottom_width // 6, self.height), 2)
        pygame.draw.line(cockpit_overlay, light_gray,
                        (right_panel_x, panel_top),
                        (right_panel_x + nose_bottom_width // 6, self.height), 2)

        # Draw cockpit frame edges with improved thickness based on screen size
        frame_width = max(4, int(self.width * 0.01))  # Responsive width
        pygame.draw.rect(cockpit_overlay, black,
                        (0, self.height - cockpit_height, frame_width, cockpit_height))
        pygame.draw.rect(cockpit_overlay, black,
                        (self.width - frame_width, self.height - cockpit_height,
                         frame_width, cockpit_height))

        # Draw dashboard/instrument panel with shadow for depth
        panel_height = int(self.height * 0.03)  # Responsive height
        panel_width = int(self.width * 0.6)
        panel_x = (self.width - panel_width) // 2
        panel_y = self.height - cockpit_height - panel_height

        # Draw shadow first
        shadow_offset = 2
        pygame.draw.rect(cockpit_overlay, (10, 10, 10, 200),
                        (panel_x + shadow_offset, panel_y + shadow_offset,
                         panel_width, panel_height))

        # Then draw panel
        pygame.draw.rect(cockpit_overlay, black,
                        (panel_x, panel_y, panel_width, panel_height))

        # Add cockpit overlay to main screen
        self.screen.blit(cockpit_overlay, (0, 0))

        # Always render the HUD
        self.render_hud()

        # Removed cockpit view indicator since it's the only option

        # Display status messages prominently if needed
        if self.aircraft.status_message:
            self.render_status_message()

        # Update display
        pygame.display.flip()

    def render_hud(self) -> None:
        """
        Render the heads-up display with flight information, performance metrics and control visualizer.
        """
        # Get telemetry data
        telemetry = self.aircraft.telemetry

        # Set up colors and fonts
        white = (255, 255, 255)
        yellow = (255, 255, 0)
        red = (255, 0, 0)
        green = (0, 255, 0)
        font = self.fonts["normal"]
        small_font = self.fonts["small"]

        # Draw artificial horizon line at center of screen
        # Get aircraft pitch and roll
        roll, pitch, _ = self.aircraft.state[6:9]

        # Calculate horizon line offset based on pitch
        pitch_offset = int(pitch * self.height * 0.3)  # Scale factor to make pitch more visible

        # Draw horizon line with roll
        horizon_length = 100
        center_x, center_y = self.width // 2, self.height // 2

        # Calculate endpoints of rolled horizon line
        sin_roll, cos_roll = jnp.sin(roll), jnp.cos(roll)
        end1_x = center_x - int(horizon_length * cos_roll)
        end1_y = center_y - int(horizon_length * sin_roll) + pitch_offset
        end2_x = center_x + int(horizon_length * cos_roll)
        end2_y = center_y + int(horizon_length * sin_roll) + pitch_offset

        # Draw horizon line
        pygame.draw.line(self.screen, white, (end1_x, end1_y), (end2_x, end2_y), 2)

        # Draw aircraft reference
        pygame.draw.line(self.screen, yellow, (center_x - 30, center_y), (center_x - 10, center_y), 2)
        pygame.draw.line(self.screen, yellow, (center_x + 10, center_y), (center_x + 30, center_y), 2)
        pygame.draw.line(self.screen, yellow, (center_x, center_y - 5), (center_x, center_y + 5), 2)

        # ------------------- Left side panel: Flight data -------------------
        # HUD background - even smaller semi-transparent panel for left side
        panel_surface = pygame.Surface((200, 115), pygame.SRCALPHA)
        panel_surface.fill((0, 0, 0, 100))  # RGBA, even more transparent black
        self.screen.blit(panel_surface, (8, 8))

        y_pos = 12

        # Altitude
        altitude = telemetry["altitude"]
        terrain_clearance = telemetry["terrain_clearance"]
        alt_color = red if terrain_clearance < 50 else white
        alt_text = font.render(f"ALT: {altitude:.1f} m", True, alt_color)
        self.screen.blit(alt_text, (15, y_pos))
        y_pos += 25

        # Speed
        speed = telemetry["ground_speed"]
        speed_text = font.render(f"SPD: {speed:.1f} m/s", True, white)
        self.screen.blit(speed_text, (15, y_pos))
        y_pos += 25

        # Vertical speed
        vspeed = telemetry["vertical_speed"]
        vspeed_color = green if vspeed > 0 else yellow if vspeed < 0 else white
        vspeed_text = font.render(f"V/S: {vspeed:.1f} m/s", True, vspeed_color)
        self.screen.blit(vspeed_text, (15, y_pos))
        y_pos += 25

        # Throttle
        throttle = self.controls[0]
        throttle_text = font.render(f"THR: {throttle:.2f}", True, white)
        self.screen.blit(throttle_text, (15, y_pos))

        # Draw throttle gauge (even smaller)
        throttle_x, throttle_y = 115, y_pos + 10
        pygame.draw.rect(self.screen, (50, 50, 50), (throttle_x, throttle_y - 10, 80, 10))
        pygame.draw.rect(self.screen, (200, 100, 0), (throttle_x, throttle_y - 10, int(throttle * 80), 10))

        # ------------------- Control visualizer (right bottom) -------------------
        # Control panel background - even smaller size
        control_panel = pygame.Surface((100, 100), pygame.SRCALPHA)
        control_panel.fill((0, 0, 0, 100))  # Even more transparent black
        self.screen.blit(control_panel, (self.width - 108, self.height - 108))

        # Control text and values
        throttle, elevator, aileron, rudder = self.controls
        control_x = self.width - 100
        control_y = self.height - 100

        # Control visualizer - smaller size
        stick_x = control_x + 50
        stick_y = control_y + 50
        stick_radius = 22

        # Draw outer circle
        pygame.draw.circle(self.screen, (100, 100, 100), (stick_x, stick_y), stick_radius, 1)

        # Draw crosshairs
        pygame.draw.line(self.screen, (100, 100, 100), (stick_x - stick_radius, stick_y), (stick_x + stick_radius, stick_y), 1)
        pygame.draw.line(self.screen, (100, 100, 100), (stick_x, stick_y - stick_radius), (stick_x, stick_y + stick_radius), 1)

        # Draw stick position (aileron, elevator)
        stick_pos_x = stick_x + aileron * stick_radius
        stick_pos_y = stick_y - elevator * stick_radius
        pygame.draw.circle(self.screen, (255, 0, 0), (int(stick_pos_x), int(stick_pos_y)), 4)

        # Draw rudder indicator
        rudder_x = stick_x
        rudder_y = stick_y + stick_radius + 8
        rudder_width = stick_radius * 2

        pygame.draw.rect(self.screen, (100, 100, 100), (rudder_x - rudder_width // 2, rudder_y, rudder_width, 4), 1)
        rudder_pos = rudder_x + rudder * rudder_width // 2
        pygame.draw.rect(self.screen, (255, 0, 0), (int(rudder_pos - 3), rudder_y, 6, 4))

        # Draw throttle indicator
        throttle_x = stick_x - stick_radius - 8
        throttle_y = stick_y
        throttle_height = stick_radius * 2

        pygame.draw.rect(self.screen, (100, 100, 100), (throttle_x - 4, throttle_y - throttle_height // 2, 4, throttle_height), 1)
        throttle_pos = throttle_y + throttle_height // 2 - throttle * throttle_height
        pygame.draw.rect(self.screen, (255, 0, 0), (throttle_x - 4, int(throttle_pos - 2), 4, 4))

        # Draw stall warning if active
        if telemetry["stall_warning"]:
            stall_warning = self.fonts["large"].render("STALL", True, red)
            self.screen.blit(stall_warning, (self.width // 2 - stall_warning.get_width() // 2, 50))

        # Draw terrain warning if too close to ground
        if terrain_clearance < 100 and terrain_clearance > 0:
            terrain_warning = self.fonts["large"].render(f"TERRAIN: {terrain_clearance:.1f}m", True, yellow)
            self.screen.blit(terrain_warning, (self.width // 2 - terrain_warning.get_width() // 2, 90))

    # Debug and performance info methods removed and incorporated into render_hud

    def render_status_message(self) -> None:
        """
        Render important status messages with a simple approach.
        """
        if not self.aircraft.status_message:
            return

        font = self.fonts["title"]

        # Use single color for all messages
        color = (255, 0, 0)  # Red for all status messages

        text = font.render(self.aircraft.status_message, True, color)

        # Center on screen
        text_x = self.width // 2 - text.get_width() // 2
        text_y = self.height // 4 - text.get_height() // 2

        # Simple background
        bg_rect = pygame.Rect(text_x - 10, text_y - 10, text.get_width() + 20, text.get_height() + 20)
        pygame.draw.rect(self.screen, (0, 0, 0), bg_rect)

        # Render text
        self.screen.blit(text, (text_x, text_y))

        # Add recovery hint if crashed
        if self.aircraft.status_message == "CRASHED":
            hint = self.fonts["normal"].render("Press R to reset aircraft", True, (255, 255, 255))
            hint_x = self.width // 2 - hint.get_width() // 2
            self.screen.blit(hint, (hint_x, text_y + text.get_height() + 10))

    def main_loop(self) -> None:
        """
        Run the main simulation loop with frame rate independence.
        """
        last_time = time.time()

        # Keyboard key mapping - expanded for Phase 4
        key_description = [
            "Flight Controls:",
            "W/S: Pitch down/up",
            "A/D: Roll left/right",
            "Q/E: Yaw left/right",
            "Space: Increase throttle",
            "Shift: Decrease throttle",
            "R: Reset aircraft",
            "",
            "System Controls:",
            "[ ]: Adjust render quality",
            "ESC: Quit"
        ]

        instruction_start = time.time()
        fixed_physics_dt = 1.0 / self.config.physics_update_rate
        physics_accumulator = 0.0
        simulation_time = 0.0

        while self.running:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            frame_start_time = time.time()
            
            dt = min(dt, 0.1)
            physics_accumulator += dt
            simulation_time += dt

            self.handle_input()

            while physics_accumulator >= fixed_physics_dt:
                self.update(fixed_physics_dt, simulation_time)
                physics_accumulator -= fixed_physics_dt

            self.render()

            if False:
                # Draw semi-transparent background with enhanced styling
                instr_width = 500  # Wider for expanded controls
                instr_height = 500  # Taller for more controls
                instr_background = pygame.Surface((instr_width, instr_height), pygame.SRCALPHA)
                instr_background.fill((0, 0, 0, 230))  # More opaque for better readability

                # Add border
                border_thickness = 2
                pygame.draw.rect(instr_background, (100, 100, 100, 255),
                                pygame.Rect(0, 0, instr_width, instr_height), border_thickness)

                # Add title
                title_text = self.fonts["large"].render("JAXwing Flight Controls", True, (255, 200, 0))
                instr_background.blit(title_text,
                                     (instr_width // 2 - title_text.get_width() // 2, 15))

                # Draw instructions with better spacing and sections
                y_offset = 60  # Start lower to accommodate title
                for i, line in enumerate(key_description):
                    # Different color for section headers (empty strings are spacing)
                    if line and line[-1] == ":":  # Section header
                        color = (255, 200, 0)  # Gold for headers
                        font = self.fonts["normal"]
                        # Add extra space before sections (except the first)
                        if i > 0:
                            y_offset += 5
                    elif not line:  # Empty line for spacing
                        continue  # Skip rendering but maintain list structure
                    else:
                        color = (255, 255, 255)  # White for regular text
                        font = self.fonts["small"]

                    instr_text = font.render(line, True, color)
                    instr_background.blit(instr_text, (25, y_offset + i * 22))

                # Position and draw instructions background
                instr_x = self.width // 2 - instr_width // 2
                instr_y = self.height // 2 - instr_height // 2
                self.screen.blit(instr_background, (instr_x, instr_y))

                # Add dismiss message
                dismiss_text = self.fonts["small"].render("Press any key to dismiss", True, (200, 200, 200))
                self.screen.blit(dismiss_text,
                                (self.width // 2 - dismiss_text.get_width() // 2,
                                 instr_y + instr_height - 30))

                # Update the display again to show instructions
                pygame.display.flip()

            target_fps = self.config.target_fps
            fps = self.clock.get_fps()
            self.performance.metrics["fps"] = fps
            self.clock.tick(target_fps)

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFTBRACKET]:
                self.render_quality = max(0.02, self.render_quality - 0.01)
            elif keys[pygame.K_RIGHTBRACKET]:
                self.render_quality = min(1.0, self.render_quality + 0.01)

        pygame.quit()


def run_simulation() -> None:
    """
    Entry point to run the simulator.
    """
    simulator = Simulator()
    simulator.initialize()
    simulator.main_loop()