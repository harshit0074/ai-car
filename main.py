"""Autonomous Self-Driving Car Simulation with NEAT (NeuroEvolution of Augmenting Topologies).

This simulation utilizes genetic algorithms to train autonomous agents to navigate complex
2D race tracks using simulated raycasting radar sensors and feedforward neural networks.
"""

import argparse
import math
import os
import pickle
import sys
from typing import List, Tuple

import neat
import pygame

# Directory & Asset Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
TRACKS_DIR = os.path.join(ASSETS_DIR, 'tracks')
DEFAULT_CONFIG = os.path.join(BASE_DIR, 'config.txt')

# Simulation Constants
BORDER_COLOR = pygame.Color(2, 105, 31, 255)  # Green grass border
START_POS = (464, 480)
RADAR_ANGLES = (-80, -30, 0, 30, 80)
RADAR_MAX_LENGTH = 200
BASE_SPEED = 2.4
ROTATION_VEL = 5.0
CAR_SCALE = 0.1 / 3

# Global Simulation State
GENERATION = 0
BEST_FITNESS_EVER = 0.0


def resolve_track_path(track_arg: str) -> str:
    """Resolve track identifier or path to an absolute image path."""
    # Check if user specified a number (e.g. 1, 2, 3...)
    candidate_paths = []
    if track_arg.isdigit():
        num = track_arg
        candidate_paths.extend([
            os.path.join(TRACKS_DIR, f"track{num}.png"),
            os.path.join(ASSETS_DIR, f"track{num}.png"),
            os.path.join(BASE_DIR, f"track{num}.png"),
            os.path.join(TRACKS_DIR, "track.png") if num == "1" else None,
            os.path.join(ASSETS_DIR, "track.png") if num == "1" else None,
        ])
    else:
        candidate_paths.extend([
            track_arg,
            os.path.join(TRACKS_DIR, track_arg),
            os.path.join(ASSETS_DIR, track_arg),
            os.path.join(BASE_DIR, track_arg),
            os.path.join(TRACKS_DIR, f"{track_arg}.png"),
            os.path.join(ASSETS_DIR, f"{track_arg}.png"),
            os.path.join(BASE_DIR, f"{track_arg}.png"),
        ])

    for path in candidate_paths:
        if path and os.path.isfile(path):
            return os.path.abspath(path)

    # Fallback to default track in assets
    default_track = os.path.join(ASSETS_DIR, "track.png")
    if os.path.isfile(default_track):
        return default_track
    root_track = os.path.join(BASE_DIR, "track.png")
    if os.path.isfile(root_track):
        return root_track

    raise FileNotFoundError(f"Track file '{track_arg}' could not be located.")


class Car(pygame.sprite.Sprite):
    """Represents an autonomous vehicle agent equipped with raycasting radars."""

    # Cached surface to avoid disk I/O per agent instantiation
    _cached_car_image = None

    def __init__(self, start_pos: Tuple[int, int] = START_POS):
        super().__init__()
        if Car._cached_car_image is None:
            car_path = os.path.join(ASSETS_DIR, "car.png")
            Car._cached_car_image = pygame.image.load(car_path).convert_alpha()

        self.original_image = Car._cached_car_image
        self.angle = 90.0
        self.image = pygame.transform.rotozoom(self.original_image, self.angle, CAR_SCALE)
        self.rect = self.image.get_rect(center=start_pos)
        self.vel_vector = pygame.math.Vector2(0, -BASE_SPEED)
        self.rotation_vel = ROTATION_VEL
        self.direction = 0  # -1: Left, 0: Straight, 1: Right
        self.alive = True
        self.radars: List[List[float]] = []
        self.distance_traveled = 0.0

    def update(self, screen: pygame.Surface, track_mask: pygame.mask.Mask, screen_dims: Tuple[int, int]):
        """Advance agent state for one frame."""
        self.radars.clear()
        self.drive()
        self.rotate()

        screen_w, screen_h = screen_dims
        for radar_angle in RADAR_ANGLES:
            self.radar(radar_angle, screen, track_mask, screen_w, screen_h)

        self.collision(screen, track_mask)
        self.distance_traveled += self.vel_vector.length()

    def drive(self):
        """Translate vehicle along current velocity vector."""
        self.rect.center += self.vel_vector

    def collision(self, screen: pygame.Surface, track_mask: pygame.mask.Mask):
        """Pixel-perfect collision check using bitmask overlap."""
        car_mask = pygame.mask.from_surface(self.image)
        offset = (self.rect.x, self.rect.y)
        if track_mask.overlap(car_mask, offset):
            self.alive = False

        # Visual collision sensors (diagnostic points)
        length = 40 / 3
        right_pt = [
            int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),
            int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)
        ]
        left_pt = [
            int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length),
            int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)
        ]
        pygame.draw.circle(screen, (0, 255, 255), right_pt, 2)
        pygame.draw.circle(screen, (0, 255, 255), left_pt, 2)

    def rotate(self):
        """Adjust orientation and directional heading vector."""
        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        elif self.direction == -1:
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)

        self.image = pygame.transform.rotozoom(self.original_image, self.angle, CAR_SCALE)
        self.rect = self.image.get_rect(center=self.rect.center)

    def radar(
        self,
        radar_angle: float,
        screen: pygame.Surface,
        track_mask: pygame.mask.Mask,
        screen_w: int,
        screen_h: int
    ):
        """Cast ray until boundary hit or maximum range is reached."""
        length = 0
        cx, cy = self.rect.center
        rad = math.radians(self.angle + radar_angle)
        cos_val = math.cos(rad)
        sin_val = math.sin(rad)

        x = int(cx)
        y = int(cy)

        while length < RADAR_MAX_LENGTH:
            length += 1
            x = int(cx + cos_val * length)
            y = int(cy - sin_val * length)

            # Check boundary conditions
            if x < 0 or x >= screen_w or y < 0 or y >= screen_h:
                break
            if track_mask.get_at((x, y)) == 1:
                break

        # Render radar raycast lines and terminal impact points
        pygame.draw.line(screen, (255, 255, 255), (cx, cy), (x, y), 1)
        pygame.draw.circle(screen, (0, 255, 0), (x, y), 3)

        self.radars.append([radar_angle, length])

    def data(self) -> List[float]:
        """Normalize sensor radar inputs to [0.0, 1.0] for neural network ingestion."""
        input_data = [0.0] * len(RADAR_ANGLES)
        for i, radar in enumerate(self.radars):
            input_data[i] = radar[1] / float(RADAR_MAX_LENGTH)
        return input_data


class SimulationEnvironment:
    """Manages Pygame canvas, graphics rendering, HUD telemetry, and event loop."""

    def __init__(self, track_path: str, fps: int = 60):
        pygame.init()
        self.track_path = track_path
        self.track_name = os.path.basename(track_path)
        raw_track = pygame.image.load(track_path)
        self.screen_width, self.screen_height = raw_track.get_size()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.track_image = raw_track.convert()
        pygame.display.set_caption(f'Autonomous AI Car - NEAT Simulation [{self.track_name}]')

        self.font = pygame.font.SysFont("Arial", 22)
        self.font_bold = pygame.font.SysFont("Arial", 24, bold=True)
        self.track_mask = pygame.mask.from_threshold(self.track_image, BORDER_COLOR, (1, 1, 1, 255))
        self.clock = pygame.time.Clock()
        self.fps = fps

    def render_hud(self, generation: int, alive_count: int, max_fitness: float, mode_str: str = "Training"):
        """Render informative telemetry dashboard onto screen."""
        # Semi-transparent overlay box
        hud_surface = pygame.Surface((310, 140), pygame.SRCALPHA)
        hud_surface.fill((15, 23, 42, 210))  # Dark slate blue HUD panel
        pygame.draw.rect(hud_surface, (59, 130, 246), hud_surface.get_rect(), width=2, border_radius=8)

        # Telemetry metrics
        title = self.font_bold.render("NEAT Telemetry HUD", True, (96, 165, 250))
        gen_txt = self.font.render(f"Generation: {generation}", True, (241, 245, 249))
        alive_txt = self.font.render(f"Active Agents: {alive_count}", True, (52, 211, 153))
        fit_txt = self.font.render(f"Peak Fitness: {int(max_fitness)}", True, (251, 191, 36))
        track_txt = self.font.render(f"Track: {self.track_name} ({mode_str})", True, (148, 163, 184))

        hud_surface.blit(title, (14, 10))
        hud_surface.blit(gen_txt, (14, 38))
        hud_surface.blit(alive_txt, (14, 62))
        hud_surface.blit(fit_txt, (14, 86))
        hud_surface.blit(track_txt, (14, 110))

        self.screen.blit(hud_surface, (20, 20))


def eval_genomes(genomes, config, env: SimulationEnvironment):
    """Fitness evaluation function invoked across generations by NEAT."""
    global GENERATION, BEST_FITNESS_EVER

    GENERATION += 1
    cars: List[pygame.sprite.GroupSingle] = []
    ge: List[neat.DefaultGenome] = []
    nets: List[neat.nn.FeedForwardNetwork] = []

    for genome_id, genome in genomes:
        cars.append(pygame.sprite.GroupSingle(Car(START_POS)))
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0.0

    running = True
    screen_dims = (env.screen_width, env.screen_height)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit(0)

        # Clear canvas with track background
        env.screen.blit(env.track_image, (0, 0))

        # Check termination condition
        if len(cars) == 0:
            break

        # Neural network inference for each surviving agent
        for i, car in enumerate(cars):
            inputs = car.sprite.data()
            output = nets[i].activate(inputs)

            # Steering decision with priority logic
            if output[0] > 0.7 and output[0] > output[1]:
                car.sprite.direction = 1   # Steer Right
            elif output[1] > 0.7 and output[1] > output[0]:
                car.sprite.direction = -1  # Steer Left
            else:
                car.sprite.direction = 0   # Maintain heading

        # Physics, collision detection & reverse safe removal
        current_max_fitness = BEST_FITNESS_EVER
        for i in range(len(cars) - 1, -1, -1):
            car_sprite = cars[i].sprite
            # Increment fitness for survival & exploration
            ge[i].fitness += 1.0

            if ge[i].fitness > BEST_FITNESS_EVER:
                BEST_FITNESS_EVER = ge[i].fitness
            if ge[i].fitness > current_max_fitness:
                current_max_fitness = ge[i].fitness

            if not car_sprite.alive:
                cars.pop(i)
                ge.pop(i)
                nets.pop(i)

        # Update and render agents
        for car in cars:
            car.draw(env.screen)
            car.sprite.update(env.screen, env.track_mask, screen_dims)

        # Draw HUD dashboard
        env.render_hud(
            generation=GENERATION,
            alive_count=len(cars),
            max_fitness=current_max_fitness,
            mode_str="Training"
        )

        pygame.display.update()
        if env.fps > 0:
            env.clock.tick(env.fps)


def replay_champion(model_path: str, config_path: str, env: SimulationEnvironment):
    """Replay a serialized champion genome on the specified track."""
    print(f"\n[*] Loading pre-trained champion genome from: {model_path}")
    with open(model_path, 'rb') as f:
        champion_genome = pickle.load(f)

    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    net = neat.nn.FeedForwardNetwork.create(champion_genome, config)
    car_group = pygame.sprite.GroupSingle(Car(START_POS))
    screen_dims = (env.screen_width, env.screen_height)
    fitness = 0.0

    print("[*] Replay started. Press ESC or close window to exit.")
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit(0)

        env.screen.blit(env.track_image, (0, 0))

        if not car_group.sprite.alive:
            print(f"[!] Champion car crashed after fitness score: {fitness:.1f}. Restarting test lap...")
            car_group = pygame.sprite.GroupSingle(Car(START_POS))
            fitness = 0.0

        # Inference
        inputs = car_group.sprite.data()
        output = net.activate(inputs)
        if output[0] > 0.7 and output[0] > output[1]:
            car_group.sprite.direction = 1
        elif output[1] > 0.7 and output[1] > output[0]:
            car_group.sprite.direction = -1
        else:
            car_group.sprite.direction = 0

        fitness += 1.0
        car_group.draw(env.screen)
        car_group.sprite.update(env.screen, env.track_mask, screen_dims)

        env.render_hud(
            generation=1,
            alive_count=1 if car_group.sprite.alive else 0,
            max_fitness=fitness,
            mode_str="Champion Replay"
        )

        pygame.display.update()
        clock.tick(env.fps if env.fps > 0 else 60)


def run_training(config_path: str, track_path: str, generations: int, save_path: str, fps: int):
    """Configure NEAT population and launch evolutionary training loop."""
    print("==================================================")
    print("  Autonomous AI Car Simulation via NEAT")
    print("==================================================")
    print(f"[*] Track:       {track_path}")
    print(f"[*] Config:      {config_path}")
    print(f"[*] Generations: {generations}")
    print(f"[*] Save Target: {save_path}")
    print(f"[*] FPS Limit:   {fps if fps > 0 else 'Uncapped'}")
    print("==================================================\n")

    env = SimulationEnvironment(track_path=track_path, fps=fps)

    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # Wrap evaluator to include environment reference
    def evaluator(genomes, cfg):
        return eval_genomes(genomes, cfg, env)

    try:
        winner = pop.run(evaluator, generations)
        print("\n[+] Training successfully concluded.")
        print(f"[+] Champion Genome Key: {winner.key} | Fitness: {winner.fitness}")

        if save_path:
            with open(save_path, "wb") as f:
                pickle.dump(winner, f)
            print(f"[+] Champion model serialized to: {os.path.abspath(save_path)}")

    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user.")
    finally:
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Self-Driving Car Simulation powered by NEAT & Pygame.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--track", "-t",
        type=str,
        default="1",
        help="Track selection: index 1-5 or custom path to track image."
    )
    parser.add_argument(
        "--generations", "-g",
        type=int,
        default=50,
        help="Number of evolutionary generations to simulate."
    )
    parser.add_argument(
        "--save-best", "-s",
        type=str,
        default="champion.pkl",
        help="Destination filepath to serialize the highest-fitness genome."
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Path to pre-trained champion genome (.pkl) to replay without training."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Frame rate cap. Set to 0 for maximum unconstrained simulation speed."
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to NEAT configuration file."
    )

    args = parser.parse_args()

    track_path = resolve_track_path(args.track)

    if args.test:
        env = SimulationEnvironment(track_path=track_path, fps=args.fps)
        replay_champion(args.test, args.config, env)
    else:
        run_training(
            config_path=args.config,
            track_path=track_path,
            generations=args.generations,
            save_path=args.save_best,
            fps=args.fps
        )


if __name__ == '__main__':
    main()
