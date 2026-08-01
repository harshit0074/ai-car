import pygame
import os
import math
import sys
import neat

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')

pygame.init()
TRACK = pygame.image.load(os.path.join(ASSETS_DIR, "track.png"))
SCREEN_WIDTH, SCREEN_HEIGHT = TRACK.get_size()
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('AI Car')

TRACK = TRACK.convert()
FONT = pygame.font.SysFont("Arial", 32)
BORDER_COLOR = pygame.Color(2, 105, 31, 255)
TRACK_MASK = pygame.mask.from_threshold(TRACK, BORDER_COLOR, (1, 1, 1, 255))
START_POS = (464, 480)
GENERATION = 0

def is_off_track(point):
    x, y = int(point[0]), int(point[1])
    if x < 0 or x >= SCREEN_WIDTH or y < 0 or y >= SCREEN_HEIGHT:
        return True
    return TRACK_MASK.get_at((x, y)) == 1

class Car(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.original_image = pygame.image.load(os.path.join(ASSETS_DIR, "car.png")).convert_alpha()
        self.angle = 90
        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1 / 3)
        self.rect = self.image.get_rect(center=START_POS)
        self.vel_vector = pygame.math.Vector2(0, -0.8)
        self.rotation_vel = 5
        self.direction = 0
        self.alive = True
        self.radars = []

    def update(self):
        self.radars.clear()
        self.drive()
        self.rotate()
        for radar_angle in (-80, -30, 0, 30, 80):
            self.radar(radar_angle)
        self.collision()
        self.data()

    def drive(self):
        self.rect.center += self.vel_vector * 3

    def collision(self):
        # Pixel-perfect collision check using mask overlap
        car_mask = pygame.mask.from_surface(self.image)
        offset = (self.rect.x, self.rect.y)
        if TRACK_MASK.overlap(car_mask, offset):
            self.alive = False

        # Draw Collision Points (for visual diagnostics)
        length = 40 / 3
        collision_point_right = [int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),
                                 int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)]
        collision_point_left = [int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length),
                                int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)]
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_right, 2)
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_left, 2)

    def rotate(self):
        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        if self.direction == -1:
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)

        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1 / 3)
        self.rect = self.image.get_rect(center=self.rect.center)

    def radar(self, radar_angle):
        length = 0
        x = int(self.rect.center[0])
        y = int(self.rect.center[1])

        while not is_off_track((x, y)) and length < 200:
            length += 1
            x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle)) * length)
            y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)

        # Draw Radar
        pygame.draw.line(SCREEN, (255, 255, 255, 255), self.rect.center, (x, y), 1)
        pygame.draw.circle(SCREEN, (0, 255, 0, 0), (x, y), 3)

        dist = int(math.sqrt(math.pow(self.rect.center[0] - x, 2)
                             + math.pow(self.rect.center[1] - y, 2)))

        self.radars.append([radar_angle, dist])

    def data(self):
        input = [0, 0, 0, 0, 0]
        for i, radar in enumerate(self.radars):
            input[i] = radar[1] / 200.0
        return input


def remove(index):
    cars.pop(index)
    ge.pop(index)
    nets.pop(index)


def eval_genomes(genomes, config):
    global cars, ge, nets, GENERATION

    GENERATION += 1
    print(f'Generation {GENERATION}')

    cars = []
    ge = []
    nets = []

    for genome_id, genome in genomes:
        cars.append(pygame.sprite.GroupSingle(Car()))
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.blit(TRACK, (0, 0))

        generation_text = FONT.render(f"Generation: {GENERATION}", True, (255, 255, 255))
        alive_text = FONT.render(f"Cars alive: {len(cars)}", True, (255, 255, 255))
        SCREEN.blit(generation_text, (30, 30))
        SCREEN.blit(alive_text, (30, 70))

        if len(cars) == 0:
            break

        for i, car in enumerate(cars):
            ge[i].fitness += 1
            if not car.sprite.alive:
                remove(i)

        for i, car in enumerate(cars):
            output = nets[i].activate(car.sprite.data())
            if output[0] > 0.7:
                car.sprite.direction = 1
            if output[1] > 0.7:
                car.sprite.direction = -1
            if output[0] <= 0.7 and output[1] <= 0.7:
                car.sprite.direction = 0

        # Update
        for car in cars:
            car.draw(SCREEN)
            car.update()
        pygame.display.update()


# Setup NEAT Neural Network
def run(config_path):
    global pop
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

    pop.run(eval_genomes, 50)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)
