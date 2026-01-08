import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import random

# Constants
GREEN_DURATION = 100 
RED_DURATION = 100    
YELLOW_DURATION = 30  
CITY_SIZE = 100
VISIBLE_RANGE = 60
CAR_LENGTH = 3.0
CAR_WIDTH = 1.6
STOP_LINE_OFFSET = 5.0
MAX_SPEED = 0.8
ACCELERATION = 0.05
DECELERATION = 0.1
SAFE_DISTANCE = 4.0

class Car:
    def __init__(self, direction, lane_pos):
        self.direction = direction # 0:N, 1:E, 2:S, 3:W
        self.speed = random.uniform(0.5, 0.8) * MAX_SPEED
        self.max_speed = self.speed
        
        # Initial positions (Start far out)
        if direction == 0: # From North going South
            self.x = -2.0
            self.y = VISIBLE_RANGE + lane_pos * 5
        elif direction == 1: # From East going West
            self.x = VISIBLE_RANGE + lane_pos * 5
            self.y = 2.0
        elif direction == 2: # From South going North
            self.x = 2.0
            self.y = -VISIBLE_RANGE - lane_pos * 5
        elif direction == 3: # From West going East
            self.x = -VISIBLE_RANGE - lane_pos * 5
            self.y = -2.0
            
    def move(self, traffic_light_state, lead_car_dist):
        # Determine target speed
        target_speed = self.max_speed
        
        # Distance to stop line
        dist_to_stop = 1000
        if self.direction == 0: dist_to_stop = self.y - STOP_LINE_OFFSET
        elif self.direction == 1: dist_to_stop = self.x - STOP_LINE_OFFSET
        elif self.direction == 2: dist_to_stop = -STOP_LINE_OFFSET - self.y
        elif self.direction == 3: dist_to_stop = -STOP_LINE_OFFSET - self.x
        
        # Check Traffic Light
        can_go = False
        if traffic_light_state == 'GREEN' or (traffic_light_state == 'YELLOW' and dist_to_stop < 5) or dist_to_stop < 0:
            can_go = True
            
        if not can_go:
            if dist_to_stop > 0 and dist_to_stop < 20: # Approaching red light
                target_speed = 0
        
        # Check Lead Car
        if lead_car_dist < 1000:
            if lead_car_dist < SAFE_DISTANCE:
                target_speed = 0
            elif lead_car_dist < SAFE_DISTANCE * 3:
                target_speed = min(target_speed, lead_car_dist * 0.1)

        # Apply Physics
        if self.speed < target_speed:
            self.speed += ACCELERATION
        elif self.speed > target_speed:
            self.speed -= DECELERATION
            
        if self.speed < 0: self.speed = 0
        
        # Update Position
        if self.direction == 0: self.y -= self.speed
        elif self.direction == 1: self.x -= self.speed
        elif self.direction == 2: self.y += self.speed
        elif self.direction == 3: self.x += self.speed

class TrafficSimulation:
    def __init__(self):
        self.cars = []
        self.lights = [1, 0, 1, 0] # N, E, S, W (1:Green, 0:Red, 2:Yellow)
        self.timer = 0
        self.state = 'NS_GREEN' 

    def step(self, action=None):
        # Action: 0 = Keep current state, 1 = Switch phases (force switch if allowed)
        # For simplicity in this basic RL:
        # Action 0: Do nothing (let timer run)
        # Action 1: Force switch to next phase (reset timer)
        
        # RL Control Logic
        if action is not None:
             if action == 1:
                 # Force switch to next phase logic
                 if self.state == 'NS_GREEN': self.state = 'NS_YELLOW'; self.timer = 0
                 elif self.state == 'NS_YELLOW': self.state = 'EW_GREEN'; self.timer = 0
                 elif self.state == 'EW_GREEN': self.state = 'EW_YELLOW'; self.timer = 0
                 elif self.state == 'EW_YELLOW': self.state = 'NS_GREEN'; self.timer = 0
        
        self.timer += 1
        
        # Traffic Light Logic (Automatic transitions if no RL or simple timer expiry)
        # We keep the original logic as a fallback or for the "Keep" action behavior
        if self.state == 'NS_GREEN':
            self.lights = [1, 0, 1, 0]
            if self.timer > GREEN_DURATION:
                self.state = 'NS_YELLOW'
                self.timer = 0
        elif self.state == 'NS_YELLOW':
            self.lights = [2, 0, 2, 0]
            if self.timer > YELLOW_DURATION:
                self.state = 'EW_GREEN'
                self.timer = 0
        elif self.state == 'EW_GREEN':
            self.lights = [0, 1, 0, 1]
            if self.timer > GREEN_DURATION:
                self.state = 'EW_YELLOW'
                self.timer = 0
        elif self.state == 'EW_YELLOW':
            self.lights = [0, 2, 0, 2]
            if self.timer > YELLOW_DURATION:
                self.state = 'NS_GREEN'
                self.timer = 0
                
    def get_state(self):
        # State: [N_Queue, E_Queue, S_Queue, W_Queue, Light_Phase_Index]
        # Discretize queues for Q-table simplicity (0=Empty, 1=Low, 2=High)
        q_states = []
        for q in self.get_queue_lengths():
            if q == 0: val = 0
            elif q < 5: val = 1
            else: val = 2
            q_states.append(val)
        
        phase_map = {'NS_GREEN': 0, 'NS_YELLOW': 1, 'EW_GREEN': 2, 'EW_YELLOW': 3}
        return tuple(q_states + [phase_map[self.state]])

    def get_reward(self):
        # Reward = Negative of Total Queue Length (Minimize waiting cars)
        total_queue = sum(self.get_queue_lengths())
        return -total_queue

    def get_queue_lengths(self):
        # Calculate queue lengths dynamically from cars
        queues = [0, 0, 0, 0]
        for car in self.cars:
            # Simple logic: car is in queue if speed is low
            if car.speed < 0.1:
                queues[car.direction] += 1
        return queues

        # Spawn Cars
        for d in range(4):
            if random.random() < 0.05: # Spawn rate
                # Check if spawn area is clear
                clear = True
                for car in self.cars:
                    if car.direction == d:
                        dist = 0
                        if d == 0: dist = car.y - VISIBLE_RANGE
                        elif d == 1: dist = car.x - VISIBLE_RANGE
                        elif d == 2: dist = -VISIBLE_RANGE - car.y
                        elif d == 3: dist = -VISIBLE_RANGE - car.x
                        
                        if abs(dist) < 10: 
                            clear = False
                            break
                if clear:
                    self.cars.append(Car(d, 0))

        # Move Cars
        # Sort cars by position to easily find lead car logic per lane
        # This is a simple approximation
        self.cars.sort(key=lambda c: -c.y if c.direction == 0 else (-c.x if c.direction == 1 else (c.y if c.direction == 2 else c.x)))
        
        cars_to_keep = []
        for i, car in enumerate(self.cars):
            # Find distance to car ahead in same lane
            min_dist = 1000
            for other in self.cars:
                if car == other: continue
                if car.direction == other.direction:
                    dist = 1000
                    if car.direction == 0 and other.y < car.y: dist = car.y - other.y
                    elif car.direction == 1 and other.x < car.x: dist = car.x - other.x
                    elif car.direction == 2 and other.y > car.y: dist = other.y - car.y
                    elif car.direction == 3 and other.x > car.x: dist = other.x - car.x
                    
                    if dist > 0 and dist < min_dist:
                        min_dist = dist
            
            # Get Light State for this car
            light_color = 'RED'
            if self.lights[car.direction] == 1: light_color = 'GREEN'
            elif self.lights[car.direction] == 2: light_color = 'YELLOW'
            
            car.move(light_color, min_dist - CAR_LENGTH)
            
            # Despawn if out of bounds
            if abs(car.x) < VISIBLE_RANGE + 5 and abs(car.y) < VISIBLE_RANGE + 5:
                cars_to_keep.append(car)
                
        self.cars = cars_to_keep

def run_simulation():
    sim = TrafficSimulation()
    
    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patch.set_facecolor('#e6e6e6')
    
    # Static Background Gen
    buildings = []
    for _ in range(60):
        quad = random.randint(0, 3)
        if quad == 0: # TL
            x, y = random.uniform(-VISIBLE_RANGE, -6), random.uniform(6, VISIBLE_RANGE)
        elif quad == 1: # TR
            x, y = random.uniform(6, VISIBLE_RANGE), random.uniform(6, VISIBLE_RANGE)
        elif quad == 2: # BL
            x, y = random.uniform(-VISIBLE_RANGE, -6), random.uniform(-VISIBLE_RANGE, -6)
        else: # BR
            x, y = random.uniform(6, VISIBLE_RANGE), random.uniform(-VISIBLE_RANGE, -6)
        w, h = random.uniform(4, 10), random.uniform(4, 10)
        buildings.append((x, y, w, h))

    def update(frame):
        sim.step()
        ax.clear()
        
        ax.set_xlim(-VISIBLE_RANGE, VISIBLE_RANGE)
        ax.set_ylim(-VISIBLE_RANGE, VISIBLE_RANGE)
        ax.axis('off')
        
        # Background
        ax.add_patch(patches.Rectangle((-VISIBLE_RANGE, -VISIBLE_RANGE), 2*VISIBLE_RANGE, 2*VISIBLE_RANGE, color='#d0e0d0'))
        
        # Buildings
        for (bx, by, bw, bh) in buildings:
            color = random.choice(['#a0a0b0', '#b0b0c0'])
            ax.add_patch(patches.Rectangle((bx, by), bw, bh, color=color))
            ax.add_patch(patches.Rectangle((bx+0.5, by+0.5), bw-1, bh-1, color='#e0e0e0', alpha=0.3))

        # Roads
        ax.add_patch(patches.Rectangle((-4, -VISIBLE_RANGE), 8, 2*VISIBLE_RANGE, color='#333333'))
        ax.add_patch(patches.Rectangle((-VISIBLE_RANGE, -4), 2*VISIBLE_RANGE, 8, color='#333333'))
        ax.add_patch(patches.Rectangle((-4, -4), 8, 8, color='#444444'))
        
        # Markings
        ax.plot([0, 0], [-VISIBLE_RANGE, -5], color='white', linestyle='--', linewidth=1)
        ax.plot([0, 0], [5, VISIBLE_RANGE], color='white', linestyle='--', linewidth=1)
        ax.plot([-VISIBLE_RANGE, -5], [0, 0], color='white', linestyle='--', linewidth=1)
        ax.plot([5, VISIBLE_RANGE], [0, 0], color='white', linestyle='--', linewidth=1)
        
        # Stop Lines
        ax.plot([-4, 0], [5, 5], color='white', linewidth=3)
        ax.plot([5, 5], [0, 4], color='white', linewidth=3)
        ax.plot([0, 4], [-5, -5], color='white', linewidth=3)
        ax.plot([-5, -5], [-4, 0], color='white', linewidth=3)
        
        # Crosswalks
        for i in range(4):
            ax.add_patch(patches.Rectangle((-4 + i, 5.5), 0.5, 3, color='white'))
            ax.add_patch(patches.Rectangle((-4 + i, -8.5), 0.5, 3, color='white'))
            ax.add_patch(patches.Rectangle((5.5, -4 + i), 3, 0.5, color='white'))
            ax.add_patch(patches.Rectangle((-8.5, -4 + i), 3, 0.5, color='white'))

        # Cars
        for car in sim.cars:
            c = 'blue' if car.direction in [0, 2] else 'red'
            # Draw car centered at x,y
            if car.direction in [0, 2]: # Vertical
                ax.add_patch(patches.Rectangle((car.x - CAR_WIDTH/2, car.y - CAR_LENGTH/2), CAR_WIDTH, CAR_LENGTH, color=c))
            else: # Horizontal
                ax.add_patch(patches.Rectangle((car.x - CAR_LENGTH/2, car.y - CAR_WIDTH/2), CAR_LENGTH, CAR_WIDTH, color=c))

        # Lights
        colors = ['red', 'green', 'yellow']
        r = 0.8
        ax.add_patch(patches.Circle((-5, 5), r, color=colors[sim.lights[0]], zorder=10))
        ax.add_patch(patches.Circle((5, 5), r, color=colors[sim.lights[1]], zorder=10))
        ax.add_patch(patches.Circle((5, -5), r, color=colors[sim.lights[2]], zorder=10))
        ax.add_patch(patches.Circle((-5, -5), r, color=colors[sim.lights[3]], zorder=10))

        # Labels
        ax.text(0, -VISIBLE_RANGE + 2, "Atatürk Blv", color='white', fontsize=14, ha='center', fontweight='bold', bbox=dict(facecolor='black', alpha=0.6))
        ax.text(-VISIBLE_RANGE + 2, 0, "GMK Blv", color='white', fontsize=14, ha='center', rotation=90, fontweight='bold', bbox=dict(facecolor='black', alpha=0.6))
        ax.text(0, 0, VISIBLE_RANGE-5, f"ANKARA SMART TRAFFIC\nActive Cars: {len(sim.cars)}", color='white', fontsize=12, ha='center', bbox=dict(facecolor='black', alpha=0.6))

    ani = animation.FuncAnimation(fig, update, frames=500, interval=50, repeat=False)
    plt.show()

if __name__ == "__main__":
    run_simulation()
