import cv2
import mediapipe as mp
import time
import numpy as np
import random
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class PhysicsObject:
    def __init__(self, x, y, radius=10, color=(255, 0, 0)):
        self.pos = np.array([float(x), float(y)])
        self.vel = np.array([0.0, 0.0])
        self.radius = radius
        self.color = color
        self.mass = radius ** 2 # Mass proportional to area

    def update(self, gravity, dt, width, height):
        self.vel += gravity * dt
        self.pos += self.vel * dt
        
        # Side Boundary checks (bounce)
        if self.pos[0] - self.radius < 0:
            self.pos[0] = self.radius
            self.vel[0] *= -0.7
        if self.pos[0] + self.radius > width:
            self.pos[0] = width - self.radius
            self.vel[0] *= -0.7
            
        # Vertical boundaries: Remove if off-screen
        if self.pos[1] - self.radius > height:
            return False # Remove
        if self.pos[1] + self.radius < -200: 
            return False # Remove
            
        return True # Keep

    def draw(self, img):
        cv2.circle(img, (int(self.pos[0]), int(self.pos[1])), self.radius, self.color, cv2.FILLED)
        # Add a shine/highlight for 3D effect
        cv2.circle(img, (int(self.pos[0] - self.radius*0.3), int(self.pos[1] - self.radius*0.3)), int(self.radius*0.2), (255, 255, 255), cv2.FILLED)

def resolve_collisions(objects):
    # Simple O(N^2) collision detection
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            obj1 = objects[i]
            obj2 = objects[j]
            
            # Vector from 2 to 1
            delta = obj1.pos - obj2.pos
            dist_sq = np.dot(delta, delta)
            min_dist = obj1.radius + obj2.radius
            
            if dist_sq < min_dist * min_dist:
                dist = np.sqrt(dist_sq)
                if dist == 0: # Handle exact overlap
                    delta = np.array([1.0, 0.0])
                    dist = 1.0
                
                # Minimum Translation Vector to separate
                overlap = min_dist - dist
                n = delta / dist
                
                # Move apart (inverse mass weighting could be better, but equal split is stable enough)
                total_mass = obj1.mass + obj2.mass
                m1_ratio = obj2.mass / total_mass
                m2_ratio = obj1.mass / total_mass
                
                obj1.pos += n * overlap * m1_ratio
                obj2.pos -= n * overlap * m2_ratio
                
                # Elastic Collision Response
                # Relative velocity
                rel_vel = obj1.vel - obj2.vel
                vel_along_normal = np.dot(rel_vel, n)
                
                # Do not resolve if velocities are separating
                if vel_along_normal > 0:
                    continue
                
                # Restitution (bounciness) - slightly less than 1 for realism
                e = 0.9 
                
                j_val = -(1 + e) * vel_along_normal
                j_val /= (1 / obj1.mass + 1 / obj2.mass)
                
                impulse = j_val * n
                obj1.vel += impulse / obj1.mass
                obj2.vel -= impulse / obj2.mass

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    img_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    img_copy = np.copy(img_bgr)

    # Loop through each hand
    for hand_landmarks in hand_landmarks_list:
        h, w, c = img_copy.shape
        
        # Simple skeletal lines
        connections = [
            (0,1), (1,2), (2,3), (3,4), # Thumb
            (0,5), (5,6), (6,7), (7,8), # Index
            (0,9), (9,10), (10,11), (11,12), # Middle
            (0,13), (13,14), (14,15), (15,16), # Ring
            (0,17), (17,18), (18,19), (19,20), # Pinky
            (5,9), (9,13), (13,17) # Palm
        ]
        
        # Get pixel coordinates
        coords = []
        for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            coords.append((cx, cy))
            
        # Draw lines
        for start_idx, end_idx in connections:
            cv2.line(img_copy, coords[start_idx], coords[end_idx], (200, 200, 200), 2)
            
        # Draw points
        for cx, cy in coords:
            cv2.circle(img_copy, (cx, cy), 4, (255, 0, 0), cv2.FILLED)

    return img_copy

def main():
    model_path = 'hand_landmarker.task'

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    try:
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        
        landmarker = HandLandmarker.create_from_options(options)
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Make sure 'hand_landmarker.task' is in the same directory.")
        return

    # Open Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Physics Setup
    objects = []
    base_gravity = np.array([0.0, 500.0])
    
    prev_time = time.time()
    last_spawn_time = time.time()
    spawn_interval = 0.5 
    
    print("Anti-Gravity Simulator.")
    print("Control: Pinch = Normal Gravity, Open Hand = Reverse Gravity (-0.5g)")
    print("Press 'r' to clear objects, 'q' to exit.")
    
    start_time_ms = int(time.time() * 1000)

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        h, w, c = img.shape
        
        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        
        # Spawner Logic
        if curr_time - last_spawn_time > spawn_interval:
            last_spawn_time = curr_time
            r = random.randint(15, 60)
            x = random.randint(r, w - r)
            y = -r * 2 
            color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            objects.append(PhysicsObject(x, y, radius=r, color=color))
            if len(objects) > 30: # Reduce count for collision perf
                objects.pop(0)
        
        timestamp_ms = int(curr_time * 1000) - start_time_ms
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        gravity_modifier = 1.0 # Default: Normal gravity
        
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                thumb = hand_landmarks[4]
                index = hand_landmarks[8]
                t_x, t_y = int(thumb.x * w), int(thumb.y * h)
                i_x, i_y = int(index.x * w), int(index.y * h)
                
                cv2.line(img, (t_x, t_y), (i_x, i_y), (0, 255, 0), 3)
                cv2.circle(img, (t_x, t_y), 8, (255, 0, 255), -1)
                cv2.circle(img, (i_x, i_y), 8, (255, 0, 255), -1)
                
                # Interaction Logic
                dist = np.hypot(i_x - t_x, i_y - t_y)
                
                max_dist = 200.0 # Open hand
                min_dist = 20.0  # Pinch
                
                clamped = max(min_dist, min(dist, max_dist))
                normalized = (clamped - min_dist) / (max_dist - min_dist)
                
                # Mapping:
                # normalized 0 (Cancel/Pinch) -> 1.0
                # normalized 1 (Open) -> -0.5
                # Formula: 1.0 - (normalized * 1.5)
                # Check: 1 - 0 = 1.0. 1 - 1.5 = -0.5.
                
                gravity_modifier = 1.0 - (normalized * 1.5)
                
                status_text = "NORMAL GRAVITY" if gravity_modifier > 0 else "ANTI-GRAVITY"
                cv2.putText(img, f"{status_text} ({gravity_modifier:.2f}g)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        transformation_matrix = np.array([
            [1.0, 0.0],
            [0.0, gravity_modifier]
        ])
        effective_gravity = transformation_matrix @ base_gravity
        
        # Physics Update
        active_objects = []
        for obj in objects:
            keep = obj.update(effective_gravity, dt, w, h)
            if keep:
                active_objects.append(obj)
        objects = active_objects
        
        # Collisions
        resolve_collisions(objects)
        
        # Draw
        for obj in objects:
            obj.draw(img)

        cv2.imshow("Anti-Gravity Simulator", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
             objects.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
