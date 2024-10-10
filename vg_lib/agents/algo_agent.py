
from typing import Optional, Dict
from collections import deque

import cv2
import numpy as np

from aij_multiagent_rl.agents import BaseAgent


def cart2pol(coords):
    rho = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
    phi = np.arctan2(coords[:, 1], coords[:, 0])
    return np.vstack([rho, phi]).T

def pol_north2agent(pol, north_dir):
    if north_dir != 0:
        pol[:, 1] += (2 * np.pi - north_dir) #вектор в ориентации что куда смотрит агент = pi/2
    return pol

def pol_agent2north(pol, north_dir):
    pol[:, 1] += (np.pi / 2 - north_dir) #вектор в ориентации что север = pi/2
    return pol

def pol2cart(pol):
    x = pol[:, 0] * np.cos(pol[:, 1])
    y = pol[:, 0] * np.sin(pol[:, 1])
    return np.vstack([x, y]).T

def sum_pol(left, right):
    left_cart = pol2cart(left)
    right_cart = pol2cart(right)
    return cart2pol(left_cart + right_cart)

def l1_norm(coords):
    return np.abs(coords[:, 0]) + np.abs(coords[:, 1])

class DeterministicAgent(BaseAgent):
    def __init__(
        self,
        action_dim: int = 9,
        seed: Optional[int] = None
    ):
        self.agent_image_pos = np.array([30, 60])
        self.action_map = dict(zip([
            "FORWARD",
            "LEFT",
            "RIGHT",
            "BACKWARD",
            "PICKUP_RESOURCE",
            "PICKUP_TRASH",
            "DROP_RESOURCE",
            "DROP_TRASH",
            "NOOP"], range(action_dim)))
        self.move_step = 7
        self.red_hsv_range_low = np.array([100, 100, 100])
        self.red_hsv_range_up = np.array([130, 255, 255])
        self.reset_state()

    def reset_state(self) -> None:
        self.flag = True
        self.action_queue = deque()
        self.garbage_loc = None
        self.fabric_loc = None
        for _ in range(4):
            self.action_queue.append(self.action_map['LEFT']) # чтобы найти урну и фабрику
    
    def load(self, ckpt_dir: str) -> None:
        pass
    
    def find_fabric_and_garbage(self, image: np.ndarray):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red_image = cv2.inRange(hsv_image,
                                self.red_hsv_range_low,
                                self.red_hsv_range_up)
        contours, _ = cv2.findContours(red_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        square_coords = None
        circle_coords = None
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                coords = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]
                approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:
                    square_coords = np.array(coords)
                else:
                    circle_coords = np.array(coords)
        return square_coords, circle_coords

    def find_trash(self, image: np.ndarray) -> Optional[np.ndarray]:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        brown_image = cv2.inRange(hsv_image,
                                np.array([20, 90, 30]),
                                np.array([30, 100, 40]))
        contours, _ = cv2.findContours(brown_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        trash_centers = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                coord_x = int(M["m10"] / M["m00"])
                coord_y = int(M["m01"] / M["m00"])
                approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:
                    trash_centers.append = np.array([coord_x, coord_y])
        return np.array(trash_centers) if trash_centers else None

    def find_white_stars(self, image: np.ndarray) -> Optional[np.ndarray]:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray_image, 180, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        star_centers = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                coord_x = int(M["m10"] / M["m00"])
                coord_y = int(M["m01"] / M["m00"])
                star_centers.append([coord_x, coord_y])
        return np.array(star_centers) if star_centers else None
    
    def update_garbage_and_fabric_loc(self, observation):
        garbage_coords, fabric_coords = self.find_fabric_and_garbage(observation['image'])
        if (self.garbage_loc is None) and (garbage_coords is not None):
            polar_garbage = cart2pol(garbage_coords[None, ...])
            polar_garbage = pol_agent2north(polar_garbage, observation['proprio'][-2]) #['north_dir'])
            self.garbage_loc = polar_garbage
        if (self.fabric_loc is None) and (fabric_coords is not None):
            polar_fabric = cart2pol(fabric_coords[None, ...])
            polar_fabric = pol_agent2north(polar_fabric, observation['proprio'][-2]) #['north_dir'])
            self.fabric_loc = polar_fabric

    def get_vector_to_fabric(self, proprio):
        to_center = np.array([[proprio[3], proprio[4]]]) # rho, phi
        to_center = pol_agent2north(to_center, proprio[-2]) #['north_dir'])
        to_fabric = sum_pol(to_center, self.fabric_loc) #в центрации что в север = pi/2
        to_fabric = pol_north2agent(to_fabric, proprio[-2]) #['north_dir']) #в центрации агента(куда смотрим = pi/2)
        to_fabric = pol2cart(to_fabric)
        return to_fabric

    def get_vector_to_garbage(self, proprio):
        to_center = np.array([[proprio[3], proprio[4]]])
        to_center = pol_agent2north(to_center, proprio[-2]) #'north_dir'])
        to_garbage = sum_pol(to_center, self.garbage_loc) #в центрации что в север = pi/2
        to_garbage = pol_north2agent(to_garbage, proprio[-2]) #'north_dir']) #в центрации агента(куда смотрим = pi/2)
        to_garbage = pol2cart(to_garbage)
        return to_garbage

    def get_fastest_path_to_aim(self, found_items, aim, pickup_action, drop_action):
        item2aim = aim - found_items
        path_len = l1_norm(item2aim) + l1_norm(found_items)
        nearest_item_idx = np.argmin(path_len)
        nearest_item = found_items[nearest_item_idx]
        nearest_item2aim = item2aim[nearest_item_idx]

        path = []
        cur_rotate_agent = 0 # (0 * pi/2) radians
        up2item, cur_rotate_agent = self.path_to_go(
            nearest_item[0].item(),
            nearest_item[1].item(),
            cur_rotate_agent)
        path.extend(up2item)
        path.append(self.action_map[pickup_action])
        from_item2aim, cur_rotate_agent = self.path_to_go(
            nearest_item2aim[0].item(),
            nearest_item2aim[1].item(),
            cur_rotate_agent)
        path.extend(from_item2aim)
        path.append(self.action_map[drop_action])
        return path

    def get_actions_star2fabric(self, observation):
        found_stars = self.find_white_stars(observation['image'])
        if found_stars is None:
            return None
        
        to_fabric = self.get_vector_to_fabric(observation['proprio'])
        path = self.get_fastest_path_to_aim(
            found_stars,
            to_fabric,
            'PICKUP_RESOURCE',
            'DROP_RESOURCE')
        return path

    def go_to_fabric(self, observation):
        to_fabric = self.get_vector_to_fabric(observation['proprio'])[0]
        path, cur_rotate = self.path_to_go(to_fabric[0], to_fabric[1])
        return path

        
    def get_actions_trash2garbage(self, observation):
        found_trash = self.find_trash(observation['image'])
        if found_trash is None:
            return None
        
        to_garbage = self.get_vector_to_garbage(observation['proprio'])
        path = self.get_fastest_path_to_aim(
            found_trash,
            to_garbage,
            'PICKUP_TRASH',
            'DROP_TRASH')
        return path

    def dir2action(self, dir_times_half_pi: int):
        if dir_times_half_pi % 4 == 1:
            return 'LEFT'
        elif dir_times_half_pi % 4 == 2:
            return 'BACKWARD'
        elif dir_times_half_pi % 4 == 3:
            return 'RIGHT'
        return None

    def path_to_go(self, x: int, y: int, cur_rotate: int = 0):

        def rotate_and_forward(path: list, need_rotate: int, cur_rotate: int, num_forward: int):
            action = self.dir2action(need_rotate - cur_rotate)
            if action:
                path.append(self.action_map[action])
                path.extend([self.action_map['FORWARD']] * (num_forward - 1))
            else:
                path.extend([self.action_map['FORWARD']] * num_forward)
            return need_rotate

        x, y = int(x // 7), int(y // 7)
        path = []
        if y == 0:
            if x > 0:
                cur_rotate = rotate_and_forward(path, 3, cur_rotate, abs(x))
            elif x < 0:
                cur_rotate = rotate_and_forward(path, 1, cur_rotate, abs(x))
            return path, cur_rotate
        
        if y < 0:
            cur_rotate = rotate_and_forward(path, 2, cur_rotate, abs(y))
        if y > 0:
            cur_rotate = rotate_and_forward(path, 0, cur_rotate, abs(y))

        if x > 0:
            cur_rotate = rotate_and_forward(path, 3, cur_rotate, abs(y))
        if x < 0:
            cur_rotate = rotate_and_forward(path, 1, cur_rotate, abs(y))
        return path, cur_rotate

        
    def get_action(self, observation: Dict[str, np.ndarray]) -> int:
        if (self.garbage_loc is None or self.fabric_loc is None): #это только в начале/центре по дизайну
            self.update_garbage_and_fabric_loc(observation)
            return self.action_queue.popleft()

        if len(self.action_queue) > 0:
            return self.action_queue.popleft()
        
        if self.flag:
            path = self.go_to_fabric(observation)
            self.flag = False
            for act in path:
                self.action_queue.append(act)
            return self.action_queue.popleft()
        
        # path = self.get_actions_trash2garbage(observation)
        # if path:
        #     for act in path:
        #         self.action_queue.append(act)
        #     return self.action_queue.popleft()

        # path = self.get_actions_star2fabric(observation)
        # if path:
        #     for act in path:
        #         self.action_queue.append(act)
        #     return self.action_queue.popleft()
        
        return 2



class ConstAgent(BaseAgent):
    def __init__(
        self,
        action_dim: int = 9,
        const_action: int = 2,
        seed: Optional[int] = None
    ):
        self.action_map = dict(zip([
            "FORWARD",
            "LEFT",
            "RIGHT",
            "BACKWARD",
            "PICKUP_RESOURCE",
            "PICKUP_TRASH",
            "DROP_RESOURCE",
            "DROP_TRASH",
            "NOOP"], range(1, action_dim + 1)))
        self.const_action = const_action
        self.action_queue = deque()
        self.reset_state()

    def load(self, ckpt_dir: str) -> None:
        pass

    def get_action(self, observation: Dict[str, np.ndarray]) -> int:
        if len(self.action_queue) > 0:
            return self.action_queue.popleft()
        return self.const_action

    def reset_state(self) -> None:
        pass
        # self.action_queue = deque()