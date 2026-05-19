#!/usr/bin/env python3

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


class ReactiveFollowGap(Node):
    """
    F1TENTH Lab 4: Follow the Gap.

    The node reads LiDAR data, removes unsafe ranges around the closest obstacle,
    finds the largest free-space gap, selects a stable target point inside that gap,
    and publishes Ackermann steering and speed commands.
    """

    def __init__(self):
        super().__init__('reactive_node')

        self.scan_topic = '/scan'
        self.drive_topic = '/drive'

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.lidar_callback,
            qos_profile_sensor_data
        )

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            self.drive_topic,
            10
        )

        self.max_lidar_range = 10.0
        self.bubble_radius = 80
        self.free_threshold = 1.2
        self.window_size = 5

        self.get_logger().info("Follow the Gap node initialized")

    def preprocess_lidar(self, ranges):
        ranges = np.array(ranges, dtype=np.float32)

        ranges[np.isnan(ranges)] = 0.0
        ranges[np.isinf(ranges)] = self.max_lidar_range
        ranges = np.clip(ranges, 0.0, self.max_lidar_range)

        # Smooth LiDAR data to reduce noisy steering.
        kernel = np.ones(self.window_size) / self.window_size
        ranges = np.convolve(ranges, kernel, mode='same')

        return ranges

    def find_max_gap(self, free_space_ranges):
        """
        Find the largest continuous segment where ranges are above the free threshold.
        """
        max_start = 0
        max_end = 0
        current_start = None

        for i, value in enumerate(free_space_ranges):
            if value > self.free_threshold:
                if current_start is None:
                    current_start = i
            else:
                if current_start is not None:
                    if i - current_start > max_end - max_start:
                        max_start = current_start
                        max_end = i - 1
                    current_start = None

        if current_start is not None:
            if len(free_space_ranges) - current_start > max_end - max_start:
                max_start = current_start
                max_end = len(free_space_ranges) - 1

        return max_start, max_end

    def find_best_point(self, start_i, end_i, ranges):
        """
        Choose the best point in the selected gap.

        Instead of blindly picking the midpoint, this selects the farthest point
        within the gap and averages around it for smoother steering.
        """
        if end_i <= start_i:
            return int((start_i + end_i) / 2)

        gap_ranges = ranges[start_i:end_i + 1]
        farthest_index = int(np.argmax(gap_ranges)) + start_i

        average_radius = 20
        left = max(start_i, farthest_index - average_radius)
        right = min(end_i, farthest_index + average_radius)

        return int((left + right) / 2)

    def publish_drive(self, steering_angle):
        drive_msg = AckermannDriveStamped()

        max_steering = 0.4189
        steering_angle = float(np.clip(steering_angle, -max_steering, max_steering))

        abs_angle = abs(steering_angle)

        if abs_angle > 0.35:
            speed = 1.0
        elif abs_angle > 0.20:
            speed = 1.8
        else:
            speed = 3.0

        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed

        self.drive_pub.publish(drive_msg)

    def lidar_callback(self, scan_msg):
        ranges = self.preprocess_lidar(scan_msg.ranges)

        # Use only the forward-facing LiDAR region for driving decisions.
        total_points = len(ranges)
        front_start = int(total_points * 0.25)
        front_end = int(total_points * 0.75)

        proc_ranges = np.zeros_like(ranges)
        proc_ranges[front_start:front_end] = ranges[front_start:front_end]

        # Find closest obstacle in the forward region.
        front_ranges = proc_ranges[front_start:front_end]
        if len(front_ranges) == 0:
            self.publish_drive(0.0)
            return

        closest_index = int(np.argmin(front_ranges)) + front_start

        # Create safety bubble around closest obstacle.
        bubble_start = max(front_start, closest_index - self.bubble_radius)
        bubble_end = min(front_end, closest_index + self.bubble_radius)
        proc_ranges[bubble_start:bubble_end] = 0.0

        gap_start, gap_end = self.find_max_gap(proc_ranges)
        best_point = self.find_best_point(gap_start, gap_end, proc_ranges)

        steering_angle = scan_msg.angle_min + best_point * scan_msg.angle_increment

        self.publish_drive(steering_angle)


def main(args=None):
    rclpy.init(args=args)

    node = ReactiveFollowGap()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
