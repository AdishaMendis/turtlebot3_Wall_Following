#!/usr/bin/env python3

import csv
import math
import time
from datetime import datetime
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan


def clamp(value, low, high):
    return max(low, min(high, value))


def angle_difference(target, current):
    difference = target - current
    return math.atan2(math.sin(difference), math.cos(difference))


class SinglePID(Node):
    def __init__(self):
        super().__init__('single_PID')

        # Distance limits in metres.
        self.target = 0.30
        self.max_left = 0.50
        self.min_left = 0.20
        self.front_limit = 0.35
        self.front_rearm = 0.45
        self.front_left_limit = 0.22

        # Movement settings.
        self.speed = 0.08
        self.max_turn_speed = 0.50
        self.recovery_turn_speed = 0.22

        # PID gains: configurable at startup and while running.
        self.declare_parameter('kp', 1.0)
        self.declare_parameter('ki', 0.01)
        self.declare_parameter('kd', 0.50)

        self.kp = float(self.get_parameter('kp').value)
        self.ki = float(self.get_parameter('ki').value)
        self.kd = float(self.get_parameter('kd').value)

        for name in ('kp', 'ki', 'kd'):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 10.0:
                raise ValueError(
                    f'{name} must be between 0.0 and 10.0'
                )

        self.reset_pid()

        self.add_on_set_parameters_callback(
            self.update_pid_parameters
        )

        # Sensor and movement state.
        self.yaw = None
        self.last_scan = None
        self.last_odom = None

        self.state = 'FOLLOW'
        self.state_since = time.monotonic()
        self.turn_target = None
        self.corner_armed = True

        # One CSV per controller run: samples and settings in the same file.
        folder = Path.home() / 'wall_follower_logs'
        folder.mkdir(parents=True, exist_ok=True)
        run_name = datetime.now().strftime(
            'single_PID_%Y%m%d_%H%M%S_%f'
        )
        self.csv_path = folder / f'{run_name}.csv'
        self.csv_file = self.csv_path.open(
            'x', newline='', encoding='utf-8'
        )
        self.writer = csv.writer(self.csv_file)
        self.log_started = time.monotonic()
        self.log_setting_names = ['speed', 'target', 'max_left', 'min_left', 'front_limit', 'front_rearm', 'front_left_limit', 'max_turn_speed', 'recovery_turn_speed']
        self.writer.writerow(['scan_time_s', 'front_m', 'left_m', 'desired_m', 'error_m', 'linear_m_s', 'angular_rad_s', 'state', 'front_left_m', 'yaw_deg', 'kp', 'ki', 'kd', 'controller', 'wall_time_s', 'elapsed_wall_s', 'speed', 'target', 'max_left', 'min_left', 'front_limit', 'front_rearm', 'front_left_limit', 'max_turn_speed', 'recovery_turn_speed'])
        self.csv_file.flush()

        self.pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10,
        )

        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data,
        )

        self.odom_subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos_profile_sensor_data,
        )

        self.watchdog_timer = self.create_timer(
            0.1,
            self.watchdog,
        )

        self.get_logger().info(
            'LEFT wall follower: target 0.30 m'
        )
        self.get_logger().info(
            f'PID: Kp={self.kp}, Ki={self.ki}, Kd={self.kd}'
        )
        self.get_logger().info(
            f'CSV file: {self.csv_path}'
        )

    def update_pid_parameters(self, params):
        changes = {}

        for param in params:
            if param.name not in ('kp', 'ki', 'kd'):
                continue

            value = param.value

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 10.0
            ):
                return SetParametersResult(
                    successful=False,
                    reason=(
                        f'{param.name} must be a finite '
                        'number between 0 and 10'
                    ),
                )

            changes[param.name] = float(value)

        if changes:
            for name, value in changes.items():
                setattr(self, name, value)

            self.reset_pid()

            self.get_logger().info(
                f'PID updated: Kp={self.kp:.3f}, '
                f'Ki={self.ki:.3f}, Kd={self.kd:.3f}'
            )

        return SetParametersResult(successful=True)

    def send(self, linear=0.0, angular=0.0):
        command = Twist()
        command.linear.x = float(linear)
        command.angular.z = float(angular)

        self.pub.publish(command)
        return command

    def reset_pid(self):
        self.error_sum = 0.0
        self.last_error = None
        self.last_stamp = None
        self.filtered_d = 0.0

    def change_state(self, new_state):
        if self.state != new_state:
            self.state = new_state
            self.state_since = time.monotonic()
            self.reset_pid()

            self.get_logger().info(
                f'State: {new_state}'
            )

    def fault(self, message):
        if self.state != 'FAULT':
            self.get_logger().error(message)
            self.change_state('FAULT')

        return self.send()

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation

        components = (q.x, q.y, q.z, q.w)

        if not all(math.isfinite(value) for value in components):
            return

        norm = math.sqrt(sum(value * value for value in components))

        if norm < 1e-9:
            return

        x, y, z, w = (
            value / norm for value in components
        )

        self.yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

        self.last_odom = time.monotonic()

    def sensors_ready(self, now):
        return (
            self.last_scan is not None
            and self.last_odom is not None
            and self.yaw is not None
            and now - self.last_scan <= 1.0
            and now - self.last_odom <= 1.0
        )

    def watchdog(self):
        if not self.sensors_ready(time.monotonic()):
            self.send()
            self.reset_pid()

    def distance(
        self,
        scan,
        angle_deg,
        width=3.0,
        nearest=False,
    ):
        values = []

        for i, reading in enumerate(scan.ranges):
            if not math.isfinite(reading):
                continue

            if not scan.range_min <= reading <= scan.range_max:
                continue

            angle = math.degrees(
                scan.angle_min + i * scan.angle_increment
            )

            difference = (
                angle - angle_deg + 180.0
            ) % 360.0 - 180.0

            if abs(difference) <= width:
                values.append(reading)

        if not values:
            return None

        if nearest:
            return min(values)

        return sum(values) / len(values)

    def pid(self, error, stamp):
        if self.last_stamp is not None and self.last_error is not None:
            dt = stamp - self.last_stamp

            if 0.001 <= dt <= 1.0:
                self.error_sum += (
                    0.5 * (error + self.last_error) * dt
                )

                self.error_sum = clamp(
                    self.error_sum,
                    -0.20,
                    0.20,
                )

                derivative = (
                    error - self.last_error
                ) / dt

                # Low-pass filter for the derivative.
                alpha = dt / (0.30 + dt)

                self.filtered_d += alpha * (
                    derivative - self.filtered_d
                )
            else:
                self.error_sum = 0.0
                self.filtered_d = 0.0

        self.last_error = error
        self.last_stamp = stamp

        # Left-wall following:
        # Too far from left wall -> positive angular velocity.
        # Too close to left wall -> negative angular velocity.
        angular = (
            self.kp * error
            + self.ki * self.error_sum
            + self.kd * self.filtered_d
        )

        return clamp(
            angular,
            -self.max_turn_speed,
            self.max_turn_speed,
        )

    def record(self, stamp, front, left, front_left, command):
        self.writer.writerow([
            stamp,
            front,
            left,
            self.target,
            None if left is None else left - self.target,
            command.linear.x,
            command.angular.z,
            self.state,
            front_left,
            None if self.yaw is None else math.degrees(self.yaw),
            self.kp,
            self.ki,
            self.kd,
        ] + [
            'single_PID', time.time(),
            time.monotonic() - self.log_started,
        ] + [getattr(self, name) for name in self.log_setting_names])

        self.csv_file.flush()

    def scan_callback(self, scan):
        now = time.monotonic()
        self.last_scan = now

        stamp = (
            scan.header.stamp.sec
            + scan.header.stamp.nanosec * 1e-9
        )

        # LiDAR directions:
        # Front = 0 degrees.
        # Left = +90 degrees.
        # Front-left = +45 degrees.
        front = self.distance(
            scan,
            0,
            width=12,
            nearest=True,
        )
        left = self.distance(
            scan,
            90,
            width=3,
        )
        front_left = self.distance(
            scan,
            45,
            width=12,
            nearest=True,
        )

        # Stop on missing required data or a latched fault.
        if (
            self.state == 'FAULT'
            or front is None
            or front_left is None
            or not self.sensors_ready(now)
        ):
            command = self.send()
            self.reset_pid()
            self.record(
                stamp, front, left, front_left, command
            )
            return

        # Rearm corner detection once the front clears.
        if front >= self.front_rearm:
            self.corner_armed = True

        # Complete a 90-degree right turn using odometry.
        if self.state == 'TURN_RIGHT':
            error = angle_difference(
                self.turn_target,
                self.yaw,
            )

            if now - self.state_since > 20.0:
                command = self.fault(
                    'Right turn timeout'
                )

            elif (
                front_left < self.front_left_limit
                or (
                    left is not None
                    and left < self.min_left
                )
            ):
                command = self.fault(
                    'Obstacle too close during right turn'
                )

            elif abs(math.degrees(error)) <= 2.0:
                self.turn_target = None
                self.change_state('FOLLOW')
                command = self.send()

            else:
                command = self.send(
                    angular=clamp(
                        1.5 * error,
                        -0.40,
                        0.40,
                    )
                )

            self.record(
                stamp, front, left, front_left, command
            )
            return

        # Front wall: start one 90-degree right turn.
        if front <= self.front_limit:
            if (
                self.corner_armed
                and front_left >= self.front_left_limit
            ):
                self.corner_armed = False
                self.turn_target = self.yaw - math.pi / 2.0
                self.change_state('TURN_RIGHT')

            command = self.send()
            self.record(
                stamp, front, left, front_left, command
            )
            return

        # Stop for an obstruction near the front-left.
        if front_left < self.front_left_limit:
            command = self.send()
            self.reset_pid()
            self.record(
                stamp, front, left, front_left, command
            )
            return

        # Select normal following or recovery.
        if self.state == 'FOLLOW':
            if left is None or left > self.max_left:
                self.change_state('RECOVER_LEFT')

            elif left < self.min_left:
                self.change_state('AVOID_LEFT')

        elif self.state == 'RECOVER_LEFT':
            if left is not None and left < self.min_left:
                self.change_state('AVOID_LEFT')

            elif left is not None and left <= 0.45:
                self.change_state('FOLLOW')

        elif self.state == 'AVOID_LEFT':
            if left is None:
                command = self.fault(
                    'Lost left LiDAR reading while too close'
                )
                self.record(
                    stamp, front, left, front_left, command
                )
                return

            if left >= 0.25:
                self.change_state(
                    'RECOVER_LEFT'
                    if left > self.max_left
                    else 'FOLLOW'
                )

        if self.state in ('RECOVER_LEFT', 'AVOID_LEFT'):
            if now - self.state_since > 8.0:
                command = self.fault(
                    'Left-wall recovery timeout'
                )

            elif (
                self.state == 'RECOVER_LEFT'
                and front < self.front_left_limit
            ):
                command = self.send()

            else:
                # Recover: rotate left toward the wall.
                # Avoid: rotate right away from the wall.
                turn = (
                    self.recovery_turn_speed
                    if self.state == 'RECOVER_LEFT'
                    else -self.recovery_turn_speed
                )

                command = self.send(angular=turn)

        elif self.state == 'FOLLOW' and left is not None:
            error = left - self.target

            command = self.send(
                linear=self.speed,
                angular=self.pid(error, stamp),
            )

        else:
            command = self.send()

        self.record(
            stamp, front, left, front_left, command
        )


def main(args=None):
    # Handle Ctrl+C while ROS is still available to publish a stop.
    rclpy.init(
        args=args,
        signal_handler_options=SignalHandlerOptions.NO,
    )

    node = None

    try:
        node = SinglePID()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        try:
            if node is not None:
                try:
                    node.send()
                    time.sleep(0.1)
                finally:
                    node.csv_file.close()
                    node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    main()
