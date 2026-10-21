import csv
import math
import time
from datetime import datetime
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan


class ScanReader(Node):
    def __init__(self):
        super().__init__('scan_reader')

        # Declare settings that can be overridden at startup.
        defaults = {
            'desired_distance': 0.30,
            'kp': 1.0,
            'ki': 0.01,
            'kd': 0.5,
            'forward_speed': 0.07,
            'front_stop_distance': 0.35,
            'integral_limit': 0.20,
            'derivative_filter_time': 0.30,
            'max_angular_speed': 0.50,
            'corner_angular_speed': 0.40,
            'corner_kp': 1.5,
            'corner_tolerance_deg': 2.0,
            'sensor_timeout': 1.0,
        }

        for name, default in defaults.items():
            self.declare_parameter(name, default)

        # Read parameters into controller variables.
        settings = {
            name: float(self.get_parameter(name).value)
            for name in defaults
        }

        for name, value in settings.items():
            if not math.isfinite(value):
                raise ValueError(f'{name} must be finite')

        positive_settings = [
            'desired_distance',
            'front_stop_distance',
            'derivative_filter_time',
            'max_angular_speed',
            'corner_angular_speed',
            'corner_kp',
            'corner_tolerance_deg',
            'sensor_timeout',
        ]

        for name in positive_settings:
            if settings[name] <= 0.0:
                raise ValueError(f'{name} must be greater than zero')

        for name in ['kp', 'ki', 'kd', 'forward_speed', 'integral_limit']:
            if settings[name] < 0.0:
                raise ValueError(f'{name} must not be negative')

        if settings['corner_tolerance_deg'] >= 90.0:
            raise ValueError('corner_tolerance_deg must be below 90')

        for name, value in settings.items():
            setattr(self, name, value)

        # PID state.
        self.integral_error = 0.0
        self.previous_error = None
        self.previous_scan_time = None
        self.filtered_derivative = 0.0

        # Heading and corner-turn state.
        self.yaw = None
        self.wall_heading = None
        self.turn_target = None

        # Sensor arrival times for the watchdog.
        self.last_scan_time = None
        self.last_odom_time = None

        # Create one CSV file per run.
        log_folder = Path.home() / 'wall_follower_logs'
        log_folder.mkdir(parents=True, exist_ok=True)

        filename = datetime.now().strftime(
            'pid_run_%Y%m%d_%H%M%S_%f.csv'
        )
        self.log_path = log_folder / filename
        self.log_file = self.log_path.open('w', newline='')
        self.csv_writer = csv.writer(self.log_file)

        self.csv_writer.writerow([
            'scan_time_s',
            'front_m',
            'right_m',
            'desired_m',
            'error_m',
            'linear_m_s',
            'angular_rad_s',
        ])
        self.log_file.flush()

        # Save the startup settings beside the CSV.
        settings_path = self.log_path.with_suffix('.settings.txt')
        settings_path.write_text(
            '\n'.join(
                f'{name}={value}'
                for name, value in settings.items()
            ) + '\n',
            encoding='utf-8',
        )

        self.publisher = self.create_publisher(
            Twist, '/cmd_vel', 10
        )

        self.odom_subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos_profile_sensor_data,
        )

        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data,
        )

        self.watchdog = self.create_timer(0.1, self.check_scan)

        self.get_logger().info(
            f'PID gains: Kp={self.kp}, Ki={self.ki}, Kd={self.kd}'
        )
        self.get_logger().info(
            f'Speed: {self.forward_speed} m/s | '
            f'Target distance: {self.desired_distance} m'
        )
        self.get_logger().info(f'CSV log: {self.log_path}')
        self.get_logger().info('Waiting for laser data...')

    def stop_robot(self):
        self.publisher.publish(Twist())

        # Reset PID history after stopping or completing a corner.
        self.integral_error = 0.0
        self.previous_error = None
        self.previous_scan_time = None
        self.filtered_derivative = 0.0

    def odom_callback(self, msg):
        self.last_odom_time = time.monotonic()
        q = msg.pose.pose.orientation

        sin_yaw = 2.0 * (q.w * q.z + q.x * q.y)
        cos_yaw = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(sin_yaw, cos_yaw)

        # Start with the robot aligned with the first wall.
        if self.wall_heading is None:
            self.wall_heading = self.yaw

        self.get_logger().info(
            f'Heading: {math.degrees(self.yaw):+.1f} degrees',
            throttle_duration_sec=2.0,
        )

    def check_scan(self):
        now = time.monotonic()

        scan_missing = (
            self.last_scan_time is None
            or now - self.last_scan_time > self.sensor_timeout
        )
        odom_missing = (
            self.last_odom_time is None
            or now - self.last_odom_time > self.sensor_timeout
        )

        if scan_missing or odom_missing:
            self.stop_robot()

    def distance_at_angle(self, msg, target_degrees):
        readings = []

        for i, distance in enumerate(msg.ranges):
            angle = math.degrees(
                msg.angle_min + i * msg.angle_increment
            )
            difference = (angle - target_degrees + 180) % 360 - 180

            if abs(difference) <= 3:
                if math.isfinite(distance):
                    if msg.range_min <= distance <= msg.range_max:
                        readings.append(distance)

        if not readings:
            return None

        return sum(readings) / len(readings)

    def scan_callback(self, msg):
        self.last_scan_time = time.monotonic()

        front = self.distance_at_angle(msg, 0)
        right = self.distance_at_angle(msg, -90)

        if front is None or right is None:
            self.stop_robot()
            self.get_logger().warning(
                'Missing distance: stopping',
                throttle_duration_sec=2.0,
            )
            return

        if (
            self.yaw is None
            or self.last_odom_time is None
            or time.monotonic() - self.last_odom_time
            > self.sensor_timeout
        ):
            self.stop_robot()
            return

        # Advance the fixed heading target at each square-arena corner.
        if self.turn_target is None:
            if front <= self.front_stop_distance:
                self.wall_heading += math.pi / 2
                self.turn_target = self.wall_heading
                self.get_logger().info('Starting left turn')

        if self.turn_target is not None:
            difference = self.turn_target - self.yaw
            turn_error = math.atan2(
                math.sin(difference),
                math.cos(difference),
            )

            if abs(turn_error) <= math.radians(
                self.corner_tolerance_deg
            ):
                self.stop_robot()
                self.turn_target = None
                self.get_logger().info(
                    'Turn complete: resuming wall following'
                )
            else:
                command = Twist()
                command.angular.z = max(
                    -self.corner_angular_speed,
                    min(
                        self.corner_angular_speed,
                        self.corner_kp * turn_error,
                    ),
                )
                self.publisher.publish(command)

            # Exclude corner turns from the wall-following CSV.
            return

        error = right - self.desired_distance

        scan_time = (
            msg.header.stamp.sec
            + msg.header.stamp.nanosec * 1e-9
        )

        if self.previous_scan_time is not None:
            dt = scan_time - self.previous_scan_time

            if 0.001 <= dt <= 1.0:
                # Trapezoidal integration with bounded accumulation.
                self.integral_error += (
                    0.5 * (error + self.previous_error) * dt
                )
                self.integral_error = max(
                    -self.integral_limit,
                    min(self.integral_limit, self.integral_error),
                )

                derivative = (error - self.previous_error) / dt

                alpha = dt / (
                    self.derivative_filter_time + dt
                )
                self.filtered_derivative += alpha * (
                    derivative - self.filtered_derivative
                )
            else:
                self.integral_error = 0.0
                self.filtered_derivative = 0.0

        self.previous_error = error
        self.previous_scan_time = scan_time

        angular_velocity = -(
            self.kp * error
            + self.ki * self.integral_error
            + self.kd * self.filtered_derivative
        )
        angular_velocity = max(
            -self.max_angular_speed,
            min(self.max_angular_speed, angular_velocity),
        )

        command = Twist()
        command.linear.x = self.forward_speed
        command.angular.z = angular_velocity
        self.publisher.publish(command)

        self.csv_writer.writerow([
            scan_time,
            front,
            right,
            self.desired_distance,
            error,
            command.linear.x,
            command.angular.z,
        ])
        self.log_file.flush()

        self.get_logger().info(
            f'Front: {front:.3f} m | '
            f'Right: {right:.3f} m | '
            f'Error: {error:+.3f} m | '
            f'Turn: {angular_velocity:+.3f} rad/s',
            throttle_duration_sec=2.0,
        )


def main(args=None):
    rclpy.init(
        args=args,
        signal_handler_options=SignalHandlerOptions.NO,
    )
    node = None

    try:
        node = ScanReader()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.stop_robot()
                time.sleep(0.1)
            finally:
                node.log_file.close()
                node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
