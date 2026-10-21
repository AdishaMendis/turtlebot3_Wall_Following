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

        # Controller settings.
        self.desired_distance = 0.30
        self.kp = 1.0
        self.kd = 0.5
        self.forward_speed = 0.07
        self.front_stop_distance = 0.35

        # Heading and corner-turn state.
        self.yaw = None
        self.wall_heading = None
        self.turn_target = None

        # Derivative state.
        self.previous_error = None
        self.previous_scan_time = None
        self.filtered_derivative = 0.0

        # Sensor arrival times for the watchdog.
        self.last_scan_time = None
        self.last_odom_time = None

        # Create a separate CSV file for each run.
        log_folder = Path.home() / 'wall_follower_logs'
        log_folder.mkdir(parents=True, exist_ok=True)

        filename = datetime.now().strftime(
            'pd_run_%Y%m%d_%H%M%S_%f.csv'
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

        self.get_logger().info(f'CSV log: {self.log_path}')
        self.get_logger().info('Waiting for laser data...')

    def stop_robot(self):
        self.publisher.publish(Twist())

        # Reset derivative history after a stop or completed turn.
        self.previous_error = None
        self.previous_scan_time = None
        self.filtered_derivative = 0.0

    def odom_callback(self, msg):
        self.last_odom_time = time.monotonic()
        q = msg.pose.pose.orientation

        sin_yaw = 2.0 * (q.w * q.z + q.x * q.y)
        cos_yaw = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(sin_yaw, cos_yaw)

        # Start aligned with the first wall when launching this node.
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
            or now - self.last_scan_time > 1.0
        )
        odom_missing = (
            self.last_odom_time is None
            or now - self.last_odom_time > 1.0
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
            or time.monotonic() - self.last_odom_time > 1.0
        ):
            self.stop_robot()
            return

        # Advance the fixed heading target at each corner.
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

            if abs(turn_error) <= math.radians(2):
                self.stop_robot()
                self.turn_target = None
                self.get_logger().info(
                    'Turn complete: resuming wall following'
                )
            else:
                command = Twist()
                command.angular.z = max(
                    -0.4, min(0.4, 1.5 * turn_error)
                )
                self.publisher.publish(command)

            # Do not record corner turns as wall-following samples.
            return

        error = right - self.desired_distance

        scan_time = (
            msg.header.stamp.sec
            + msg.header.stamp.nanosec * 1e-9
        )

        if self.previous_scan_time is not None:
            dt = scan_time - self.previous_scan_time

            if 0.001 <= dt <= 1.0:
                derivative = (error - self.previous_error) / dt

                # Low-pass filter the derivative.
                alpha = dt / (0.3 + dt)
                self.filtered_derivative += alpha * (
                    derivative - self.filtered_derivative
                )
            else:
                self.filtered_derivative = 0.0

        self.previous_error = error
        self.previous_scan_time = scan_time

        angular_velocity = -(
            self.kp * error
            + self.kd * self.filtered_derivative
        )
        angular_velocity = max(-0.5, min(0.5, angular_velocity))

        command = Twist()
        command.linear.x = self.forward_speed
        command.angular.z = angular_velocity
        self.publisher.publish(command)

        # Record every valid wall-following sample.
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
    node = ScanReader()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop_robot()
            time.sleep(0.1)
        finally:
            node.log_file.close()
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
