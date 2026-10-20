import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan


class ScanReader(Node):
    def __init__(self):
        super().__init__('scan_reader')

        self.desired_distance = 0.30
        self.kp = 1.0
        self.forward_speed = 0.05
        self.front_stop_distance = 0.35

        self.last_scan_time = None
        self.front_stop = False

        self.publisher = self.create_publisher(
            Twist, '/cmd_vel', 10
        )

        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data,
        )

        # Stop if laser messages stop arriving.
        self.watchdog = self.create_timer(0.1, self.check_scan)

        self.get_logger().info('Waiting for laser data...')

    def stop_robot(self):
        self.publisher.publish(Twist())

    def check_scan(self):
        if self.last_scan_time is None:
            self.stop_robot()
        elif time.monotonic() - self.last_scan_time > 1.0:
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

        # Once triggered, stay stopped until this node is restarted.
        if front <= self.front_stop_distance:
            self.front_stop = True

        if self.front_stop:
            self.stop_robot()
            self.get_logger().info(
                f'Front wall reached: stopped at {front:.3f} m',
                throttle_duration_sec=2.0,
            )
            return

        error = right - self.desired_distance
        angular_velocity = -self.kp * error
        angular_velocity = max(-0.5, min(0.5, angular_velocity))

        command = Twist()
        command.linear.x = self.forward_speed
        command.angular.z = angular_velocity
        self.publisher.publish(command)

        self.get_logger().info(
            f'Front: {front:.3f} m | '
            f'Right: {right:.3f} m | '
            f'Error: {error:+.3f} m | '
            f'Turn: {angular_velocity:+.3f} rad/s',
            throttle_duration_sec=2.0,
        )


def main(args=None):
    # Keep ROS active while handling Ctrl+C and sending the stop command.
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
        node.stop_robot()
        time.sleep(0.1)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
