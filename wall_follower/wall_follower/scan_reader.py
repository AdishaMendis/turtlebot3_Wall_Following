import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class ScanReader(Node):
    def __init__(self):
        super().__init__('scan_reader')

        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info('Waiting for laser data...')

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
        front = self.distance_at_angle(msg, 0)
        right = self.distance_at_angle(msg, -90)

        front_text = 'unavailable' if front is None else f'{front:.3f} m'
        right_text = 'unavailable' if right is None else f'{right:.3f} m'

        self.get_logger().info(
            f'Front: {front_text} | Right: {right_text}',
            throttle_duration_sec=2.0,
        )


def main(args=None):
    rclpy.init(args=args)
    node = ScanReader()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
