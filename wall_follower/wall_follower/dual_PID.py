#!/usr/bin/env python3
import csv
import math
import time
from datetime import datetime
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


def clamp(value, low, high):
    return max(low, min(high, value))


def angle_difference(target, current):
    return math.atan2(math.sin(target - current), math.cos(target - current))


class DualPID(Node):
    def __init__(self):
        super().__init__('dual_PID')

        # Defaults match the original working controller.
        self.defaults = {
            'target': 0.3,
            'max_left': 0.4,
            'min_left': 0.15,
            'front_limit': 0.35,
            'front_rearm': 0.45,
            'front_left_limit': 0.22,
            'max_turn_speed': 0.8,
            'recovery_turn_speed': 0.3,
            'recovery_clear_distance': 0.4,
            'avoid_clear_distance': 0.28,
            'integral_limit': 0.2,
            'derivative_filter_time': 0.3,
            'pid_dt_min': 0.001,
            'pid_dt_max': 1.0,
            'corner_angle_deg': 90.0,
            'corner_kp': 1.5,
            'corner_max_turn_speed': 0.4,
            'corner_tolerance_deg': 2.0,
            'turn_timeout': 20.0,
            'recovery_timeout': 8.0,
            'sensor_timeout': 1.0,
            'watchdog_period': 0.1,
            'front_angle_deg': 0.0,
            'left_angle_deg': 90.0,
            'front_left_angle_deg': 45.0,
            'front_scan_half_width_deg': 12.0,
            'left_scan_half_width_deg': 3.0,
            'diagonal_scan_half_width_deg': 12.0,
            'log_directory': '~/wall_follower_logs',
            'kp': 0.9,
            'ki': 0.01,
            'kd': 0.65,
            'kp_linear': 0.3,
            'ki_linear': 0.04,
            'kd_linear': 0.02,
            'minimum_speed': 0.07,
            'nominal_speed': 0.09,
            'maximum_speed': 0.1,
            'speed_integral_limit': 0.1,
            'speed_derivative_filter_time': 0.25,
            'wall_slowdown_error': 0.15,
            'front_slowdown_start': 0.65,
            'wall_slowdown_fraction': 0.6,
            'front_min_speed_fraction': 0.4,
            'front_right_angle_deg': -45.0,
            'status_log_period': 0.5,
        }
        self.live_parameters = ('kp', 'ki', 'kd', 'kp_linear', 'ki_linear', 'kd_linear', 'minimum_speed', 'nominal_speed', 'maximum_speed')
        self.load_settings()
        self.yaw = None
        self.measured_speed = None
        self.last_scan = None
        self.last_odom = None
        self.state = 'FOLLOW'
        self.state_since = time.monotonic()
        self.turn_target = None
        self.corner_armed = True
        self.reset_pid()

        # One CSV per controller run: samples and settings in the same file.
        folder = Path(self.log_directory).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        run_name = datetime.now().strftime(
            'dual_PID_%Y%m%d_%H%M%S_%f'
        )
        self.csv_path = folder / f'{run_name}.csv'
        self.csv_file = self.csv_path.open(
            'x', newline='', encoding='utf-8'
        )
        self.writer = csv.writer(self.csv_file)
        self.log_started = time.monotonic()
        self.log_setting_names = ['target', 'max_left', 'min_left', 'front_limit', 'front_rearm', 'front_left_limit', 'max_turn_speed', 'recovery_turn_speed', 'wall_slowdown_error', 'front_slowdown_start', 'recovery_clear_distance', 'avoid_clear_distance', 'integral_limit', 'derivative_filter_time', 'pid_dt_min', 'pid_dt_max', 'corner_angle_deg', 'corner_kp', 'corner_max_turn_speed', 'corner_tolerance_deg', 'turn_timeout', 'recovery_timeout', 'sensor_timeout', 'watchdog_period', 'front_angle_deg', 'left_angle_deg', 'front_left_angle_deg', 'front_scan_half_width_deg', 'left_scan_half_width_deg', 'diagonal_scan_half_width_deg', 'log_directory', 'speed_integral_limit', 'speed_derivative_filter_time', 'wall_slowdown_fraction', 'front_min_speed_fraction', 'front_right_angle_deg', 'status_log_period']
        self.writer.writerow([
            'scan_time_s',
            'front_m',
            'left_m',
            'desired_m',
            'error_m',
            'linear_m_s',
            'angular_rad_s',
            'state',
            'front_left_m',
            'yaw_deg',
            'speed_reference_m_s',
            'measured_speed_m_s',
            'speed_error_m_s',
            'kp',
            'ki',
            'kd',
            'kp_linear',
            'ki_linear',
            'kd_linear',
            'minimum_speed',
            'nominal_speed',
            'maximum_speed',
            'controller',
            'wall_time_s',
            'elapsed_wall_s',
            'target',
            'max_left',
            'min_left',
            'front_limit',
            'front_rearm',
            'front_left_limit',
            'max_turn_speed',
            'recovery_turn_speed',
            'wall_slowdown_error',
            'front_slowdown_start',
            'recovery_clear_distance',
            'avoid_clear_distance',
            'integral_limit',
            'derivative_filter_time',
            'pid_dt_min',
            'pid_dt_max',
            'corner_angle_deg',
            'corner_kp',
            'corner_max_turn_speed',
            'corner_tolerance_deg',
            'turn_timeout',
            'recovery_timeout',
            'sensor_timeout',
            'watchdog_period',
            'front_angle_deg',
            'left_angle_deg',
            'front_left_angle_deg',
            'front_scan_half_width_deg',
            'left_scan_half_width_deg',
            'diagonal_scan_half_width_deg',
            'log_directory',
            'speed_integral_limit',
            'speed_derivative_filter_time',
            'wall_slowdown_fraction',
            'front_min_speed_fraction',
            'front_right_angle_deg',
            'status_log_period',
        ])
        self.csv_file.flush()

        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos_profile_sensor_data
        )
        self.create_timer(self.watchdog_period, self.watchdog)
        self.create_timer(self.status_log_period, self.print_pid_error)
        self.add_on_set_parameters_callback(self.update_parameters)
        self.get_logger().info('LEFT WALL: angular PID + linear speed PID')
        self.get_logger().info(
            f'LEFT wall target={self.target:.2f} m; nominal speed={self.nominal_speed:.3f} m/s'
        )
        self.get_logger().info(f'CSV: {self.csv_path}')
    def print_pid_error(self):
        if not hasattr(self, 'left_distance'):
            return

        if self.left_distance is None:
            return

        target = self.target
        error = self.left_distance - target

        self.get_logger().info(
            f"LEFT PID | Target: {target:.3f} m | "
            f"Actual: {self.left_distance:.3f} m | "
            f"Error: {error:+.4f} m"
        )

        


    def load_settings(self):
        # Geometry, limits and timer settings are loaded at startup.
        # Existing live-tunable gains/speeds remain adjustable while running.
        for name, default in self.defaults.items():
            self.declare_parameter(
                name, default,
                descriptor=ParameterDescriptor(
                    read_only=name not in self.live_parameters
                ),
            )
        settings = {
            name: self.get_parameter(name).value for name in self.defaults
        }
        reason = self.validate_settings(settings)
        if reason:
            raise ValueError(reason)
        for name, value in settings.items():
            setattr(self, name, value if isinstance(value, str) else float(value))

    def validate_settings(self, settings):
        for name, value in settings.items():
            if name == 'log_directory':
                if not isinstance(value, str) or not value.strip():
                    return 'log_directory must be a nonempty path'
                continue
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)):
                return f'{name} must be a finite number'
            if name.endswith('_angle_deg'):
                if not -180.0 <= value <= 180.0:
                    return f'{name} must be between -180 and 180 degrees'
            elif value < 0.0:
                return f'{name} must be nonnegative'
            if (name in ('kp', 'ki', 'kd', 'kp_linear', 'ki_linear', 'kd_linear',
                         'corner_kp') and value > 10.0):
                return f'{name} must be between 0 and 10'
            if name in ('speed', 'minimum_speed', 'nominal_speed', 'maximum_speed'):
                if value > 0.12:
                    return f'{name} must be between 0 and 0.12 m/s'
        positive = (
            'target', 'max_left', 'min_left', 'front_limit', 'front_rearm',
            'front_left_limit', 'max_turn_speed', 'recovery_turn_speed',
            'recovery_clear_distance', 'avoid_clear_distance',
            'pid_dt_min', 'pid_dt_max', 'corner_angle_deg', 'corner_kp',
            'corner_max_turn_speed', 'corner_tolerance_deg', 'turn_timeout',
            'recovery_timeout', 'sensor_timeout', 'watchdog_period',
            'front_scan_half_width_deg', 'left_scan_half_width_deg',
            'diagonal_scan_half_width_deg',
        )
        for name in positive:
            if settings[name] <= 0.0:
                return f'{name} must be positive'
        if not (settings['min_left'] < settings['avoid_clear_distance']
                <= settings['target'] <= settings['recovery_clear_distance']
                <= settings['max_left']):
            return ('Require min_left < avoid_clear_distance <= target '
                    '<= recovery_clear_distance <= max_left')
        if settings['front_rearm'] <= settings['front_limit']:
            return 'front_rearm must exceed front_limit'
        if settings['pid_dt_min'] > settings['pid_dt_max']:
            return 'pid_dt_min must not exceed pid_dt_max'
        if not (settings['corner_tolerance_deg'] < settings['corner_angle_deg']
                <= 180.0):
            return 'Require corner_tolerance_deg < corner_angle_deg <= 180'
        for name in ('front_scan_half_width_deg', 'left_scan_half_width_deg',
                     'diagonal_scan_half_width_deg'):
            if settings[name] > 180.0:
                return f'{name} must not exceed 180 degrees'
        if 'minimum_speed' in settings:
            if not (settings['minimum_speed'] <= settings['nominal_speed']
                    <= settings['maximum_speed']):
                return 'Require minimum_speed <= nominal_speed <= maximum_speed'
            if settings['front_slowdown_start'] <= settings['front_limit']:
                return 'front_slowdown_start must exceed front_limit'
            if settings['wall_slowdown_error'] <= 0.0:
                return 'wall_slowdown_error must be positive'
            if not 0.0 <= settings['wall_slowdown_fraction'] <= 1.0:
                return 'wall_slowdown_fraction must be between 0 and 1'
            if not 0.0 < settings['front_min_speed_fraction'] <= 1.0:
                return 'front_min_speed_fraction must be greater than 0 and at most 1'
            if settings['status_log_period'] <= 0.0:
                return 'status_log_period must be positive'
        return ''

    def update_parameters(self, params):
        pending = {name: getattr(self, name) for name in self.defaults}
        changes = {}
        for param in params:
            if param.name not in self.defaults:
                continue
            if param.name not in self.live_parameters:
                return SetParametersResult(
                    successful=False,
                    reason=f'{param.name} is a startup setting; edit YAML and restart',
                )
            pending[param.name] = param.value
            changes[param.name] = param.value
        reason = self.validate_settings(pending)
        if reason:
            return SetParametersResult(successful=False, reason=reason)
        if changes:
            for name, value in changes.items():
                setattr(self, name, float(value))
            self.reset_pid()
            self.get_logger().info(f'Parameters updated: {changes}')
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
        self.speed_error_sum = 0.0
        self.last_speed_error = None
        self.last_speed_stamp = None
        self.filtered_speed_d = 0.0

    def change_state(self, new_state):
        if new_state != self.state:
            self.state = new_state
            self.state_since = time.monotonic()
            self.reset_pid()
            self.get_logger().info(f'State: {new_state}')

    def fault(self, message):
        if self.state != 'FAULT':
            self.get_logger().error(message)
            self.change_state('FAULT')
        return self.send()

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        self.measured_speed = msg.twist.twist.linear.x
        self.last_odom = time.monotonic()

    def sensors_ready(self, now):
        return (
            self.last_scan is not None and self.last_odom is not None
            and self.yaw is not None and self.measured_speed is not None
            and math.isfinite(self.measured_speed)
            and now - self.last_scan <= self.sensor_timeout
            and now - self.last_odom <= self.sensor_timeout
        )

    def watchdog(self):
        if not self.sensors_ready(time.monotonic()):
            self.send()
            self.reset_pid()

    @staticmethod
    def distance(scan, angle_deg, width=3.0, nearest=False):
        readings = []
        for i, value in enumerate(scan.ranges):
            if not math.isfinite(value):
                continue
            if not scan.range_min <= value <= scan.range_max:
                continue
            angle = math.degrees(scan.angle_min + i * scan.angle_increment)
            # Correct wrapped angle; NO arbitrary angular offset.
            difference = (angle - angle_deg + 180.0) % 360.0 - 180.0
            if abs(difference) <= width:
                readings.append(value)
        if not readings:
            return None
        return min(readings) if nearest else sum(readings) / len(readings)

    def angular_pid(self, error, stamp):
        if self.last_stamp is not None and self.last_error is not None:
            dt = stamp - self.last_stamp
            if self.pid_dt_min <= dt <= self.pid_dt_max:
                self.error_sum = clamp(
                    self.error_sum + 0.5 * (error + self.last_error) * dt,
                    -self.integral_limit, self.integral_limit
                )
                derivative = (error - self.last_error) / dt
                alpha = dt / (self.derivative_filter_time + dt)
                self.filtered_d += alpha * (derivative - self.filtered_d)
            else:
                self.error_sum = 0.0
                self.filtered_d = 0.0
        self.last_error = error
        self.last_stamp = stamp
        command = (
            self.kp * error + self.ki * self.error_sum
            + self.kd * self.filtered_d
        )
        return clamp(command, -self.max_turn_speed, self.max_turn_speed)

    def reference_speed(self, wall_error, front):
        severity = clamp(abs(wall_error) / self.wall_slowdown_error, 0.0, 1.0)
        target_speed = max(
            self.minimum_speed,
            self.nominal_speed * (1.0 - self.wall_slowdown_fraction * severity)
        )
        if front < self.front_slowdown_start:
            front_fraction = clamp(
                (front - self.front_limit)
                / (self.front_slowdown_start - self.front_limit),
                0.0, 1.0
            )
            target_speed *= self.front_min_speed_fraction + (1.0 - self.front_min_speed_fraction) * front_fraction
        return clamp(target_speed, 0.0, self.nominal_speed)

    def linear_pid(self, speed_reference, stamp):
        error = speed_reference - self.measured_speed
        if self.last_speed_stamp is not None and self.last_speed_error is not None:
            dt = stamp - self.last_speed_stamp
            if self.pid_dt_min <= dt <= self.pid_dt_max:
                previous_integral = self.speed_error_sum
                self.speed_error_sum = clamp(
                    self.speed_error_sum
                    + 0.5 * (error + self.last_speed_error) * dt,
                    -self.speed_integral_limit, self.speed_integral_limit
                )
                derivative = (error - self.last_speed_error) / dt
                alpha = dt / (self.speed_derivative_filter_time + dt)
                self.filtered_speed_d += alpha * (
                    derivative - self.filtered_speed_d
                )
                raw = (
                    speed_reference + self.kp_linear * error
                    + self.ki_linear * self.speed_error_sum
                    + self.kd_linear * self.filtered_speed_d
                )
                if ((raw > self.maximum_speed and error > 0)
                        or (raw < 0.0 and error < 0)):
                    self.speed_error_sum = previous_integral
            else:
                self.speed_error_sum = 0.0
                self.filtered_speed_d = 0.0
        self.last_speed_error = error
        self.last_speed_stamp = stamp
        raw = (
            speed_reference + self.kp_linear * error
            + self.ki_linear * self.speed_error_sum
            + self.kd_linear * self.filtered_speed_d
        )
        return clamp(raw, 0.0, self.maximum_speed)

    def record(self, stamp, front, left, front_left, cmd, speed_ref=0.0):
        self.writer.writerow([
            stamp, front, left, self.target,
            None if left is None else left - self.target,
            cmd.linear.x, cmd.angular.z, self.state, front_left,
            None if self.yaw is None else math.degrees(self.yaw),
            speed_ref, self.measured_speed,
            None if self.measured_speed is None else speed_ref - self.measured_speed,
            self.kp, self.ki, self.kd,
            self.kp_linear, self.ki_linear, self.kd_linear,
            self.minimum_speed, self.nominal_speed, self.maximum_speed,
        ] + [
            'dual_PID', time.time(),
            time.monotonic() - self.log_started,
        ] + [getattr(self, name) for name in self.log_setting_names])
        self.csv_file.flush()

    def scan_callback(self, scan):
        now = time.monotonic()
        self.last_scan = now
        stamp = scan.header.stamp.sec + scan.header.stamp.nanosec * 1e-9
        front = self.distance(scan, self.front_angle_deg, width=self.front_scan_half_width_deg, nearest=True)
        left = self.distance(scan, self.left_angle_deg, width=self.left_scan_half_width_deg)
        self.left_distance = left 
        front_left = self.distance(scan, self.front_left_angle_deg, width=self.diagonal_scan_half_width_deg, nearest=True)
        front_right = self.distance(scan, self.front_right_angle_deg, width=self.diagonal_scan_half_width_deg, nearest=True)

        if (self.state == 'FAULT' or front is None or front_left is None or front_right is None
                or not self.sensors_ready(now)):
            cmd = self.send()
            self.reset_pid()
            self.record(stamp, front, left, front_left, cmd)
            return

        if front >= self.front_rearm:
            self.corner_armed = True

        if self.state == 'TURN_RIGHT':
            error = angle_difference(self.turn_target, self.yaw)
            if now - self.state_since > self.turn_timeout:
                cmd = self.fault('Right turn timeout')
            elif front_right < self.front_left_limit:
                # Check direction of the clockwise turn (front-right).
                cmd = self.fault('Obstacle too close during right turn')
            elif abs(math.degrees(error)) <= self.corner_tolerance_deg:
                self.turn_target = None
                self.change_state('FOLLOW')
                cmd = self.send()
            else:
                cmd = self.send(angular=clamp(self.corner_kp * error, -self.corner_max_turn_speed, self.corner_max_turn_speed))
            self.record(stamp, front, left, front_left, cmd)
            return

        if front <= self.front_limit:
            if self.corner_armed and front_right >= self.front_left_limit:
                self.corner_armed = False
                self.turn_target = self.yaw - math.radians(self.corner_angle_deg)
                self.change_state('TURN_RIGHT')
            cmd = self.send()
            self.record(stamp, front, left, front_left, cmd)
            return

        if front_left < self.front_left_limit:
            cmd = self.send()
            self.reset_pid()
            self.record(stamp, front, left, front_left, cmd)
            return

        if self.state == 'FOLLOW':
            if left is None or left > self.max_left:
                self.change_state('RECOVER_LEFT')
            elif left < self.min_left:
                self.change_state('AVOID_LEFT')
        elif self.state == 'RECOVER_LEFT':
            if left is not None and left < self.min_left:
                self.change_state('AVOID_LEFT')
            elif left is not None and left <= self.recovery_clear_distance:
                self.change_state('FOLLOW')
        elif self.state == 'AVOID_LEFT':
            if left is None:
                cmd = self.fault('Lost left LiDAR reading while too close')
                self.record(stamp, front, left, front_left, cmd)
                return
            if left >= self.avoid_clear_distance:
                self.change_state(
                    'RECOVER_LEFT' if left > self.max_left else 'FOLLOW'
                )

        if self.state in ('RECOVER_LEFT', 'AVOID_LEFT'):
            if now - self.state_since > self.recovery_timeout:
                cmd = self.fault('Left-wall recovery timeout')
            elif ((self.state == 'RECOVER_LEFT' and front_left < self.front_left_limit)
                  or (self.state == 'AVOID_LEFT' and front_right < self.front_left_limit)):
                cmd = self.fault('Insufficient clearance for wall recovery turn')
            else:
                turn = (
                    self.recovery_turn_speed
                    if self.state == 'RECOVER_LEFT'
                    else -self.recovery_turn_speed
                )
                cmd = self.send(angular=turn)
        elif self.state == 'FOLLOW' and left is not None:
            error = left - self.target
            angular = self.angular_pid(error, stamp)
            speed_ref = self.reference_speed(error, front)
            linear = self.linear_pid(speed_ref, stamp)
            cmd = self.send(linear=linear, angular=angular)
            self.record(stamp, front, left, front_left, cmd, speed_ref)
            return
        else:
            cmd = self.send()

        self.record(stamp, front, left, front_left, cmd)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = DualPID()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.send()
            time.sleep(0.1)
            node.csv_file.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

