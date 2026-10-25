#!/usr/bin/env python3
import csv
import json
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


class SinglePID(Node):
    """Left-wall follower: ONE angular PID (+ wall-angle term), speed from a
    simple slowdown rule (no linear PID). Same structure as dual_PID."""

    def __init__(self):
        super().__init__('single_PID')

        self.defaults = {
            'target': 0.3,
            'max_left': 0.4,
            'min_left': 0.15,
            'front_limit': 0.35,
            'front_left_limit': 0.22,
            'max_turn_speed': 0.8,
            'recovery_turn_speed': 0.3,
            # Slightly below max_left so recovery has a hysteresis band.
            'recovery_clear_distance': 0.36,
            'avoid_clear_distance': 0.28,
            'recovery_linear': 0.03,
            'recovery_left_turn_speed': 0.25,
            'recovery_timeout': 20.0,
            'integral_limit': 0.8,
            'derivative_filter_time': 0.3,
            'pid_dt_min': 0.001,
            'pid_dt_max': 1.0,
            'corner_max_turn_speed': 0.4,
            'turn_exit_front': 0.8,
            'turn_exit_left_max': 0.7,
            'turn_min_angle_deg': 60.0,
            'turn_force_angle_deg': 130.0,
            'front_confirm_scans': 2.0,
            'turn_min_time': 0.8,
            'turn_cooldown': 1.5,
            'turn_timeout': 12.0,
            'sensor_timeout': 1.0,
            'watchdog_period': 0.1,
            'front_angle_deg': 0.0,
            'left_angle_deg': 90.0,
            'front_left_angle_deg': 45.0,
            'front_right_angle_deg': -45.0,
            'front_scan_half_width_deg': 12.0,
            'left_scan_half_width_deg': 3.0,
            'diagonal_scan_half_width_deg': 12.0,
            'log_directory': '~/wall_follower_logs',
            # Angular PID (same gains as dual_PID).
            'kp': 2.40,
            'ki': 0.03,
            'kd': 0.50,
            'heading_kp': 0.8,
            # Speed (single value, slowed near walls / big errors).
            'speed': 0.09,
            'minimum_speed': 0.08,
            'wall_slowdown_error': 0.15,
            'front_slowdown_start': 0.65,
            'wall_slowdown_fraction': 0.6,
            'front_min_speed_fraction': 0.4,
            'status_log_period': 0.5,
            # Fault handling: wait, check clear, resume (bounded retries).
            'fault_retry_delay': 3.0,
            'max_fault_retries': 3.0,
            'fault_retry_reset_time': 10.0,
            'fault_log_period': 1.0,
        }
        self.live_parameters = (
            'kp', 'ki', 'kd', 'heading_kp', 'speed', 'minimum_speed',
        )
        self.load_settings()

        self.yaw = None
        self.last_scan = None
        self.last_odom = None
        self.state = 'FOLLOW'
        self.state_since = time.monotonic()
        self.last_turn_end = -1e9
        self.turn_done_count = 0
        self.front_hits = 0
        self.turn_accum = 0.0
        self.turn_last_yaw = None
        self.left_distance = None
        self.front_right = None
        self.wall_angle = 0.0
        self.row_count = 0
        self.fault_retries = 0
        self.follow_ok_since = None
        self.last_fault_log = None
        self.reset_pid()

        folder = Path(self.log_directory).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        run_name = datetime.now().strftime('single_PID_%Y%m%d_%H%M%S_%f')
        self.csv_path = folder / f'{run_name}.csv'
        self.csv_file = self.csv_path.open('x', newline='', encoding='utf-8')
        self.writer = csv.writer(self.csv_file)
        self.log_started = time.monotonic()
        self.writer.writerow([
            'scan_time_s', 'front_m', 'left_m', 'front_left_m',
            'front_right_m', 'wall_angle_rad', 'error_m',
            'linear_m_s', 'angular_rad_s', 'state', 'yaw_deg',
            'turn_accum_deg', 'speed_reference_m_s',
            'kp', 'ki', 'kd', 'heading_kp', 'speed', 'minimum_speed',
            'controller', 'elapsed_wall_s',
        ])
        self.csv_file.flush()
        settings_path = self.csv_path.with_suffix('.settings.json')
        with settings_path.open('w', encoding='utf-8') as f:
            json.dump({n: getattr(self, n) for n in self.defaults}, f, indent=2)

        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos_profile_sensor_data)
        self.create_timer(self.watchdog_period, self.watchdog)
        self.create_timer(self.status_log_period, self.print_pid_error)
        self.add_on_set_parameters_callback(self.update_parameters)
        self.get_logger().info('LEFT WALL: single angular PID + wall angle')
        self.get_logger().info(f'CSV: {self.csv_path}')

    def print_pid_error(self):
        if self.left_distance is None:
            return
        error = self.left_distance - self.target
        self.get_logger().info(
            f'{self.state} | left {self.left_distance:.3f} m | '
            f'error {error:+.4f} m | wall angle {self.wall_angle:+.3f} rad')

    # ------------------------------------------------------------ settings
    def load_settings(self):
        for name, default in self.defaults.items():
            self.declare_parameter(
                name, default,
                descriptor=ParameterDescriptor(
                    read_only=name not in self.live_parameters))
        settings = {n: self.get_parameter(n).value for n in self.defaults}
        reason = self.validate_settings(settings)
        if reason:
            raise ValueError(reason)
        for name, value in settings.items():
            setattr(self, name, value if isinstance(value, str) else float(value))

    def validate_settings(self, s):
        gains = ('kp', 'ki', 'kd', 'heading_kp')
        speeds = ('speed', 'minimum_speed', 'recovery_linear')
        for name, value in s.items():
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
            if name in gains and value > 10.0:
                return f'{name} must be between 0 and 10'
            if name in speeds and value > 0.12:
                return f'{name} must be between 0 and 0.12 m/s'
        positive = (
            'target', 'max_left', 'min_left', 'front_limit', 'front_left_limit',
            'max_turn_speed', 'recovery_turn_speed', 'recovery_clear_distance',
            'avoid_clear_distance', 'recovery_linear',
            'recovery_left_turn_speed', 'recovery_timeout', 'pid_dt_min',
            'pid_dt_max', 'corner_max_turn_speed', 'turn_exit_front',
            'turn_exit_left_max', 'turn_min_angle_deg', 'turn_force_angle_deg',
            'front_confirm_scans', 'turn_min_time', 'turn_timeout',
            'sensor_timeout', 'watchdog_period', 'front_scan_half_width_deg',
            'left_scan_half_width_deg', 'diagonal_scan_half_width_deg',
            'wall_slowdown_error', 'status_log_period', 'speed',
            'fault_retry_delay', 'fault_retry_reset_time', 'fault_log_period',
        )
        for name in positive:
            if s[name] <= 0.0:
                return f'{name} must be positive'
        if not (s['min_left'] < s['avoid_clear_distance'] <= s['target']
                <= s['recovery_clear_distance'] <= s['max_left']):
            return ('Require min_left < avoid_clear_distance <= target '
                    '<= recovery_clear_distance <= max_left')
        if s['pid_dt_min'] > s['pid_dt_max']:
            return 'pid_dt_min must not exceed pid_dt_max'
        if s['minimum_speed'] > s['speed']:
            return 'minimum_speed must not exceed speed'
        if s['front_slowdown_start'] <= s['front_limit']:
            return 'front_slowdown_start must exceed front_limit'
        if s['turn_min_angle_deg'] >= s['turn_force_angle_deg']:
            return 'turn_min_angle_deg must be below turn_force_angle_deg'
        if not 0.0 <= s['wall_slowdown_fraction'] <= 1.0:
            return 'wall_slowdown_fraction must be between 0 and 1'
        if not 0.0 < s['front_min_speed_fraction'] <= 1.0:
            return 'front_min_speed_fraction must be in (0, 1]'
        return ''

    def update_parameters(self, params):
        pending = {n: getattr(self, n) for n in self.defaults}
        changes = {}
        for param in params:
            if param.name not in self.defaults:
                continue
            if param.name not in self.live_parameters:
                return SetParametersResult(
                    successful=False,
                    reason=f'{param.name} is a startup setting; edit YAML and restart')
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

    # ------------------------------------------------------------- helpers
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
        if new_state != self.state:
            self.state = new_state
            self.state_since = time.monotonic()
            self.reset_pid()
            self.csv_file.flush()
            self.get_logger().info(f'State: {new_state}')

    def fault(self, message):
        if self.state != 'FAULT':
            self.get_logger().error(message)
            self.change_state('FAULT')
            self.last_fault_log = None
        return self.send()

    def try_clear_fault(self, now, front, front_left, front_right, left):
        """Resume from FAULT after a pause once the robot has room to move.
        After max_fault_retries consecutive failures the fault stays latched."""
        if self.fault_retries >= self.max_fault_retries:
            return False
        if now - self.state_since < self.fault_retry_delay:
            return False
        if (front < self.front_slowdown_start
                or front_left < self.front_left_limit
                or front_right < self.front_left_limit):
            return False
        if left is not None and left < self.min_left:
            return False
        self.fault_retries += 1
        self.follow_ok_since = None
        self.get_logger().warn(
            f'Retrying after fault ({self.fault_retries}/'
            f'{int(self.max_fault_retries)})')
        self.change_state('FOLLOW')
        return True

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        components = (q.x, q.y, q.z, q.w)
        if not all(math.isfinite(v) for v in components):
            return
        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.last_odom = time.monotonic()

    def sensors_ready(self, now):
        return (
            self.last_scan is not None and self.last_odom is not None
            and self.yaw is not None
            and now - self.last_scan <= self.sensor_timeout
            and now - self.last_odom <= self.sensor_timeout)

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
            difference = (angle - angle_deg + 180.0) % 360.0 - 180.0
            if abs(difference) <= width:
                readings.append(value)
        if not readings:
            return None
        if nearest:
            # second-smallest ray: one ghost point cannot trigger a stop/turn
            if len(readings) < 2:
                return None
            readings.sort()
            return readings[1]
        return sum(readings) / len(readings)

    # ---------------------------------------------------------- controller
    def angular_pid(self, error, stamp):
        if self.last_stamp is not None and self.last_error is not None:
            dt = stamp - self.last_stamp
            if self.pid_dt_min <= dt <= self.pid_dt_max:
                self.error_sum = clamp(
                    self.error_sum + 0.5 * (error + self.last_error) * dt,
                    -self.integral_limit, self.integral_limit)
                derivative = (error - self.last_error) / dt
                alpha = dt / (self.derivative_filter_time + dt)
                self.filtered_d += alpha * (derivative - self.filtered_d)
            else:
                self.error_sum = 0.0
                self.filtered_d = 0.0
        self.last_error = error
        self.last_stamp = stamp
        return (self.kp * error + self.ki * self.error_sum
                + self.kd * self.filtered_d)

    def reference_speed(self, wall_error, front):
        """Single forward speed, reduced for large wall error / close front."""
        severity = clamp(abs(wall_error) / self.wall_slowdown_error, 0.0, 1.0)
        target_speed = max(
            self.minimum_speed,
            self.speed * (1.0 - self.wall_slowdown_fraction * severity))
        if front < self.front_slowdown_start:
            front_fraction = clamp(
                (front - self.front_limit)
                / (self.front_slowdown_start - self.front_limit), 0.0, 1.0)
            target_speed *= (self.front_min_speed_fraction
                             + (1.0 - self.front_min_speed_fraction) * front_fraction)
        return clamp(target_speed, 0.0, self.speed)

    # ---------------------------------------------------------------- log
    def record(self, stamp, front, left, front_left, cmd, speed_ref=0.0):
        # While latched in FAULT only log occasionally.
        if self.state == 'FAULT':
            now = time.monotonic()
            if (self.last_fault_log is not None
                    and now - self.last_fault_log < self.fault_log_period):
                return
            self.last_fault_log = now
        self.writer.writerow([
            stamp, front, left, front_left, self.front_right, self.wall_angle,
            None if left is None else left - self.target,
            cmd.linear.x, cmd.angular.z, self.state,
            None if self.yaw is None else math.degrees(self.yaw),
            math.degrees(self.turn_accum), speed_ref,
            self.kp, self.ki, self.kd, self.heading_kp,
            self.speed, self.minimum_speed,
            'single_PID', time.monotonic() - self.log_started,
        ])
        self.row_count += 1
        # 10 Hz is cheap to flush; avoids a truncated last line on Ctrl+C.
        self.csv_file.flush()

    # --------------------------------------------------------------- main
    def scan_callback(self, scan):
        now = time.monotonic()
        self.last_scan = now
        stamp = scan.header.stamp.sec + scan.header.stamp.nanosec * 1e-9
        far = min(scan.range_max, 10.0)

        front = self.distance(scan, self.front_angle_deg,
                              self.front_scan_half_width_deg, nearest=True)
        left = self.distance(scan, self.left_angle_deg,
                             self.left_scan_half_width_deg)
        ahead = self.distance(scan, self.front_left_angle_deg,
                              self.left_scan_half_width_deg)
        front_left = self.distance(scan, self.front_left_angle_deg,
                                   self.diagonal_scan_half_width_deg, nearest=True)
        front_right = self.distance(scan, self.front_right_angle_deg,
                                    self.diagonal_scan_half_width_deg, nearest=True)

        front = far if front is None else min(front, far)
        front_left = far if front_left is None else min(front_left, far)
        front_right = far if front_right is None else min(front_right, far)

        self.left_distance = left
        self.front_right = front_right

        # Wall angle from the 90 deg and 45 deg rays; wall_dist is the
        # perpendicular distance to the wall.
        wall_angle = 0.0
        wall_dist = left
        if left is not None and ahead is not None:
            theta = math.radians(self.left_angle_deg - self.front_left_angle_deg)
            wall_angle = clamp(
                math.atan2(ahead * math.cos(theta) - left,
                           ahead * math.sin(theta)), -0.6, 0.6)
            wall_dist = left * math.cos(wall_angle)
        self.wall_angle = wall_angle

        # Latched fault: stop, and resume once it is safe and retries remain.
        if self.state == 'FAULT':
            cmd = self.send()
            self.reset_pid()
            if self.sensors_ready(now):
                self.try_clear_fault(now, front, front_left, front_right, left)
            self.record(stamp, front, left, front_left, cmd)
            return

        if not self.sensors_ready(now):
            cmd = self.send()
            self.reset_pid()
            self.record(stamp, front, left, front_left, cmd)
            return

        # A stable FOLLOW period earns back the fault retries.
        if self.state == 'FOLLOW':
            if self.follow_ok_since is None:
                self.follow_ok_since = now
            elif (self.fault_retries
                  and now - self.follow_ok_since >= self.fault_retry_reset_time):
                self.fault_retries = 0
        else:
            self.follow_ok_since = None

        # ---- right turn: LiDAR + accumulated odom yaw
        if self.state == 'TURN_RIGHT':
            elapsed = now - self.state_since
            if self.yaw is not None and self.turn_last_yaw is not None:
                self.turn_accum += -angle_difference(self.yaw, self.turn_last_yaw)
            self.turn_last_yaw = self.yaw
            accum = self.turn_accum

            if front_right < self.front_left_limit:
                cmd = self.fault('Obstacle too close during right turn')
            elif elapsed > self.turn_timeout:
                self.get_logger().warn('Right turn timeout: back to FOLLOW')
                self.last_turn_end = now
                self.change_state('FOLLOW')
                cmd = self.send()
            else:
                wall_ok = (left is not None
                           and self.min_left <= left <= self.turn_exit_left_max)
                normal = (elapsed >= self.turn_min_time
                          and accum >= math.radians(self.turn_min_angle_deg)
                          and front >= self.turn_exit_front and wall_ok)
                forced = (accum >= math.radians(self.turn_force_angle_deg)
                          and front >= self.front_slowdown_start)
                self.turn_done_count = (self.turn_done_count + 1
                                        if (normal or forced) else 0)
                if self.turn_done_count >= 2:
                    self.last_turn_end = now
                    self.change_state('FOLLOW')
                    cmd = self.send()
                else:
                    cmd = self.send(angular=-self.corner_max_turn_speed)
            self.record(stamp, front, left, front_left, cmd)
            return

        # ---- wall ahead: confirm over several scans, then turn
        if front <= self.front_limit:
            self.front_hits += 1
            if (self.front_hits >= int(self.front_confirm_scans)
                    and now - self.last_turn_end >= self.turn_cooldown
                    and front_right >= self.front_left_limit):
                self.turn_done_count = 0
                self.turn_accum = 0.0
                self.turn_last_yaw = self.yaw
                self.change_state('TURN_RIGHT')
                cmd = self.send(angular=-self.corner_max_turn_speed)
            else:
                cmd = self.send()
            self.record(stamp, front, left, front_left, cmd)
            return
        self.front_hits = 0

        if front_left < self.front_left_limit:
            cmd = self.send()
            self.reset_pid()
            self.record(stamp, front, left, front_left, cmd)
            return

        # ---- state transitions
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
                    'RECOVER_LEFT' if left > self.max_left else 'FOLLOW')

        # ---- motion
        speed_ref = 0.0
        if (self.state in ('RECOVER_LEFT', 'AVOID_LEFT')
                and now - self.state_since > self.recovery_timeout):
            cmd = self.fault('Left-wall recovery timeout')
        elif self.state == 'AVOID_LEFT':
            if front_right < self.front_left_limit:
                cmd = self.fault('Insufficient clearance to move away from wall')
            else:
                cmd = self.send(angular=-self.recovery_turn_speed)
        elif left is None or self.state == 'RECOVER_LEFT':
            # Arc left toward the wall (creeping forward, not spinning).
            cmd = self.send(linear=self.recovery_linear,
                            angular=self.recovery_left_turn_speed)
        else:
            error = wall_dist - self.target
            angular = clamp(
                self.angular_pid(error, stamp) + self.heading_kp * wall_angle,
                -self.max_turn_speed, self.max_turn_speed)
            speed_ref = self.reference_speed(error, front)
            cmd = self.send(linear=speed_ref, angular=angular)

        self.record(stamp, front, left, front_left, cmd, speed_ref)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SinglePID()
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
