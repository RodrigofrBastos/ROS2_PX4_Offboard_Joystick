#!/usr/bin/env python3
############################################################################
# Controlador PID para Drone (ROS2 + PX4)
# Versão Corrigida e Funcional
############################################################################

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.duration import Duration
from enum import Enum

from ros_rl_interfaces.msg import VisualInfo
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import VehicleAttitude

from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float64

def get_message_name_version(msg_class):
    if msg_class.MESSAGE_VERSION == 0:
        return ""
    return f"_v{msg_class.MESSAGE_VERSION}"
    
class State(Enum):
    STARTING = 1  # Estabelecendo stream offboard
    IDLE = 2      # Aguardando setpoint
    MOVING = 3    # Movendo para o alvo

class SetpointCamera(Node):
    def __init__(self):
        super().__init__('setpoint_camera_pid_controller')
        
        self.state = State.STARTING
        
        self.init_parameters()
        self.get_parameters()
        self.init_variables()
        self.init_ros_interfaces()

    def init_parameters(self):
        self.declare_parameter('timer_period', 0.05)
        self.declare_parameter('feature_topic', '/pose_feature')
        self.declare_parameter('offset_z_distance', 0.5)
        self.declare_parameter('threshold_distance', 0.1)

        self.declare_parameter('px4_cmd_topic', '/fmu/in/vehicle_command')
        self.declare_parameter('px4_pos_topic', '/fmu/out/vehicle_local_position_v1')
        self.declare_parameter('offboard_mode_topic', '/fmu/in/offboard_control_mode')
        self.declare_parameter('trajectory_setpoint_topic', '/fmu/in/trajectory_setpoint')

        self.declare_parameter('offboard_stream_delay_s', 2.0)
        self.declare_parameter('pos_staleness_threshold_s', 0.5)

    def get_parameters(self):
        self.timer_period = self.get_parameter('timer_period').get_parameter_value().double_value
        self.feature_topic = self.get_parameter('feature_topic').get_parameter_value().string_value
        self.offset_z_distance = self.get_parameter('offset_z_distance').get_parameter_value().double_value
        self.threshold_distance = self.get_parameter('threshold_distance').get_parameter_value().double_value
        
        self.px4_cmd_topic = self.get_parameter('px4_cmd_topic').get_parameter_value().string_value
        self.px4_pos_topic = self.get_parameter('px4_pos_topic').get_parameter_value().string_value
        self.offboard_mode_topic = self.get_parameter('offboard_mode_topic').get_parameter_value().string_value
        self.trajectory_setpoint_topic = self.get_parameter('trajectory_setpoint_topic').get_parameter_value().string_value

        self.offboard_stream_delay_s = self.get_parameter('offboard_stream_delay_s').get_parameter_value().double_value
        self.pos_staleness_threshold_s = self.get_parameter('pos_staleness_threshold_s').get_parameter_value().double_value
        self.offboard_cycles_needed = int(self.offboard_stream_delay_s / self.timer_period)

    def init_ros_interfaces(self):
        qos_px4 = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        qos_cmds = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        traj_topic = f"{self.trajectory_setpoint_topic}{get_message_name_version(TrajectorySetpoint)}"
        cmd_topic = self.px4_cmd_topic
        offb_topic = self.offboard_mode_topic
        pos_topic = f"{self.px4_pos_topic}{get_message_name_version(VehicleLocalPosition)}"
        vehicle_att_top = f"/fmu/out/vehicle_attitude{get_message_name_version(VehicleAttitude)}"
        
        self.cmd_pub = self.create_publisher(VehicleCommand, cmd_topic, qos_cmds)
        self.offboard_control_pub = self.create_publisher(OffboardControlMode, offb_topic, qos_cmds)
        self.trajectory_setpoint_pub = self.create_publisher(TrajectorySetpoint, traj_topic, qos_cmds)

        self.pos_sub = self.create_subscription(
            VehicleLocalPosition, pos_topic, self.px4_pos_callback, qos_px4)
        self.feature_sub = self.create_subscription(
            VisualInfo, self.feature_topic, self.visual_info_callback, 10)
        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            vehicle_att_top,
            self.attitude_callback,
            qos_px4)

        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def init_variables(self):
        self.setpoint_ned = np.array([0.0, 0.0, 0.0])
        self.current_pos_ned = np.array([0.0, 0.0, 0.0])
        self.current_pos_received = False
        self.last_pos_timestamp = None
        
        self.offboard_stream_counter = 0
        self.target_coords = None
        self.new_setpoint_received = False
        self.trueYaw = 0.0  #current yaw value of drone
        self.trueRoll = 0.0  #current roll value of drone
        self.truePitch = 0.0  #current pitch value of drone


    # --- CALLBACKS ---
    def visual_info_callback(self, msg):
        """Recebe coordenadas da câmera e atualiza o setpoint."""
        if not msg:
            return
            
        try:
            self.kp_phi = 10.0
            self.kp_theta = 10.0
            self.factor = 10.0
            self.point_u = Float64MultiArray()
            self.nose_v = Float64MultiArray()
            self.nose_u = np.array(msg.u)
            self.nose_v = np.array(msg.v)
            self.depth = np.array(msg.z)
            self.jacobian = np.array(msg.jacobian.data).reshape(
                msg.jacobian.layout.dim[0].size,   # rows
                msg.jacobian.layout.dim[1].size    # cols
                )
            self.references = np.array(msg.feature_reference.data)
            self.error = np.zeros(len(self.references))
            if self.nose_u is not None and self.nose_v is not None:
                kp = []
                for i in range(0, len(self.references)//2, 1):
                    kp.append(self.kp_phi)
                    kp.append(self.kp_theta)
                kp = np.array(kp).reshape(-1, 1)  # transforma em coluna
                for i in range(0, len(self.references) - 1, 2):
                    self.error[i+1] = self.references[i]-self.nose_u[i//2]
                    self.error[i] = self.factor*(self.references[i+1]-self.nose_v[i//2])

                error_column = np.array(self.error).reshape(-1, 1)
                qpoint = (self.jacobian @ (kp * error_column))
                self.get_logger().debug(f"VisualInfo recebido: u={self.nose_u}, v={self.nose_v}, z={self.depth}")
                
        except (TypeError, IndexError):
            # Fallback se não for iterável (ou seja, se for apenas um escalar "5.0" ou 5.0)
            self.get_logger().warn("Campos da VisualInfo não parecem ser listas. Tentando conversão direta.", throttle_duration_sec=5.0)

        self.target_coords = np.array((qpoint[0], qpoint[1], qpoint[2])).reshape(3)

        self.target_coords = np.column_stack((qpoint[0], qpoint[1], qpoint[2]))
        self.setpoint_ned = self.transform_target_to_ned_setpoint(self.target_coords)
        self.new_setpoint_received = True
        
        self.get_logger().debug(f"Novo setpoint: {self.setpoint_ned}")
          
    def px4_pos_callback(self, msg):
        """Atualiza posição do drone (já em NED)."""
        self.current_pos_ned = np.array([msg.x, msg.y, msg.z])
        self.current_pos_received = True
        self.last_pos_timestamp = self.get_clock().now()

    def attitude_callback(self, msg):
        """Atualiza atitude do drone."""
        orientation_q = msg.q
        t0 = +2.0 * (orientation_q[0] * orientation_q[1] + orientation_q[2] * orientation_q[3])
        t1 = orientation_q[0] * orientation_q[0] - orientation_q[1] * orientation_q[1] - orientation_q[2] * orientation_q[2] + orientation_q[3] * orientation_q[3]
        self.trueRoll = np.arctan2(t0, t1)
        t2 = +2.0 * (orientation_q[0] * orientation_q[2] - orientation_q[3] * orientation_q[1])
        self.truePitch = np.arcsin(t2)
        t3 = +2.0*(orientation_q[0]*orientation_q[3] + orientation_q[1]*orientation_q[2])
        t4 = +2.0*(orientation_q[0]*orientation_q[0] + orientation_q[1]*orientation_q[1] - orientation_q[2]*orientation_q[2] - orientation_q[3]*orientation_q[3])
        self.trueYaw = -np.arctan2(t3, t4)
        
    # --- TRANSFORMAÇÕES ---

    def wrap_pi(self, a):
        return (a + np.pi) % (2*np.pi) - np.pi

    def enu_to_ned(self, v):
        v = np.asarray(v, dtype=np.float64).reshape(3)
        return np.array([v[1], v[0], -v[2]], dtype=np.float64)

    def ned_to_enu(self, v):
        v = np.asarray(v, dtype=np.float64).reshape(3)
        return np.array([v[1], v[0], -v[2]], dtype=np.float64)

    def flu_to_frd(self, v):
        v = np.asarray(v, dtype=np.float64).reshape(3)
        return np.array([v[0], -v[1], -v[2]], dtype=np.float64)

    def frd_to_flu(self, v):
        v = np.asarray(v, dtype=np.float64).reshape(3)
        return np.array([v[0], -v[1], -v[2]], dtype=np.float64)

    def rpy_ned_from_enu(self, roll_flu, pitch_flu, yaw_enu):
        roll_frd  = roll_flu
        pitch_frd = -pitch_flu
        yaw_ned   = np.pi/2 - yaw_enu
        return roll_frd, pitch_frd, self.wrap_pi(yaw_ned)

    def R_body_to_ned_from_rpy_ZYX(self, roll_frd, pitch_frd, yaw_ned):
        cr, sr = np.cos(roll_frd),  np.sin(roll_frd)
        cp, sp = np.cos(pitch_frd), np.sin(pitch_frd)
        cy, sy = np.cos(yaw_ned),   np.sin(yaw_ned)
        return np.array([
            [cy*cp,               cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
            [sy*cp,               sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
            [-sp,                 cp*sr,             cp*cr           ]
        ], dtype=np.float64)
        
    def get_rotation_matrix_body_to_ned(self, roll_deg, pitch_deg, yaw_deg):
        """
        Retorna matriz de rotação do frame body FRD para mundo NED
        usando convenção Tait-Bryan ZYX (yaw-pitch-roll)
        """
        # Converter para radianos
        roll = np.deg2rad(roll_deg)
        pitch = np.deg2rad(pitch_deg)
        yaw = np.deg2rad(yaw_deg)
        
        # Matriz de rotação em X (roll)
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ], dtype=np.float64)
        
        # Matriz de rotação em Y (pitch)
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ], dtype=np.float64)
        
        # Matriz de rotação em Z (yaw)
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Rotação composta: R = R_z * R_y * R_x (ordem ZYX)
        R = R_z @ R_y @ R_x
        
        return R


    def get_rotation_matrix_ned_to_body(self, roll_deg, pitch_deg, yaw_deg):
        """
        Retorna matriz de rotação do mundo NED para frame body FRD
        (transposta da body_to_ned)
        """
        R_body_to_ned = self.get_rotation_matrix_body_to_ned(roll_deg, pitch_deg, yaw_deg)
        return R_body_to_ned.T

    def transform_target_to_ned_setpoint(self,
                                        target_vec,  # Vetor deslocamento em ENU mundo
                                        apply_offset=False):
        
        print("\n--- TRANSFORMAÇÃO: Deslocamento Mundo → Setpoint ---")
        print(f"    target_vec (deslocamento ENU)={target_vec}")

        # 1) Target é o vetor deslocamento em ENU mundo
        displacement_enu_old = np.array(target_vec, dtype=np.float64).reshape(3)
        displacement_enu = np.zeros(3, dtype=np.float64)
        displacement_enu[0] = displacement_enu_old[1]  # Inverter X (câmera)
        displacement_enu[1] = displacement_enu_old[0]  # Inverter Y (câmera)
        displacement_enu[2] = displacement_enu_old[2] - 1.0  # Aplicar offset Z
        self.get_logger().info(f"1) Deslocamento em ENU mundo: {displacement_enu}")

        # 2) Posição atual do robô em NED
        robot_pos_ned = np.array([
            float(self.current_pos_ned[0]),
            float(self.current_pos_ned[1]),
            float(self.current_pos_ned[2])
        ], dtype=np.float64)
        
        self.get_logger().info(f"2) Robô em NED: {robot_pos_ned}")

        # 3) Converter o deslocamento de ENU para NED
        # NÃO rotaciona! O vetor já está em coordenadas mundo
        displacement_ned = self.enu_to_ned(displacement_enu)
        self.get_logger().info(f"3) Deslocamento ENU → NED: {displacement_ned}")

        # 4) Calcular posição do objeto (setpoint final)
        # objeto_ned = robô_ned + deslocamento_ned
        target_position_ned = displacement_ned + robot_pos_ned
        self.get_logger().info(f"4) Posição do objeto (setpoint): {target_position_ned}")
        self.get_logger().info(f"   = {robot_pos_ned} + {displacement_ned}")

        # 5) Aplicar offset se necessário
        if apply_offset:
            offset_ned = np.array([
                float(self.offset_x),
                float(self.offset_y),
                float(self.offset_z)
            ], dtype=np.float64)
            target_position_ned = target_position_ned + offset_ned
            self.get_logger().info(f"5) Com offset: {target_position_ned}")
        
        return target_position_ned
    
    # --- MÁQUINA DE ESTADOS ---
    def timer_callback(self):
        self.publish_offboard_mode()

        if self.state == State.STARTING:
            self.run_state_starting()
        elif self.state == State.IDLE:
            self.run_state_idle()
        # elif self.state == State.MOVING:
        #     self.run_state_moving()

    def run_state_starting(self):
        """Estabelece stream antes de ativar Offboard."""
        self.publish_velocity_setpoint(np.array([0.0, 0.0, 0.0]))

        if self.offboard_stream_counter < self.offboard_cycles_needed:
            self.offboard_stream_counter += 1
            if self.offboard_stream_counter % 20 == 0:
                self.get_logger().info(
                    f"Stream offboard: {self.offboard_stream_counter}/{self.offboard_cycles_needed}")
        else:
            self.get_logger().info("Ativando modo Offboard...")
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                param1=1.0,
                param2=6.0
            )
            self.state = State.IDLE
            self.get_logger().info(f"Transição: STARTING -> {self.state.name}")

    def run_state_idle(self):
        """Aguarda novo setpoint com posição válida."""
        self.publish_velocity_setpoint(np.array([0.0, 0.0, 0.0]))
        
        if self.new_setpoint_received:
            if self.is_position_fresh():
                self.get_logger().info(
                    f"Novo alvo (NED): [{self.setpoint_ned[0]:.2f}, "
                    f"{self.setpoint_ned[1]:.2f}, {self.setpoint_ned[2]:.2f}]"
                )
                self.reset_pid_controller()
                self.state = State.MOVING
                self.new_setpoint_received = False
            else:
                self.get_logger().warn(
                    "Setpoint recebido mas posição do drone inválida. Aguardando...",
                    throttle_duration_sec=2.0
                )
    def reset_pid_controller(self):
        """Reseta estado do PID."""
        self.integral_error = np.array([0.0, 0.0, 0.0])
        self.last_error = np.array([0.0, 0.0, 0.0])
        self.last_pid_time = self.get_clock().now()
    def is_position_fresh(self):
        if not self.current_pos_received or self.last_pos_timestamp is None:
            self.get_logger().warn("❌ Nenhuma posição recebida ainda!")  # ← ADICIONE
            return False
            
        elapsed = self.get_clock().now() - self.last_pos_timestamp
        elapsed_sec = elapsed.nanoseconds / 1e9
        
        # ← ADICIONE ESTE LOG
        self.get_logger().info(f"⏱️  Última posição: {elapsed_sec:.3f}s atrás")
        
        is_fresh = elapsed < Duration(seconds=self.pos_staleness_threshold_s)
        
        if not is_fresh:
            self.get_logger().error(
                f"Posição antiga: {elapsed_sec:.2f}s > {self.pos_staleness_threshold_s}s",
                throttle_duration_sec=1.0
            )
        
        return is_fresh
    
    # --- PUBLICAÇÕES ---

    def publish_offboard_mode(self):
        """Mantém modo Offboard ativo."""
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_pub.publish(msg)

    def publish_vehicle_command(self, command, **params):
        """Envia comando ao PX4."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.cmd_pub.publish(msg)
        
    def publish_velocity_setpoint(self, velocity: np.ndarray):
        """Envia comando de velocidade (NED)."""
        msg = TrajectorySetpoint()
        msg.velocity = [float(velocity[0]), float(velocity[1]), float(velocity[2])]
        msg.position = [np.nan, np.nan, np.nan]
        msg.yaw = np.nan
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    controller_node = SetpointCamera()
    
    try:
        rclpy.spin(controller_node)
    except KeyboardInterrupt:
        controller_node.get_logger().info("Interrompido pelo usuário.")
    except Exception as e:
        controller_node.get_logger().error(f"Erro: {e}")
        raise
    finally:
        controller_node.get_logger().info("Desligando...")
        controller_node.publish_velocity_setpoint(np.array([0.0, 0.0, 0.0]))
        controller_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()