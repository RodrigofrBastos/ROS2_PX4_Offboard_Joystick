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
        
        self.get_logger().info("Nó de controle PID iniciado.")
        self.get_logger().info(f"Estado inicial: {self.state.name}")

    def init_parameters(self):
        self.declare_parameter('timer_period', 0.05)
        self.declare_parameter('feature_topic', '/pose_feature')
        self.declare_parameter('offset_z_distance', 0.5)
        self.declare_parameter('threshold_distance', 0.15)

        self.declare_parameter('px4_cmd_topic', '/fmu/in/vehicle_command')
        self.declare_parameter('px4_pos_topic', '/fmu/out/vehicle_local_position_v1')
        self.declare_parameter('offboard_mode_topic', '/fmu/in/offboard_control_mode')
        self.declare_parameter('trajectory_setpoint_topic', '/fmu/in/trajectory_setpoint')

        self.declare_parameter('kp', 1.0)
        self.declare_parameter('ki', 0.1)
        self.declare_parameter('kd', 0.5)
        self.declare_parameter('max_vel', 0.5)
        self.declare_parameter('integral_limit', 1.0)

        self.declare_parameter('offboard_stream_delay_s', 2.0)
        self.declare_parameter('pos_staleness_threshold_s', 0.5)
        
        default_tf_matrix = [0., 1., 0., 0., 0., 1., 1., 0., 0.]
        self.declare_parameter('target_to_enu_tf', default_tf_matrix)

    def get_parameters(self):
        self.timer_period = self.get_parameter('timer_period').get_parameter_value().double_value
        self.feature_topic = self.get_parameter('feature_topic').get_parameter_value().string_value
        self.offset_z_distance = self.get_parameter('offset_z_distance').get_parameter_value().double_value
        self.threshold_distance = self.get_parameter('threshold_distance').get_parameter_value().double_value
        
        self.px4_cmd_topic = self.get_parameter('px4_cmd_topic').get_parameter_value().string_value
        self.px4_pos_topic = self.get_parameter('px4_pos_topic').get_parameter_value().string_value
        self.offboard_mode_topic = self.get_parameter('offboard_mode_topic').get_parameter_value().string_value
        self.trajectory_setpoint_topic = self.get_parameter('trajectory_setpoint_topic').get_parameter_value().string_value

        self.kp = self.get_parameter('kp').get_parameter_value().double_value
        self.ki = self.get_parameter('ki').get_parameter_value().double_value
        self.kd = self.get_parameter('kd').get_parameter_value().double_value
        self.max_vel = self.get_parameter('max_vel').get_parameter_value().double_value
        self.integral_limit = self.get_parameter('integral_limit').get_parameter_value().double_value

        self.offboard_stream_delay_s = self.get_parameter('offboard_stream_delay_s').get_parameter_value().double_value
        self.pos_staleness_threshold_s = self.get_parameter('pos_staleness_threshold_s').get_parameter_value().double_value
        self.offboard_cycles_needed = int(self.offboard_stream_delay_s / self.timer_period)
        
        tf_list = self.get_parameter('target_to_enu_tf').get_parameter_value().double_array_value
        self.T_target_to_enu = np.array(tf_list).reshape(3, 3)
        self.get_logger().info(f"Matriz de Transformação (Target -> ENU):\n{self.T_target_to_enu}")

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
        
        self.integral_error = np.array([0.0, 0.0, 0.0])
        self.last_error = np.array([0.0, 0.0, 0.0])
        self.last_pid_time = self.get_clock().now()
        
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
            x_val = np.array(msg.x)
            y_val = np.array(msg.y)
            z_val = np.array(msg.z)
        except Exception as e:
            self.get_logger().error(f"Erro ao extrair coordenadas: {e}", throttle_duration_sec=5.0)
            self.get_logger().warn("Campos da VisualInfo não parecem ser listas. Tentando conversão direta.", throttle_duration_sec=5.0)
            x_val = np.array(float(msg.x))
            y_val = np.array(float(msg.y))
            z_val = np.array(float(msg.z))
            return

        self.target_coords = np.column_stack((x_val, y_val, z_val))
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
        t0 = +2.0 * (orientation_q[3] * orientation_q[0] + orientation_q[1] * orientation_q[2])
        t1 = +1.0 - 2.0 * (orientation_q[0] * orientation_q[0] + orientation_q[1] * orientation_q[1])
        self.trueRoll = np.arctan2(t0, t1)
        t2 = +2.0 * (orientation_q[3] * orientation_q[1] - orientation_q[2] * orientation_q[0])
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        self.truePitch = np.arcsin(t2)
        t3 = +2.0*(orientation_q[3]*orientation_q[0] + orientation_q[1]*orientation_q[2])
        t4 = +1.0 - 2.0*(orientation_q[0]*orientation_q[0] + orientation_q[1]*orientation_q[1])   
        self.trueYaw = -np.arctan2(t3, t4)     
        
    # --- TRANSFORMAÇÕES ---

    def transform_target_to_ned_setpoint(self, target_coords_body):
        """
        Converte coordenadas do frame do robô (body FRD) para o frame global (world NED).
        
        Args:
            target_coords_body: np.array([x, y, z]) no frame do robô (FRD)
            
        Returns:
            setpoint_ned: np.array([north, east, down]) no frame global (NED)
        """
        print("\n" + "="*60)
        print("DEBUG: Transformação Body -> NED")
        print("="*60)
        
        # Garantir que target_coords_body é um array numpy float64 simples
        try:
            target_coords_body = np.array(target_coords_body, dtype=np.float64).flatten()
            if len(target_coords_body) != 3:
                raise ValueError(f"Target deve ter 3 coordenadas, recebeu {len(target_coords_body)}")
        except Exception as e:
            print(f"[ERRO] Falha ao converter target_coords_body: {e}")
            print(f"Tipo: {type(target_coords_body)}")
            print(f"Valor: {target_coords_body}")
            return np.array([0.0, 0.0, 0.0])
        
        # Obter posição e orientação atual do robô (do PX4)
        robot_pos_ned = np.array([
            float(self.current_pos_ned[0]),  # North
            float(self.current_pos_ned[1]),  # East
            float(self.current_pos_ned[2])   # Down
        ], dtype=np.float64)
        
        print(f"📍 Posição do robô (NED): {robot_pos_ned}")
        print(f"   North: {robot_pos_ned[0]:.3f} m")
        print(f"   East:  {robot_pos_ned[1]:.3f} m")
        print(f"   Down:  {robot_pos_ned[2]:.3f} m")
        
        # Obter orientação do robô (roll, pitch, yaw)
        roll = float(self.trueRoll)
        pitch = float(self.truePitch)
        yaw = float(self.trueYaw)
        
        print(f"\n🧭 Orientação do robô (rad):")
        print(f"   Roll:  {roll:.4f} rad ({np.degrees(roll):.2f}°)")
        print(f"   Pitch: {pitch:.4f} rad ({np.degrees(pitch):.2f}°)")
        print(f"   Yaw:   {yaw:.4f} rad ({np.degrees(yaw):.2f}°)")
        
        # Criar matriz de rotação do body para world (NED)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        
        # Matriz de rotação completa (ZYX - yaw, pitch, roll)
        R_body_to_ned = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr           ]
        ], dtype=np.float64)
        
        print(f"\n🔄 Matriz de Rotação (Body -> NED):")
        print(R_body_to_ned)
        
        print(f"\n🎯 Target no frame do robô (Body FRD):")
        print(f"   X (Forward): {float(target_coords_body[0]):.3f} m")
        print(f"   Y (Right):   {float(target_coords_body[1]):.3f} m")
        print(f"   Z (Down):    {float(target_coords_body[2]):.3f} m")
        
        # Aplicar rotação ao ponto no frame do robô
        target_rotated = R_body_to_ned @ target_coords_body
        
        print(f"\n↻ Target após rotação:")
        print(f"   North: {float(target_rotated[0]):.3f} m")
        print(f"   East:  {float(target_rotated[1]):.3f} m")
        print(f"   Down:  {float(target_rotated[2]):.3f} m")
        
        # Aplicar translação (posição do robô no mundo)
        target_global_ned = target_rotated + robot_pos_ned
        
        print(f"\n🌍 Target no frame global (NED):")
        print(f"   North: {float(target_global_ned[0]):.3f} m")
        print(f"   East:  {float(target_global_ned[1]):.3f} m")
        print(f"   Down:  {float(target_global_ned[2]):.3f} m")
        
        # Aplicar offset (para ficar acima do alvo)
        setpoint_ned = target_global_ned.copy()
        setpoint_ned[2] -= float(self.offset_z_distance)
        
        print(f"\n✈️ Setpoint final (com offset de {float(self.offset_z_distance):.3f} m):")
        print(f"   North: {float(setpoint_ned[0]):.3f} m")
        print(f"   East:  {float(setpoint_ned[1]):.3f} m")
        print(f"   Down:  {float(setpoint_ned[2]):.3f} m")
        print(f"   (Altitude relativa: {float(-setpoint_ned[2]):.3f} m)")
        print("="*60 + "\n")
        
        return setpoint_ned
        
    # --- MÁQUINA DE ESTADOS ---
    def timer_callback(self):
        self.publish_offboard_mode()

        if self.state == State.STARTING:
            self.run_state_starting()
        elif self.state == State.IDLE:
            self.run_state_idle()
        elif self.state == State.MOVING:
            self.run_state_moving()

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

    def run_state_moving(self):
        """Executa controle PID."""
        if not self.is_position_fresh():
            self.get_logger().error("Posição antiga! Revertendo para IDLE.")
            self.state = State.IDLE
            self.publish_velocity_setpoint(np.array([0.0, 0.0, 0.0]))
            return
            
        if self.new_setpoint_received:
            self.get_logger().info(f"Alvo atualizado: {self.setpoint_ned}")
            self.reset_pid_controller()
            self.new_setpoint_received = False

        # Calcula dt
        current_time = self.get_clock().now()
        dt_duration = current_time - self.last_pid_time
        dt = dt_duration.nanoseconds / 1e9
        self.last_pid_time = current_time
        
        if dt < (self.timer_period * 0.1):
            return

        # Controlador PID
        error = self.setpoint_ned - self.current_pos_ned
        
        self.integral_error += error * dt
        self.integral_error = np.clip(
            self.integral_error,
            -self.integral_limit,
            self.integral_limit
        )
        
        derivative_error = (error - self.last_error) / dt
        self.last_error = error
        
        output_vel = (
            self.kp * error +
            self.ki * self.integral_error +
            self.kd * derivative_error
        )
        
        output_norm = np.linalg.norm(output_vel)
        if output_norm > self.max_vel:
            output_vel = output_vel * (self.max_vel / output_norm)

        # Verifica sucesso
        error_distance = np.linalg.norm(error)
        
        if error_distance < self.threshold_distance:
            self.get_logger().info(f"Alvo alcançado! Erro: {error_distance:.3f}m")
            self.state = State.IDLE
            self.publish_velocity_setpoint(np.array([0.0, 0.0, 0.0]))
        else:
            self.publish_velocity_setpoint(output_vel)
            self.get_logger().info(
                f"Erro: {error_distance:.2f}m | Vel: {output_vel}",
                throttle_duration_sec=0.5
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

    def publish_velocity_setpoint(self, velocity: np.ndarray):
        """Envia comando de velocidade (NED)."""
        msg = TrajectorySetpoint()
        msg.velocity = [float(velocity[0]), float(velocity[1]), float(velocity[2])]
        msg.position = [np.nan, np.nan, np.nan]
        msg.yaw = np.nan
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_pub.publish(msg)

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