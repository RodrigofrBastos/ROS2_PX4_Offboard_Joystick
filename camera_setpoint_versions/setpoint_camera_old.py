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
        
        self.get_logger().info("Nó de controle PID iniciado.")
        self.get_logger().info(f"Estado inicial: {self.state.name}")

    def init_parameters(self):
        self.declare_parameter('timer_period', 0.05)
        self.declare_parameter('feature_topic', '/pose_feature')
        self.declare_parameter('offset_z_distance', 0.5)
        self.declare_parameter('threshold_distance', 0.1)

        self.declare_parameter('px4_cmd_topic', '/fmu/in/vehicle_command')
        self.declare_parameter('px4_pos_topic', '/fmu/out/vehicle_local_position_v1')
        self.declare_parameter('offboard_mode_topic', '/fmu/in/offboard_control_mode')
        self.declare_parameter('trajectory_setpoint_topic', '/fmu/in/trajectory_setpoint')

        self.declare_parameter('kp', 5.5)
        self.declare_parameter('ki', 0.1)
        self.declare_parameter('kd', 0.2)
        self.declare_parameter('max_vel', 1.3)
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
        self.error_top = self.create_publisher(Float64, "/pid_error", 10)
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def init_variables(self):
        self.setpoint_ned = np.array([0.0, 0.0, 0.0])
        self.current_pos_ned = np.array([0.0, 0.0, 0.0])
        self.current_pos_received = False
        self.last_pos_timestamp = None
        
        self.integral_error = 0.0
        self.last_error = 0.0
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

    def transform_target_to_ned_setpoint(self,
                                        target_vec,
                                        target_frame="body_frd",   # {"body_flu","body_frd","world_enu","world_ned"}
                                        target_is_absolute=False,  # True se já está no mundo
                                        attitude_convention="NED", # {"NED","ENU"}
                                        apply_offset=False):
        
        print("\n--- INICIANDO TRANSFORMAÇÃO DE ALVO (target_to_ned_setpoint) ---")
        print(f"    Parâmetros: target_vec={target_vec}, frame={target_frame}, absolute={target_is_absolute}, attitude={attitude_convention}, offset={apply_offset}")

        # 1) Normaliza entrada
        t = np.array(target_vec, dtype=np.float64).reshape(3)
        print(f"1) Alvo (t): {t}")

        # 2) Pose do robô em NED
        robot_pos_ned = np.array([
            float(self.current_pos_ned[0]),
            float(self.current_pos_ned[1]),
            float(self.current_pos_ned[2])
        ], dtype=np.float64)
        print(f"2) Posição atual do Robô (NED): {robot_pos_ned}")

        # 3) Ângulos do robô -> NED/FRD
        roll = float(self.trueRoll)
        pitch = float(self.truePitch)
        yaw = float(self.trueYaw)
        print(f"3a) Atitude atual (Raw/{attitude_convention}): Roll={roll:.2f}, Pitch={pitch:.2f}, Yaw={yaw:.2f}")

        if attitude_convention.upper() == "ENU":
            roll, pitch, yaw = self.rpy_ned_from_enu(roll, pitch, yaw)
            print(f"3b) Atitude convertida (NED/FRD): Roll={roll:.2f}, Pitch={pitch:.2f}, Yaw={yaw:.2f}")
        else:
            print(f"3b) Atitude já está em NED/FRD.")

        # 4) Matriz R_body->NED (ZYX)
        R_b2n = self.R_body_to_ned_from_rpy_ZYX(roll, pitch, yaw)
        print(f"4) Matriz de Rotação (R_body_to_ned):\n{R_b2n}")

        # 5) Seleciona o fluxo conforme o frame do alvo
        if target_is_absolute:  # alvo já no mundo
            print(f"5) Fluxo: Alvo é ABSOLUTO no mundo.")
            if target_frame.lower() == "world_enu":
                target_ned = self.enu_to_ned(t)
                print(f"5a) Convertendo Alvo ENU -> NED: {target_ned}")
            elif target_frame.lower() == "world_ned":
                target_ned = t
                print(f"5a) Alvo já está em NED: {target_ned}")
            else:
                raise ValueError("Alvo absoluto exige target_frame 'world_enu' ou 'world_ned'.")
            setpoint_ned = target_ned.copy()
            print(f"5b) Setpoint (Absoluto) NED (antes offset): {setpoint_ned}")

        else:  # alvo relativo ao corpo
            print(f"5) Fluxo: Alvo é RELATIVO ao robô.")
            if target_frame.lower() == "body_flu":
                t_body_frd = self.flu_to_frd(t)
                print(f"5a) Convertendo Alvo FLU -> FRD: {t_body_frd}")
            elif target_frame.lower() == "body_frd":
                t_body_frd = t
                print(f"5a) Alvo já está em FRD: {t_body_frd}")
            else:
                raise ValueError("Alvo relativo exige target_frame 'body_flu' ou 'body_frd'.")
            
            target_rotated_ned = R_b2n @ t_body_frd
            print(f"5b) Alvo rotacionado para Frame NED (R_b2n @ t_body_frd): {target_rotated_ned}")
            
            setpoint_ned = robot_pos_ned + target_rotated_ned
            print(f"5c) Setpoint (Relativo) NED (PosRobo + AlvoRot) (antes offset): {setpoint_ned}")

        # 6) Offset de altitude (NED: positivo para baixo)
        if apply_offset:
            offset_val = float(self.offset_z_distance)
            print(f"6) Aplicando Offset Z (NED): Subtraindo {offset_val} de Z.")
            setpoint_ned = setpoint_ned.copy() # Garante que não modifica o array original se for o caso (passo 5a, abs/ned)
            setpoint_ned[2] -= offset_val
        else:
            print(f"6) Offset Z não aplicado.")
        
        print(f"--- TRANSFORMAÇÃO CONCLUÍDA ---")
        print(f"7) Setpoint NED Final: {setpoint_ned}\n")
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
        """Executa controle PID (APENAS EIXO X)."""
        if not self.is_position_fresh():
            self.get_logger().error("Posição antiga! Revertendo para IDLE.")
            self.state = State.IDLE
            self.publish_velocity_setpoint(np.array([0.0, 0.0, 0.0]))
            return
            
        # ... (reset de 'new_setpoint_received' omitido) ...

        # Calcula dt
        current_time = self.get_clock().now()
        dt_duration = current_time - self.last_pid_time
        dt = dt_duration.nanoseconds / 1e9
        self.last_pid_time = current_time
        
        if dt < (self.timer_period * 0.1):
            return

        # --- Controlador PID (1D - Eixo X) ---
        
        # 1. Erro (Escalar)
        error_x = (self.setpoint_ned[0] - self.current_pos_ned[0])
        # 2. Integral (Escalar)
        self.integral_error += error_x * dt
        
        # CORREÇÃO 1: Usar o índice [0] do limite integral (que é um array 3D)
        self.integral_error = np.clip(
            self.integral_error,
            -self.integral_limit, # Usa apenas o limite de X
            self.integral_limit
        )
        
        # 3. Derivativo (Escalar)
        derivative_error = (error_x - self.last_error) / dt
        self.last_error = error_x
        
        # CORREÇÃO 2: Usar o índice [0] dos ganhos (que são arrays 3D)
        # (escalar = escalar*escalar + escalar*escalar + escalar*escalar)
        output_vel_x = (
            self.kp * error_x +          # Usa apenas Kp de X
            self.ki * self.integral_error + # Usa apenas Ki de X
            self.kd * derivative_error    # Usa apenas Kd de X
        )
        print(output_vel_x)
        # --- Construção do Vetor de Saída ---
        
        # Agora 'output_vel_x' é um escalar, e esta linha (406) funcionará.
        output_vel = np.array([
                    output_vel_x[0], # Velocidade controlada pelo PID em X
                    0.0,          # Velocidade Y fixada em zero
                    0.0           # Velocidade Z fixada em zero
                ])
        
        # CORREÇÃO 3: A saturação deve ser aplicada ao VETOR 'output_vel'
        output_norm = np.linalg.norm(output_vel) # Calcula a norma do vetor 3D
        if output_norm > self.max_vel:
            # Satura o vetor 3D, mantendo a direção
            output_vel = output_vel * (self.max_vel / output_norm)
            
        # --- Verificação de Sucesso ---
        error_distance_x = np.linalg.norm(self.setpoint_ned[0] - self.current_pos_ned[0])
        # error_distance_offs = error_distance_x.copy()  # Correção 4: garantir que é um escalar
        # error_distance_offs = error_distance_offs - 2.0
        # self.error_top.publish(Float64(data=error_distance_offs))

        if error_distance_x < self.threshold_distance:
            self.get_logger().info(f"Alvo alcançado! Erro: {error_distance_x:.3f}m")
            self.state = State.IDLE
            self.publish_velocity_setpoint(np.array([0.0, 0.0, 0.0]))
        else:
            self.publish_velocity_setpoint(output_vel)
            self.get_logger().info(
                f"Erro: {error_distance_x:.2f}m | Vel: [{output_vel[0]:.2f}, {output_vel[1]:.2f}, {output_vel[2]:.2f}]",
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