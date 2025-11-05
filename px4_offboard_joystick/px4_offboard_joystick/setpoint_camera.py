#!/usr/bin/env python3
############################################################################
# Controlador PID para Drone (ROS2 + PX4)
# Versão Corrigida e com Lógica de "Latch"
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
from std_msgs.msg import Bool

def get_message_name_version(msg_class):
    if msg_class.MESSAGE_VERSION == 0:
        return ""
    return f"_v{msg_class.MESSAGE_VERSION}"
    
class State(Enum):
    STARTING = 1  # Estabelecendo stream offboard
    IDLE = 2      # Aguardando setpoint
    MOVING = 3    # Movendo para o alvo
    SUCCESS = 4   # Alvo alcançado com sucesso

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
        self.declare_parameter('pos_ned_offset', -1.0)  # Offset em Z para posição NED
        self.declare_parameter('threshold_distance', 0.1)
        self.declare_parameter('threshold_distance_z', 0.4) 

        self.declare_parameter('px4_cmd_topic', '/fmu/in/vehicle_command')
        self.declare_parameter('px4_pos_topic', '/fmu/out/vehicle_local_position_v1')
        self.declare_parameter('offboard_mode_topic', '/fmu/in/offboard_control_mode')
        self.declare_parameter('trajectory_setpoint_topic', '/fmu/in/trajectory_setpoint')

        self.declare_parameter('kp', 0.3)
        self.declare_parameter('ki', 0.1)
        self.declare_parameter('kd', 0.2)
        
        self.declare_parameter('kp_z', 1.0) # Novo
        self.declare_parameter('ki_z', 0.1) # Novo
        self.declare_parameter('kd_z', 0.15) # Novo
        
        self.declare_parameter('max_vel', 0.3)
        self.declare_parameter('max_vel_z', 0.8)  # Vel máxima em Z
        self.declare_parameter('integral_limit', 1.0)
        self.declare_parameter('integral_limit_z', 0.8)
        
        self.declare_parameter('offboard_stream_delay_s', 2.0)
        self.declare_parameter('pos_staleness_threshold_s', 0.5)
        
        self.declare_parameter('success_z_vel', -0.3) # Vel Z (NED) após 10s em SUCCESS
        
        default_tf_matrix = [0., 1., 0., 0., 0., 1., 1., 0., 0.]
        self.declare_parameter('target_to_enu_tf', default_tf_matrix)
        

    def get_parameters(self):
        self.timer_period = self.get_parameter('timer_period').get_parameter_value().double_value
        self.feature_topic = self.get_parameter('feature_topic').get_parameter_value().string_value
        self.threshold_distance = self.get_parameter('threshold_distance').get_parameter_value().double_value
        self.threshold_distance_z = self.get_parameter('threshold_distance_z').get_parameter_value().double_value
        
        self.px4_cmd_topic = self.get_parameter('px4_cmd_topic').get_parameter_value().string_value
        self.px4_pos_topic = self.get_parameter('px4_pos_topic').get_parameter_value().string_value
        self.offboard_mode_topic = self.get_parameter('offboard_mode_topic').get_parameter_value().string_value
        self.trajectory_setpoint_topic = self.get_parameter('trajectory_setpoint_topic').get_parameter_value().string_value

        self.kp = self.get_parameter('kp').get_parameter_value().double_value
        self.ki = self.get_parameter('ki').get_parameter_value().double_value
        self.kd = self.get_parameter('kd').get_parameter_value().double_value
        self.kp_z = self.get_parameter('kp_z').get_parameter_value().double_value # Novo
        self.ki_z = self.get_parameter('ki_z').get_parameter_value().double_value # Novo
        self.kd_z = self.get_parameter('kd_z').get_parameter_value().double_value # Novo
        
        self.max_vel = self.get_parameter('max_vel').get_parameter_value().double_value
        self.max_vel_z = self.get_parameter('max_vel_z').get_parameter_value().double_value # Novo
        self.integral_limit = self.get_parameter('integral_limit').get_parameter_value().double_value
        self.integral_limit_z = self.get_parameter('integral_limit_z').get_parameter_value().double_value # Novo

        self.offboard_stream_delay_s = self.get_parameter('offboard_stream_delay_s').get_parameter_value().double_value
        self.pos_staleness_threshold_s = self.get_parameter('pos_staleness_threshold_s').get_parameter_value().double_value
        self.offboard_cycles_needed = int(self.offboard_stream_delay_s / self.timer_period)
        
        self.success_z_vel = self.get_parameter('success_z_vel').get_parameter_value().double_value
        
        tf_list = self.get_parameter('target_to_enu_tf').get_parameter_value().double_array_value
        self.T_target_to_enu = np.array(tf_list).reshape(3, 3)
        self.get_logger().info(f"Matriz de Transformação (Target -> ENU):\n{self.T_target_to_enu}")

        self.pos_ned_offset = self.get_parameter('pos_ned_offset').get_parameter_value().double_value

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
        self.setpoint_z_ned_pub = self.create_publisher(Float64, "/setpoint_Z_NED", 10)
        self.is_stable = self.create_publisher(Bool, "/is_stable", 10)

        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def init_variables(self):
        self.setpoint_ned = np.array([0.0, 0.0, 0.0])
        self.current_pos_ned = np.array([0.0, 0.0, 0.0])
        self.current_pos_received = False
        self.last_pos_timestamp = None
        
        self.integral_error_x = 0.0
        self.last_error_x = 0.0
        self.integral_error_y = 0.0 # Novo
        self.last_error_y = 0.0 # Novo
        self.integral_error_z = 0.0 # Novo
        self.last_error_z = 0.0 # Novo
        
        self.last_pid_time = self.get_clock().now()
        
        self.offboard_stream_counter = 0
        self.target_coords = None
        self.new_setpoint_received = False
        self.trueYaw = 0.0  #current yaw value of drone
        self.trueRoll = 0.0  #current roll value of drone
        self.truePitch = 0.0  #current pitch value of drone
        
        ### MUDANÇA ###
        # Inicia como False. A FSM (máquina de estados) irá habilitá-la 
        # para True quando estiver pronta (no final do estado STARTING).
        self.is_valid = False
        self.success_counter = 0
        
        ### MUDANÇA: Sub-estado para controle sequencial ###
        self.control_phase = 'XY' # Pode ser 'XY' ou 'Z'
        
        self.success_entry_time = None  # Armazena quando entramos em SUCCESS
        
        self.is_stable_ = False  # Inicia quando entramos em SUCCESS, valida a estabilidade do drone

    # --- CALLBACKS ---
    def visual_info_callback(self, msg):
            """
            Recebe coordenadas da câmera e, SE VÁLIDO, atualiza o setpoint.
            """
            if not msg:
                self.get_logger().warn("Recebida mensagem VisualInfo vazia.", throttle_duration_sec=5.0)
                return

            try:
                # Tenta processar como listas (múltiplos alvos?)
                x_val = np.array(msg.x)
                y_val = np.array(msg.y)
                z_val = np.array(msg.z)
            except (TypeError, ValueError):
                try:
                    # Se falhar, tenta processar como floats únicos
                    self.get_logger().debug("Campos de VisualInfo não são listas. Tentando float único.", throttle_duration_sec=5.0)
                    x_val = np.array([float(msg.x)])
                    y_val = np.array([float(msg.y)])
                    z_val = np.array([float(msg.z)])
                except Exception as e:
                    # Se ambos falharem, loga o erro e aborta
                    self.get_logger().error(f"Erro fatal ao extrair coordenadas: {e}", throttle_duration_sec=5.0)
                    return # Aborta o callback

            # Empilha as coordenadas (agora funciona para 1 ou N pontos)
            target_coords_raw = np.column_stack((x_val, y_val, z_val))

            self.target_coor_z = target_coords_raw[0][2]
            ### MUDANÇA (A CORREÇÃO) ###
            # Validação de Sanidade: Checa se todos os valores recebidos são finitos
            # (ou seja, não são 'inf', '-inf' ou 'nan')
            if not np.all(np.isfinite(target_coords_raw)):
                self.get_logger().warn(
                    f"Dados visuais inválidos (inf ou nan) recebidos. Descartando: {target_coords_raw}",
                    throttle_duration_sec=5.0
                )
                return # Aborta o callback antes de processar dados inválidos

            # --- LÓGICA DE LATCH (TRAVA) ---
            # Só calcula se a flag de validade (controlada pela FSM) for True
            if self.is_valid:
                # Agora sabemos que os dados são seguros para usar
                self.target_coords = target_coords_raw
                
                # Chama a função de transformação
                self.setpoint_ned = self.transform_target_to_ned_setpoint(self.target_coords)

                # 1. Sinaliza para a FSM que há um novo setpoint
                self.new_setpoint_received = True
                
                # 2. Fecha a trava para não processar novas mensagens
                self.is_valid = False 
                self.get_logger().info("Novo setpoint recebido, 'travando' o recebimento de novos dados.")
            
            else:
                # Loga que recebeu dados mas ignorou (pois a trava está fechada)
                self.get_logger().debug("Dados visuais recebidos, mas o sistema está 'inválido' (aguardando conclusão do movimento).", throttle_duration_sec=10.0)
     
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
        t2 = np.clip(t2, -1.0, 1.0) # Evita erros de domínio no arcsin
        self.truePitch = np.arcsin(t2)
        
        t3 = +2.0*(orientation_q[0]*orientation_q[3] + orientation_q[1]*orientation_q[2])
        t4 = +2.0*(orientation_q[0]*orientation_q[0] + orientation_q[1]*orientation_q[1] - orientation_q[2]*orientation_q[2] - orientation_q[3]*orientation_q[3])
        
        ### CORREÇÃO 1: Remover a negação do Yaw ###
        # A matriz de rotação ZYX padrão espera um yaw CCW (right-hand rule),
        # que o np.arctan2(t3, t4) já fornece. A negação inverte a rotação.
        self.trueYaw = np.arctan2(t3, t4)
        
    # --- TRANSFORMAÇÕES ---
    # (Nenhuma mudança nesta seção)
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
        
    def get_rotation_matrix_body_to_ned(self, roll_rad, pitch_rad, yaw_rad):
        """
        Retorna matriz de rotação do frame body FRD para mundo NED
        usando convenção Tait-Bryan ZYX (yaw-pitch-roll)
        
        Valores de entrada DEVEM estar em RADIANOS.
        """
        
        ### CORREÇÃO 2: Remover conversão de deg2rad ###
        # Os valores de trueRoll/Pitch/Yaw já estão em radianos.
        roll = roll_rad
        pitch = pitch_rad
        yaw = yaw_rad
        
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

    def get_rotation_matrix_ned_to_body(self, roll_rad, pitch_rad, yaw_rad):
        R_body_to_ned = self.get_rotation_matrix_body_to_ned(roll_rad, pitch_rad, yaw_rad)
        return R_body_to_ned.T

    def transform_target_to_ned_setpoint(self,
                                        target_vec,
                                        apply_offset=True):
        
        # Pega o primeiro alvo da lista
        primeiro_alvo = target_vec[0]
        
        # print("\n--- TRANSFORMAÇÃO: Deslocamento Mundo → Setpoint ---")
        # print(f"    target_vec (deslocamento ENU)={primeiro_alvo}")

        # 1) Target é o vetor deslocamento em ENU mundo
        displacement_enu_old = np.array(primeiro_alvo, dtype=np.float64).reshape(3)
        displacement_enu = np.zeros(3, dtype=np.float64)
        displacement_enu[0] = displacement_enu_old[1]
        displacement_enu[1] = -1.0 * displacement_enu_old[0]
        displacement_enu[2] = displacement_enu[2]
        # self.get_logger().info(f"1) Deslocamento em ENU mundo: {displacement_enu}")

        # 2) Posição atual do robô em NED
        robot_pos_ned = np.array([
            float(self.current_pos_ned[0]),
            float(self.current_pos_ned[1]),
            float(self.current_pos_ned[2])
        ], dtype=np.float64)
        # self.get_logger().info(f"2) Robô em NED: {robot_pos_ned}")
        R_body_to_ned = self.get_rotation_matrix_body_to_ned(
            self.trueRoll, 
            self.truePitch, 
            self.trueYaw
        )
        displacement_world_ned = R_body_to_ned @ displacement_enu
        # 3) Converter o deslocamento de ENU para NED
        displacement_ned = self.enu_to_ned(displacement_world_ned)
        # self.get_logger().info(f"3) Deslocamento ENU → NED: {displacement_ned}")


        # Soma a posição do robô
        displacement_world_ned[2] = -displacement_world_ned[2]  # Z já está em NED
        target_position_ned = displacement_world_ned + robot_pos_ned
        
        # 4) Calcular posição do objeto (setpoint final)
        # target_position_ned = displacement_ned + robot_pos_ned
        # self.get_logger().info(f"4) Posição do objeto (setpoint): {target_position_ned}")

        # 5) Aplicar offset se necessário
        if apply_offset:
            # Esta parte pode falhar se apply_offset=True
            offset_ned = np.array([
                0.0,
                0.0,
                0.0
            ], dtype=np.float64)
            target_position_ned = target_position_ned + offset_ned
            # self.get_logger().info(f"5) Com offset: {target_position_ned}")

        self.get_logger().info(f"✅ SETPOINT FINAL (NED): {target_position_ned}\n")
        return target_position_ned
    
    # --- MÁQUINA DE ESTADOS ---
    def timer_callback(self):
        self.publish_offboard_mode()
        published_stp = Float64()
        published_stp.data = self.setpoint_ned[2] 
        self.setpoint_z_ned_pub.publish(published_stp)
        # self.get_logger().info(f"Estado atual: {self.state.name}")
        if self.state == State.STARTING:
            self.run_state_starting()
        elif self.state == State.IDLE:
            self.run_state_idle()
        elif self.state == State.MOVING:
             self.run_state_moving()
        elif self.state == State.SUCCESS:
            self.run_state_success()

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
            
            ### MUDANÇA ###
            # Lógica de Latch (1): Abre a trava pela primeira vez.
            # O sistema agora está pronto para aceitar um setpoint da câmera.
            self.is_valid = True
            self.get_logger().info(f"Transição: STARTING -> {self.state.name}")
            self.get_logger().info("Pronto para receber o primeiro setpoint. 'Destravando' dados visuais.")


    def run_state_idle(self):
        """Aguarda novo setpoint com posição válida."""
        # Mantém o drone parado enquanto espera
        self.publish_velocity_setpoint(np.array([0.0, 0.0, 0.0]))
        
        # self.new_setpoint_received é definido como True pelo visual_info_callback
        if self.new_setpoint_received:
            if self.is_position_fresh():
                self.get_logger().info(
                    f"Novo alvo (NED): [{self.setpoint_ned[0]:.2f}, "
                    f"{self.setpoint_ned[1]:.2f}, {self.setpoint_ned[2]:.2f}]"
                )
                self.reset_pid_controller()
                self.state = State.MOVING
                
                ### MUDANÇA ###
                # Consome o "sinal"
                self.new_setpoint_received = False 
                self.get_logger().info(f"Transição: IDLE -> {self.state.name}")
            else:
                self.get_logger().warn(
                    "Setpoint recebido mas posição do drone inválida. Aguardando...",
                    throttle_duration_sec=2.0
                )

    def run_state_moving(self):
        """
        Executa controle PID sequencial:
        1. Fase 'XY': Move em X e Y até atingir o threshold.
        2. Fase 'Z': Move em Z até atingir o threshold.
        """
        if not self.is_position_fresh():
            self.get_logger().error("Posição antiga! Revertendo para IDLE.")
            self.state = State.IDLE
            self.publish_velocity_setpoint(np.array([0.0, 0.0, 0.0]))
            
            self.success_counter = 0
            self.control_phase = 'XY' # Reseta a fase
            self.get_logger().warn("Contagem de sucessos resetada devido à posição antiga.")
            return

        # Calcula dt
        current_time = self.get_clock().now()
        dt_duration = current_time - self.last_pid_time
        dt = dt_duration.nanoseconds / 1e9
        self.last_pid_time = current_time
        
        if dt < (self.timer_period * 0.1):
            return

        # --- Calcula todos os erros ---
        error_x = (self.setpoint_ned[0] - self.current_pos_ned[0])
        error_y = (self.setpoint_ned[1] - self.current_pos_ned[1])
        # error_z = (self.setpoint_ned[2] - self.current_pos_ned[2])
        
        error_vec_xy = np.array([error_x, error_y])
        current_error_distance_xy = np.linalg.norm(error_vec_xy)
        current_error_distance_z = self.target_coor_z


        # ==========================================================
        # FASE 1: Controle X/Y
        # ==========================================================
        if self.control_phase == 'XY':
            
            if current_error_distance_xy < self.threshold_distance:
                # --- SUCESSO EM XY ---
                self.get_logger().info(f"FASE 1 (XY) CONCLUÍDA. Erro 2D: {current_error_distance_xy:.3f}m")
                
                ### MUDANÇA (Conforme solicitado): Log e transição de fase ###
                self.get_logger().info("!!! Zerando velocidades X/Y. Iniciando controle de altitude (Z).")
                
                self.publish_velocity_setpoint(np.array([0.0, 0.0, 0.0]))
                
                # Reseta o PID de Z para uma partida limpa
                self.integral_error_z = 0.0
                self.last_error_z = 0.0
                self.success_counter += 1
                self.get_logger().info(f"Ciclo de sucesso em XY: {self.success_counter}/10")
                if self.success_counter >= 10:
                    # CONDIÇÃO DE SUCESSO FINAL ATINGIDA
                    self.get_logger().info("10 sucessos consecutivos alcançados. Transição para Z.")
                    # Muda a fase
                    self.control_phase = 'Z' 
                    self.success_counter = 0 # Reseta contador para fase Z
                else:   
                    # SUCESSO, MAS AINDA NÃO TERMINOU
                    self.state = State.IDLE
                    self.is_valid = True
                    self.get_logger().info(f"Transição: MOVING -> {self.state.name}")
                    self.get_logger().info("Movimento concluído. 'Destravando' para receber próximo setpoint.")
                         
                return # Espera o próximo ciclo para iniciar o controle de Z

            else:
                # --- AINDA SE MOVENDO EM XY ---
                if self.success_counter > 0: # Perdeu o alvo
                    self.get_logger().warn(f"Alvo XY perdido (Erro 2D: {current_error_distance_xy:.3f}m). Resetando contador.")
                    self.success_counter = 0

                # --- Controlador PID (X e Y) ---
                # (Correção: Usando ganhos por eixo _x e _y que corrigi anteriormente)
                self.integral_error_x = np.clip(self.integral_error_x + error_x * dt, -self.integral_limit, self.integral_limit)
                derivative_error_x = (error_x - self.last_error_x) / dt
                self.last_error_x = error_x
                output_vel_x = (self.kp * error_x + self.ki * self.integral_error_x + self.kd * derivative_error_x)

                self.integral_error_y = np.clip(self.integral_error_y + error_y * dt, -self.integral_limit, self.integral_limit)
                derivative_error_y = (error_y - self.last_error_y) / dt
                self.last_error_y = error_y
                output_vel_y = (self.kp * error_y + self.ki * self.integral_error_y + self.kd * derivative_error_y)
                
                # --- Saída de Velocidade (XY) ---
                output_vel = np.array([output_vel_x, output_vel_y, 0.0]) # Z é zero
                
                output_norm = np.linalg.norm(output_vel)
                if output_norm > self.max_vel: # Saturação XY
                    output_vel = output_vel * (self.max_vel / output_norm)

                self.publish_velocity_setpoint(output_vel)
                self.get_logger().info(
                    f"Fase XY - Erro 2D: {current_error_distance_xy:.2f}m | Vel: [{output_vel[0]:.2f}, {output_vel[1]:.2f}, 0.00]",
                    throttle_duration_sec=0.5
                )

        # ==========================================================
        # FASE 2: Controle Z
        # ==========================================================
        elif self.control_phase == 'Z':
            self.get_logger().info(f"Erro Atual Z: {current_error_distance_z:.3f}m")
            self.get_logger().info(f" setpoint Z em NED: {self.setpoint_ned[2]:.2f}, pos Z em NED: {self.current_pos_ned[2]:.2f}, offset: {self.pos_ned_offset:.2f}")

            if abs(current_error_distance_z) < abs(self.threshold_distance_z):
                # --- SUCESSO FINAL (3D) ---
                self.get_logger().info(f"FASE 2 (Z) CONCLUÍDA. Erro Z: {current_error_distance_z:.3f}m.")
                self.publish_velocity_setpoint(np.array([0.0, 0.0, 0.0])) # Para
                self.success_counter += 1
                self.state = State.SUCCESS
                self.get_logger().info(f"Transição: MOVING -> {self.state.name}")
                return

            else:
                # --- AINDA SE MOVENDO EM Z ---
                if self.success_counter > 0: # Perdeu o alvo Z
                    self.get_logger().warn(f"Alvo Z perdido (Erro: {current_error_distance_z:.3f}m). Resetando contador.")
                    self.success_counter = 0
                
                # --- Controlador PID (Z) ---
                self.integral_error_z = np.clip(self.integral_error_z + current_error_distance_z * dt, -self.integral_limit_z, self.integral_limit_z)
                derivative_error_z = (current_error_distance_z - self.last_error_z) / dt
                self.last_error_z = current_error_distance_z
                output_vel_z = (self.kp_z * current_error_distance_z + self.ki_z * self.integral_error_z + self.kd_z * derivative_error_z)

                # --- Saída de Velocidade (Z) ---
                output_vel = np.array([0.0, 0.0, output_vel_z]) # X e Y são zero
                # Inverte o sinal para NED
                
                # Saturação Z
                if abs(output_vel[2]) > self.max_vel_z:
                    output_vel[2] = np.sign(output_vel[2]) * self.max_vel_z
                
                self.publish_velocity_setpoint(output_vel)
                self.get_logger().info(
                    f"Fase Z - Erro Z: {current_error_distance_z:.2f}m | Vel: [0.00, 0.00, {output_vel[2]:.2f}]",
                    throttle_duration_sec=0.5
                )
                
    def run_state_success(self):
        """
        Estado final. Aguarda 10s, depois aplica velocidade Z constante.
        Não aceita mais setpoints.
        """
        self.is_valid = False # Impede novos setpoints

        # 1. Armazena o tempo de entrada na primeira vez que esta função é chamada
        if self.success_entry_time is None:
            self.success_entry_time = self.get_clock().now()
            self.get_logger().info(
                "Estado SUCCESS: Alvo final alcançado. Iniciando contagem de 10 segundos..."
            )
            
        # 2. Calcula o tempo decorrido
        elapsed_time = self.get_clock().now() - self.success_entry_time
        elapsed_seconds = elapsed_time.nanoseconds / 1e9

        # 3. Lógica
        if elapsed_seconds < 10.0:
            # AINDA AGUARDANDO: Mantém o drone parado
            self.publish_velocity_setpoint(np.array([0.0, 0.0, 0.0]))
            self.get_logger().info(
                f"Estado SUCCESS: Aguardando... {elapsed_seconds:.1f}/10.0s",
                throttle_duration_sec=1.0 # Loga apenas a cada 1s para não poluir
            )
        elif elapsed_seconds > 10 and elapsed_seconds < 15:
            # 10 SEGUNDOS PASSARAM: Aplica velocidade Z
            self.get_logger().info(
                f"Estado SUCCESS: 10 segundos concluídos. Aplicando velocidade Z: {self.success_z_vel} m/s",
                throttle_duration_sec=5.0 # Loga a cada 5s
            )
            
            # (Lembre-se: em NED, velocidade Z positiva é PARA BAIXO)
            self.publish_velocity_setpoint(np.array([0.0, 0.0, self.success_z_vel]))
        else:
            # Garante que o drone recebeu o comando de parar
            self.publish_velocity_setpoint(np.array([0.0, 0.0, 0.0]))

            # Defina seu limiar de instabilidade em GRAUS
            stability_threshold_degrees = 7.0 
            stability_threshold_radians = np.deg2rad(stability_threshold_degrees)

            verify_stability = Bool()

            # Pega o valor absoluto (módulo) do pitch e roll atuais
            current_pitch_rad = abs(self.truePitch)
            current_roll_rad = abs(self.trueRoll)

            # Verifica se QUALQUER um deles ultrapassou o limiar
            if (current_pitch_rad > stability_threshold_radians) or (current_roll_rad > stability_threshold_radians):
                # Drone está INSTÁVEL
                self.get_logger().warn(
                    f"INSTÁVEL! Pitch: {np.rad2deg(current_pitch_rad):.2f}°, Roll: {np.rad2deg(current_roll_rad):.2f}°",
                    throttle_duration_sec=1.0 
                )
                self.is_stable_ = False
                verify_stability.data = self.is_stable_
                self.is_stable.publish(verify_stability)
            else:
                # Drone está ESTÁVEL
                self.get_logger().info(
                    f"Estável. Pitch: {np.rad2deg(current_pitch_rad):.2f}°, Roll: {np.rad2deg(current_roll_rad):.2f}°",
                    throttle_duration_sec=1.0 
                )
                self.is_stable_ = True
                verify_stability.data = self.is_stable_
                self.is_stable.publish(verify_stability)
            
        
    def reset_pid_controller(self):
        """Reseta estado do PID."""
        self.integral_error_x = 0.0
        self.last_error_x = 0.0
        self.integral_error_y = 0.0 # Novo
        self.last_error_y = 0.0 # Novo
        self.integral_error_z = 0.0 # Novo
        self.last_error_z = 0.0 # Novo
        
        self.last_pid_time = self.get_clock().now()
        
        ### MUDANÇA: Reseta a fase de controle ###
        self.control_phase = 'XY'

    def is_position_fresh(self):
        if not self.current_pos_received or self.last_pos_timestamp is None:
            self.get_logger().warn("Nenhuma posição recebida ainda!", throttle_duration_sec=2.0)
            return False
            
        elapsed = self.get_clock().now() - self.last_pos_timestamp
        elapsed_sec = elapsed.nanoseconds / 1e9
        
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