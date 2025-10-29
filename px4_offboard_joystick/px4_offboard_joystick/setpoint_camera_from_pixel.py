#!/usr/bin/env python3
############################################################################

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.duration import Duration
from std_msgs.msg import Float64MultiArray
from enum import Enum

from ros_rl_interfaces.msg import VisualInfo
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleLocalPosition

def get_message_name_version(msg_class):
    if msg_class.MESSAGE_VERSION == 0:
        return ""
    return f"_v{msg_class.MESSAGE_VERSION}"
    
# Enumeração para nossa Máquina de Estados
class State(Enum):
    STARTING = 1  # Iniciando, estabelecendo stream offboard
    IDLE = 2      # Armado, em offboard, aguardando setpoint
    MOVING = 3    # PID ativo, movendo-se para o setpoint
    SUCCESS = 4   # Alvo alcançado, mantendo posição

class SetpointCamera(Node):
  def __init__(self):
    """Inicializa o nó."""
    super().__init__('setpoint_camera_pixel_controller')
    
    self.state = State.STARTING # Inicializa a máquina de estados
    
    self.init_parameters()
    self.get_parameters()
    self.init_ros_interfaces()
    self.init_variables()
    self.get_logger().info("Nó de controle PID iniciado.")
    self.get_logger().info(f"Estado inicial: {self.state.name}")

  def init_parameters(self):
    """Declara os parâmetros ROS 2."""
    self.declare_parameter('timer_period', 0.05)
    self.declare_parameter('feature_topic', '/pose_feature')
    self.declare_parameter('offset_z_distance', 0.5)
    self.declare_parameter('threshold_distance', 1.5) 

    # Tópicos do PX4
    self.declare_parameter('px4_cmd_topic', '/fmu/in/vehicle_command')
    self.declare_parameter('px4_pos_topic', '/fmu/out/vehicle_local_position_v1')
    self.declare_parameter('offboard_mode_topic', '/fmu/in/offboard_control_mode')
    self.declare_parameter('trajectory_setpoint_topic', '/fmu/in/trajectory_setpoint')

    # Parâmetros do Controlador PID
    self.declare_parameter('kp', 1.0) 
    self.declare_parameter('ki', 0.1) 
    self.declare_parameter('kd', 0.5) 
    self.declare_parameter('max_vel', 0.5) 
    self.declare_parameter('integral_limit', 1.0) 

    # Parâmetros de Robustez e Configuração
    # (Usado para o streaming antes de *comandar* o offboard)
    self.declare_parameter('offboard_stream_delay_s', 2.0) 
    self.declare_parameter('pos_staleness_threshold_s', 0.5)
    
    # Matriz de transformação (Ex: Cam_FLU -> Global_ENU) como uma lista
    # Padrão: [Y -> X, Z -> Y, X -> Z] 
    default_tf_matrix = [0., 1., 0., 0., 0., 1., 1., 0., 0.]
    self.declare_parameter('target_to_enu_tf', default_tf_matrix)

  def get_parameters(self):
      """Obtém os valores dos parâmetros ROS 2."""
      self.timer_period = self.get_parameter('timer_period').get_parameter_value().double_value
      self.feature_topic = self.get_parameter('feature_topic').get_parameter_value().string_value
      self.offset_z_distance = self.get_parameter('offset_z_distance').get_parameter_value().double_value
      self.threshold_distance = self.get_parameter('threshold_distance').get_parameter_value().double_value
      
      # Tópicos PX4
      self.px4_cmd_topic = self.get_parameter('px4_cmd_topic').get_parameter_value().string_value
      self.px4_pos_topic = self.get_parameter('px4_pos_topic').get_parameter_value().string_value
      self.offboard_mode_topic = self.get_parameter('offboard_mode_topic').get_parameter_value().string_value

      # Ganhos PID
      self.kp = self.get_parameter('kp').get_parameter_value().double_value
      self.ki = self.get_parameter('ki').get_parameter_value().double_value
      self.kd = self.get_parameter('kd').get_parameter_value().double_value
      self.max_vel = self.get_parameter('max_vel').get_parameter_value().double_value
      self.integral_limit = self.get_parameter('integral_limit').get_parameter_value().double_value

      # Robustez
      self.offboard_stream_delay_s = self.get_parameter('offboard_stream_delay_s').get_parameter_value().double_value
      self.pos_staleness_threshold_s = self.get_parameter('pos_staleness_threshold_s').get_parameter_value().double_value
      self.offboard_cycles_needed = int(self.offboard_stream_delay_s / self.timer_period)
      
      # Carrega a matriz de transformação
      tf_list = self.get_parameter('target_to_enu_tf').get_parameter_value().double_array_value
      self.T_target_to_enu = np.array(tf_list).reshape(3, 3)
      self.get_logger().info(f"Matriz de Transformação (Target -> ENU) carregada:\n{self.T_target_to_enu}")

  def init_ros_interfaces(self):
      """Inicializa os publishers, subscribers e timers."""
      qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
            )
      trajectorySP = f"/fmu/in/trajectory_setpoint{get_message_name_version(TrajectorySetpoint)}"
      vehicle_command = f"/fmu/in/vehicle_command{get_message_name_version(VehicleCommand)}"
      
      # Publishers
      self.cmd_pub = self.create_publisher(VehicleCommand, vehicle_command, 10)
      self.offboard_control_pub = self.create_publisher(OffboardControlMode, self.offboard_mode_topic, 10)
      self.trajectory_setpoint_pub = self.create_publisher(TrajectorySetpoint, trajectorySP, 10)

      # Subscribers
      self.pos_sub = self.create_subscription(
        VehicleLocalPosition, self.px4_pos_topic, self.px4_pos_callback, qos_profile)
      self.feature_sub = self.create_subscription(
        VisualInfo, self.feature_topic, self.visual_info_callback, 10)
      
      # Timer (Loop de controle principal)
      self.timer = self.create_timer(self.timer_period, self.timer_callback)

  def init_variables(self):
      """Inicializa as variáveis internas do nó."""
      # Setpoint (Alvo em NED)
      self.setpoint_ned = np.array([0.0, 0.0, 0.0])
      # Posição Atual (NED)
      self.current_pos_ned = np.array([0.0, 0.0, 0.0])
      self.current_pos_received = False
      self.last_pos_timestamp = None  
      # Estado do PID
      self.integral_error = np.array([0.0, 0.0, 0.0])
      self.last_error = np.array([0.0, 0.0, 0.0])
      self.last_pid_time = None
      # Contador para streaming offboard
      self.offboard_stream_counter = 0
      self.target_coords = None # Inicializa as coordenadas do alvo como None

  def visual_info_callback(self, msg):
      """Callback para processar a mensagem VisualInfo e definir o setpoint."""
      if not msg: 
          return
          
      # 1. Extrair coordenadas. 
      # Esta lógica lida com campos que são escalares (float) ou listas/tuplas (pegando o [0]).
      try:
        self.point_u = Float64MultiArray()
        self.point_v = Float64MultiArray()
        self.nose_u = np.array(msg.u)
        self.point_v = np.array(msg.v)
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
            
      except (TypeError, IndexError):
          # Fallback se não for iterável (ou seja, se for apenas um escalar "5.0" ou 5.0)
          self.get_logger().warn("Campos da VisualInfo não parecem ser listas. Tentando conversão direta.", throttle_duration_sec=5.0)

      self.target_coords = np.array(qpoint[0], qpoint[1], qpoint[2])
      # Salva o estado antigo para referência
      old_state = self.state
      
      # 2. Executar a lógica de estado solicitada
      
      # Regra 1: "se está no starting, espera o idle está correto"
      if self.state == State.STARTING:
          # Ação: Apenas registrar o setpoint, mas não mudar o estado.
          # O drone começará a se mover quando o loop principal (timer)
          # finalmente transicionar de STARTING para IDLE.
          
        # (Calcula a transformação e salva o setpoint)
        self.setpoint_ned = self.transform_target_to_ned_setpoint(self.target_coords)            
        self.get_logger().warn(
            f"Setpoint recebido ({self.setpoint_ned[0]:.2f}, {self.setpoint_ned[1]:.2f}, {self.setpoint_ned[2]:.2f}), "
            f"mas o drone ainda está em {self.state.name}. Aguardando IDLE para iniciar."
        )

      # Regra 2: "do idle, checa se há posições válidas, senão mantem no idle"
      elif self.state == State.IDLE:
          # Checa se a posição ATUAL do drone é válida (recente)
          if self.is_position_received() or old_state != State.SUCCESS:
              # Calcula a transformação
              self.setpoint_ned = self.transform_target_to_ned_setpoint(self.target_coords)

              # Reseta o controlador PID
              self.reset_pid_controller()
              
              # Entra em moving
              self.state = State.MOVING
              
              self.get_logger().info(
                  f"Novo Setpoint (NED): [{self.setpoint_ned[0]:.2f}, {self.setpoint_ned[1]:.2f}, {self.setpoint_ned[2]:.2f}]. "
                  f"Posição do drone válida. Transicionando de {old_state.name} para {self.state.name}."
              )
          else:
              # Posição inválida (drone), mantém no IDLE
              self.state = State.IDLE # Redundante, mas explícito
              self.get_logger().info(
                  "Novo setpoint recebido, mas a posição do drone está antiga. "
                  f"Permanecendo em {self.state.name}. O setpoint foi ignorado."
              )

      # Lógica implícita para MOVING (se receber um novo ponto enquanto já se move)
      elif self.state == State.MOVING:
          # Se já estamos em MOVING, tratamos como no estado IDLE:
          # checamos a posição do drone antes de aceitar o novo setpoint.

          if self.is_position_received() and old_state != State.SUCCESS:
              # Posição válida: Aceita o novo ponto.
              
              # Calcula a transformação
              self.target_coords = np.array([x_val, y_val, z_val])
              self.setpoint_ned = self.transform_target_to_ned_setpoint(self.target_coords)
              
              # Permanece em MOVING
              self.state = State.MOVING
              
              self.get_logger().info(
                  f"Setpoint ATUALIZADO (NED): [{self.setpoint_ned[0]:.2f}, {self.setpoint_ned[1]:.2f}, {self.setpoint_ned[2]:.2f}]. "
                  f"Posição do drone válida. Permanecendo em {self.state.name}."
              )
          elif self.state == State.SUCCESS:
                self.reset_pid_controller()
                # Ação: Mudar para IDLE.
                # NOTA: Com esta regra, o novo ponto (x, y, z) é IGNORADO.
                # O drone vai para IDLE e espera outro comando.
                self.state = State.IDLE
                self.get_logger().info(
                    f"Novo ponto recebido enquanto em {old_state.name}. "
                    f"Transicionando para {self.state.name} conforme regra. O novo ponto foi ignorado."
              )
          else:
              # Posição inválida (drone): Reverte para IDLE por segurança
              self.state = State.IDLE
              self.get_logger().error(
                  "Setpoint atualizado recebido, mas a posição do drone está antiga. "
                  f"Revertendo para {self.state.name} por segurança. O setpoint foi ignorado."
              )
          
  def px4_pos_callback(self, msg):
      """Callback para atualizar a posição local atual do drone (frame NED)."""
      self.current_pos_ned = np.array([msg.x, msg.y, -msg.z])
      self.get_logger().debug(f"Posição Atual Recebida (NED): ({self.current_pos_ned[0]:.2f}, {self.current_pos_ned[1]:.2f}, {self.current_pos_ned[2]:.2f})")
      self.current_pos_received = True
      self.last_pos_timestamp = self.get_clock().now()

  # --- Funções da Máquina de Estados (Chamadas pelo Timer) ---

  def timer_callback(self):
      """Loop principal de controle (FSM)."""

      # Ação 1: Publicar modo offboard (obrigatório pelo PX4)
      self.publish_offboard_mode()

      # Ação 2: Executar a lógica do estado atual
      if self.state == State.STARTING:
          self.run_state_starting()
      elif self.state == State.IDLE:
          self.run_state_idle()
      elif self.state == State.MOVING:
          self.run_state_moving()
      elif self.state == State.SUCCESS:
          self.run_state_success()

  # --- NOVA FUNÇÃO DE ESTADO ---
  def run_state_starting(self):
      """Estado de inicialização: envia setpoints vazios até estar pronto para comandar o Offboard."""
      # Envia setpoints nulos para estabelecer o stream
      self.publish_velocity_setpoint(np.array([0.0, 0.0, 0.0])) 

      if self.offboard_stream_counter == self.offboard_cycles_needed:
          # Não Armamos (já está no ar)
          # Apenas comandamos o modo Offboard
          
          self.get_logger().info("Enviando comando para Modo Offboard...")
          self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 
                                       param1=1.0, # 1.0 = Modo customizado
                                       param2=6.0) # 6.0 = PX4_CUSTOM_MAIN_MODE_OFFBOARD
          
          # Transição para o próximo estado
          self.state = State.IDLE
          self.get_logger().info(f"Transicionando para: {self.state.name}")
      
      elif self.offboard_stream_counter < self.offboard_cycles_needed:
          self.offboard_stream_counter += 1
          if self.offboard_stream_counter % 20 == 0: # Loga a cada segundo
               self.get_logger().info("Iniciando stream de setpoints para habilitar Offboard...")

  def run_state_idle(self):
      """Estado Ocioso: armado, em offboard, mantendo posição, aguardando setpoint."""
      # O setpoint pode ter sido recebido *durante* o STARTING. 
      # Se `visual_info_callback` já mudou o estado para MOVING, este estado é pulado.
      self.publish_velocity_setpoint(np.array([0.0, 0.0, 0.0]))

  def run_state_moving(self):
      """Estado de Movimento: Executa o loop do PID."""
      
      # Verificação de Robustez: A posição ainda é válida?
      # if not self.is_position_fresh():
      #     self.get_logger().error("Feedback de Posição ANTIGO! Revertendo para IDLE e parando.")
      #     self.state = State.IDLE
      #     self.publish_velocity_setpoint(np.array([0.0, 0.0, 0.0]))
      #     return

      # --- Lógica do Controlador PID ---
      current_time = self.get_clock().now()
      dt = (current_time - self.last_pid_time).nanoseconds / 1e9
      self.last_pid_time = current_time
      
      if dt <= 0.0:
          self.get_logger().warn("Delta time (dt) é zero, pulando ciclo PID.")
          return

      # 1. Erro (Termo Proporcional)
      error = self.setpoint_ned - self.current_pos_ned
      
      # 2. Termo Integral (com Anti-Windup)
      self.integral_error += error * dt
      self.integral_error = np.clip(self.integral_error, -self.integral_limit, self.integral_limit)
      
      # 3. Termo Derivativo
      derivative_error = (error - self.last_error) / dt
      self.last_error = error
      
      # 4. Cálculo da Saída (Velocidade)
      output_vel = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * derivative_error)
      
      # 5. Limitar a velocidade máxima de saída
      output_norm = np.linalg.norm(output_vel)
      if output_norm > self.max_vel:
          output_vel = output_vel * (self.max_vel / output_norm)

      # --- Verificação de Sucesso ---
      self.get_logger().debug(f"Erro Atual: {error}, Velocidade de Saída: {output_vel}")
      error_distance = np.linalg.norm(error)
      
      if error_distance < self.threshold_distance:
          self.get_logger().info(f"SUCESSO! Alvo alcançado. Erro: {error_distance:.3f} m")
          self.state = State.SUCCESS
      else:
          # Publica a velocidade calculada
          self.publish_velocity_setpoint(output_vel)
          self.get_logger().info(f"Movendo... Erro: {error_distance:.2f} m", throttle_duration_sec=0.5)

  def run_state_success(self):
      """Estado de Sucesso: Mantém a posição no alvo."""
      self.publish_velocity_setpoint(np.array([0.0, 0.0, 0.0]))
      self.state = State.IDLE
  # --- Funções Auxiliares (Helpers) ---

  def transform_target_to_ned_setpoint(self, target_coords_cam):
      
      # 1. Transformar para ENU (East-North-Up)
      global_coords_enu = self.T_target_to_enu @ target_coords_cam
      
      # 2. Converter ENU para NED (North-East-Down) para o PX4
      setpoint_ned = np.array([
          global_coords_enu[0],                   # ENU North  -> NED X
          global_coords_enu[2],                   # ENU East   -> NED Y
          (global_coords_enu[1]) # ENU Up     -> NED Down (com offset)
      ])
      
      return setpoint_ned
      
  def reset_pid_controller(self):
      """Reseta as variáveis de estado do PID."""
      self.integral_error = np.array([0.0, 0.0, 0.0])
      self.last_error = np.array([0.0, 0.0, 0.0])
      self.last_pid_time = self.get_clock().now()

  def is_position_fresh(self):
      """Verifica se o último feedback de posição é recente."""
      if not self.current_pos_received or self.last_pos_timestamp is None:
          return False
          
      elapsed_time = self.get_clock().now() - self.last_pos_timestamp
      return elapsed_time < Duration(seconds=self.pos_staleness_threshold_s)

  def is_position_received(self):
      """Verifica se a posição atual já foi recebida ao menos uma vez."""
      return self.target_coords is not None

  # --- Funções de Publicação ---
  def publish_offboard_mode(self):
      """Publica a mensagem OffboardControlMode (necessária para manter o modo)."""
      msg = OffboardControlMode()
      msg.position = False
      msg.velocity = True # Estamos enviando comandos de velocidade
      msg.acceleration = False
      msg.attitude = False
      msg.body_rate = False
      msg.timestamp = int(self.get_clock().now().nanoseconds / 1000) # (microseconds)
      self.offboard_control_pub.publish(msg)

  def publish_velocity_setpoint(self, velocity: np.ndarray):
      """Publica a mensagem TrajectorySetpoint com a velocidade desejada (NED)."""
      msg = TrajectorySetpoint()
      msg.velocity = [float(velocity[0]), float(velocity[1]), float(velocity[2])]
      # Yaw é ignorado (NaN) por padrão, o PX4 manterá o yaw atual
      msg.yaw = np.nan 
      msg.timestamp = int(self.get_clock().now().nanoseconds / 1000) # (microseconds)
      self.trajectory_setpoint_pub.publish(msg)

  def publish_position_setpoint(self, position: np.ndarray):
      """Publica a mensagem TrajectorySetpoint com a posição desejada (NED)."""
      msg = TrajectorySetpoint()
      msg.position = [float(position[0]), float(position[1]), float(position[2])]
      # Yaw é ignorado (NaN) por padrão, o PX4 manterá o yaw atual
      msg.yaw = np.nan 
      msg.timestamp = int(self.get_clock().now().nanoseconds / 1000) # (microseconds)
      self.trajectory_setpoint_pub.publish(msg)

  def publish_vehicle_command(self, command, **params):
      """Publica um VehicleCommand para o PX4."""
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

# --- Função Main ---
def main(args=None):
    rclpy.init(args=args)
    controller_node = SetpointCamera()
    try:
        rclpy.spin(controller_node)
    except KeyboardInterrupt:
        controller_node.get_logger().info("Nó interrompido pelo usuário.")
    except Exception as e:
        controller_node.get_logger().error(f"Erro inesperado: {e}")
        raise(e)
    finally:
        controller_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()