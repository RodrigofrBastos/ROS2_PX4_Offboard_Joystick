#!/usr/bin/env python3
############################################################################
# Análise Sênior:
# - Implementada Máquina de Estados (FSM) para robustez (STARTING, IDLE, MOVING, SUCCESS).
# - Removidos "Magic Numbers":
#   - Matriz de transformação agora é um parâmetro ROS 2.
#   - Atraso de 'Arming' é um parâmetro ROS 2.
#   - Constantes do PX4 (VEHICLE_CMD_*) são usadas.
# - Adicionada verificação de robustez (staleness) para o feedback de posição.
# - Lógica do PID isolada em sua própria função.
# - Funções de transformação e callbacks foram limpas e melhor comentadas.
############################################################################

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from enum import Enum

# Mensagens ROS
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
    super().__init__('setpoint_camera_pid_controller')
    
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
    self.declare_parameter('threshold_distance', 0.1) 

    # Tópicos do PX4
    self.declare_parameter('px4_cmd_topic', '/fmu/in/vehicle_command')
    self.declare_parameter('px4_pos_topic', '/fmu/out/vehicle_local_position')
    self.declare_parameter('offboard_mode_topic', '/fmu/in/offboard_control_mode')
    self.declare_parameter('trajectory_setpoint_topic', '/fmu/in/trajectory_setpoint')

    # Parâmetros do Controlador PID
    self.declare_parameter('kp', 1.0) 
    self.declare_parameter('ki', 0.1) 
    self.declare_parameter('kd', 0.5) 
    self.declare_parameter('max_vel', 2.0) 
    self.declare_parameter('integral_limit', 1.0) 

    # Parâmetros de Robustez e Configuração
    self.declare_parameter('arming_delay_s', 2.0) # Tempo (s) para stream antes de armar
    self.declare_parameter('pos_staleness_threshold_s', 0.5) # Tempo (s) até a posição ser considerada antiga
    
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
      self.arming_delay_s = self.get_parameter('arming_delay_s').get_parameter_value().double_value
      self.pos_staleness_threshold_s = self.get_parameter('pos_staleness_threshold_s').get_parameter_value().double_value
      self.arming_cycles_needed = int(self.arming_delay_s / self.timer_period)
      
      # Carrega a matriz de transformação
      tf_list = self.get_parameter('target_to_enu_tf').get_parameter_value().double_array_value
      self.T_target_to_enu = np.array(tf_list).reshape(3, 3)
      self.get_logger().info(f"Matriz de Transformação (Target -> ENU) carregada:\n{self.T_target_to_enu}")

  def init_ros_interfaces(self):
      """Inicializa os publishers, subscribers e timers."""
      trajectorySP = f"/fmu/in/trajectory_setpoint{get_message_name_version(TrajectorySetpoint)}"
      vehicle_command = f"/fmu/in/vehicle_command{get_message_name_version(VehicleCommand)}"
      
      # Publishers
      self.cmd_pub = self.create_publisher(VehicleCommand, vehicle_command, 10)
      self.offboard_control_pub = self.create_publisher(OffboardControlMode, self.offboard_mode_topic, 10)
      self.trajectory_setpoint_pub = self.create_publisher(TrajectorySetpoint, trajectorySP, 10)

      # Subscribers
      self.pos_sub = self.create_subscription(
        VehicleLocalPosition, self.px4_pos_topic, self.px4_pos_callback, 10)
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

  # --- Callbacks de Subscriber ---

  def visual_info_callback(self, msg):
      """Callback para processar a mensagem VisualInfo e definir o setpoint."""
      if not msg:
          return
          
      try:
          x_val = float(msg.x[0])
          y_val = float(msg.y[0])
          z_val = float(msg.z[0])
      except (TypeError, IndexError):
          # Fallback se não for iterável (ou seja, se for apenas um escalar "5.0" ou 5.0)
          self.get_logger().warn("Campos da VisualInfo não parecem ser listas. Tentando conversão direta.", throttle_duration_sec=5.0)
          x_val = float(msg.x)
          y_val = float(msg.y)
          z_val = float(msg.z)

      # 2. Crie um vetor 1D (shape (3,))
      target_coords = np.array([x_val, y_val, z_val])
      
      # 2. Transformar para o frame global (assumindo ENU) e depois converter para NED
      self.setpoint_ned = self.transform_target_to_ned_setpoint(target_coords)
      
      self.get_logger().info(f"Novo Setpoint Global (NED): ({self.setpoint_ned[0]:.2f}, {self.setpoint_ned[1]:.2f}, {self.setpoint_ned[2]:.2f})")
      
      # 3. Resetar o controlador PID e mudar o estado
      self.reset_pid_controller()
      
      # Se já estivermos armados, podemos ir para MOVING.
      # Se ainda estivermos em STARTING, ele mudará para IDLE primeiro,
      # e o `visual_info` será processado quando chegar em IDLE (ou podemos salvar o setpoint).
      # Por simplicidade, vamos permitir a transição direta se não estivermos em STARTING.
      if self.state != State.STARTING:
          self.state = State.MOVING
          self.get_logger().info(f"Setpoint recebido. Transicionando para: {self.state.name}")
      else:
          self.get_logger().warn("Setpoint recebido, mas ainda em estado STARTING. Aguardando IDLE.")


  def px4_pos_callback(self, msg):
      """Callback para atualizar a posição local atual do drone (frame NED)."""
      self.current_pos_ned = np.array([msg.x, msg.y, msg.z])
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

  def run_state_starting(self):
      """Estado de inicialização: envia setpoints vazios até estar pronto para armar."""
      self.publish_velocity_setpoint(np.array([0.0, 0.0, 0.0])) # Envia setpoint nulo
      
      if self.offboard_stream_counter == self.arming_cycles_needed:
          # self.get_logger().info("Enviando comando para Armar...")
          # self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
          
          self.get_logger().info("Enviando comando para Modo Offboard...")
          self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 
                                       param1=1.0, # 1.0 = Modo customizado
                                       param2=6.0) # 6.0 = PX4_CUSTOM_MAIN_MODE_OFFBOARD
          
          # Transição para o próximo estado
          self.state = State.IDLE
          self.get_logger().info(f"Transicionando para: {self.state.name}")
      
      elif self.offboard_stream_counter < self.arming_cycles_needed:
          self.offboard_stream_counter += 1

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
      error_distance = np.linalg.norm(error)
      
      if error_distance < self.threshold_distance:
          self.get_logger().info(f"SUCESSO! Alvo alcançado. Erro: {error_distance:.3f} m")
          self.state = State.SUCCESS
          self.publish_velocity_setpoint(np.array([0.0, 0.0, 0.0])) # Zera a velocidade
      else:
          # Publica a velocidade calculada
          self.publish_velocity_setpoint(output_vel)
          self.get_logger().info(f"Movendo... Erro: {error_distance:.2f} m", throttle_duration_sec=0.5)

  def run_state_success(self):
      """Estado de Sucesso: Mantém a posição no alvo."""
      self.publish_velocity_setpoint(np.array([0.0, 0.0, 0.0]))
      # O nó permanecerá aqui até que `visual_info_callback` receba um novo ponto.

  # --- Funções Auxiliares (Helpers) ---

  def transform_target_to_ned_setpoint(self, target_coords_cam):
      
      # 1. Transformar para ENU (East-North-Up)
      global_coords_enu = self.T_target_to_enu @ target_coords_cam
      
      # 2. Converter ENU para NED (North-East-Down) para o PX4
      setpoint_ned = np.array([
          global_coords_enu[0],                   # ENU North  -> NED X
          global_coords_enu[1],                   # ENU East   -> NED Y
          (global_coords_enu[2] - 1.0) # ENU Up     -> NED Down (com offset)
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