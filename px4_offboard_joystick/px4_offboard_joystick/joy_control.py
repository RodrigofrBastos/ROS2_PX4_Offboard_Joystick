#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import std_msgs.msg
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

class JoyControlNode(Node):
    def __init__(self):
        super().__init__('joy_control_node')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        # Declare default parameters
        self.declare_parameter('input_topic', '/joy')
        self.declare_parameter('output_topic', '/offboard_velocity_cmd')
        self.declare_parameter('axis_map.set_aileron_index', 0)
        self.declare_parameter('axis_map.set_elevator_index', 1)
        self.declare_parameter('axis_map.set_throttle_index', 2)
        self.declare_parameter('axis_map.set_rudder_index', 3)
        self.declare_parameter('button_map.set_arm_disarm_index', 3)
        # self.declare_parameter('button_map.set_mode_index', 5)

        self.declare_parameter('drone_parameters.lin_speed', 0.5)
        self.declare_parameter('drone_parameters.turn_speed', 0.2)
        
        # Declare publishers and subscribers
        self.subscription = self.create_subscription(
            Joy,
            self.get_parameter('input_topic').value,
            self.joy_callback,
            qos_profile)
        
        self.pub = self.create_publisher(
            Twist, self.get_parameter('output_topic').value,
            qos_profile)

        self.arm_toggle = False
        self.arm_pub = self.create_publisher(
            std_msgs.msg.Bool,
            '/arm_message',
            qos_profile)
        self.get_logger().info("Joy Control Node has been started.")
        
        # Timer to periodically check parameters
        # self.timer = self.create_timer(2.0, self.timer_callback)

    # def timer_callback(self):
    # # 3. Obter os valores atuais dos parâmetros
    # nome_robo = self.get_parameter('nome_do_robo').get_parameter_value().string_value
    # velocidade = self.get_parameter('velocidade_maxima').get_parameter_value().integer_value
    # escala = self.get_parameter('fator_de_escala').get_parameter_value().double_value
    # debug = self.get_parameter('em_modo_debug').get_parameter_value().bool_value
    
    def joy_callback(self, msg: Joy):
        """
        Callback executado toda vez que uma nova mensagem do joystick é recebida.
        """
        try:
            # Etapa 1: Obter os parâmetros de configuração.
            # É bom fazer isso dentro do callback para que as alterações via 'ros2 param set'
            # sejam refletidas em tempo real.
            aileron_index = self.get_parameter('axis_map.set_aileron_index').value
            elevator_index = self.get_parameter('axis_map.set_elevator_index').value
            throttle_index = self.get_parameter('axis_map.set_throttle_index').value
            rudder_index = self.get_parameter('axis_map.set_rudder_index').value
            arm_button_index = self.get_parameter('button_map.set_arm_disarm_index').value
            
            speed = self.get_parameter('drone_parameters.lin_speed').value
            turn = self.get_parameter('drone_parameters.turn_speed').value

            # Fix axis and button index out of range and conventions issues
            pitch_velocity = msg.axes[aileron_index] * speed * 1.0 # Pitch is usually inverted
            roll_velocity = msg.axes[elevator_index] * speed * -1.0 # Roll is usually inverted
            throttle_velocity = msg.axes[throttle_index] * speed * -1.0 # Throttle is usually inverted
            yaw_velocity = -msg.axes[rudder_index] * turn # Yaw is not usually inverted

            # Etapa 3: Criar e publicar a mensagem Twist.
            twist = Twist()
            twist.linear.x = pitch_velocity     # Movimento para frente/trás
            twist.linear.y = roll_velocity       # Movimento para os lados
            twist.linear.z = throttle_velocity   # Movimento para cima/baixo
            twist.angular.z = yaw_velocity     # Rotação
            
            self.pub.publish(twist)

            # Etapa 4: Lidar com o comando de armar/desarmar (lógica de toggle).
            current_arm_button_state = msg.buttons[arm_button_index]
            # A ação só ocorre na TRANSIÇÃO do botão (de solto para pressionado).
            if current_arm_button_state == 1 and self.last_arm_button_state == 0:
                self.arm_toggle = not self.arm_toggle  # Inverte o estado
                arm_msg = Bool()
                arm_msg.data = self.arm_toggle
                self.arm_pub.publish(arm_msg)
                self.get_logger().info(f"Comando Armar/Desarmar enviado: {self.arm_toggle}")
            
            # Atualiza o estado do botão para a próxima verificação.
            self.last_arm_button_state = current_arm_button_state

        except IndexError as e:
            # Lida com o erro caso o índice do botão/eixo configurado não exista no joystick.
            self.get_logger().error(f"Erro de índice no joystick: {e}. Verifique os parâmetros!")
        except Exception as e:
            # Lida com outros erros inesperados.
            self.get_logger().error(f"Ocorreu um erro inesperado no joy_callback: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = JoyControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
        
if __name__ == '__main__':
    main()
        