import os
from glob import glob
from setuptools import setup

package_name = 'px4_offboard_joystick'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/ament_index/resource_index/packages',
            ['resource/' + 'visualize.rviz']),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name), glob('resource/*rviz')),
        (os.path.join('share', package_name, 'cfg'), glob('cfg/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Braden',
    maintainer_email='braden@arkelectron.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                'offboard_control = px4_offboard_joystick.offboard_control:main',
                'visualizer = px4_offboard_joystick.visualizer:main',
                'velocity_control = px4_offboard_joystick.velocity_control:main',
                'control = px4_offboard_joystick.control:main',
                'joystick_control = px4_offboard_joystick.joy_control:main',
                'processes = px4_offboard_joystick.processes:main'
        ],
    },
)
