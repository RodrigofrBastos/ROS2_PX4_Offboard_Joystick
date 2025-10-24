
#!/usr/bin/env python3

# Import the subprocess and time modules
import subprocess
import time

# List of commands to run
commands = [
    # Run the Micro XRCE-DDS Agent
    "MicroXRCEAgent udp4 -p 8888",

    # Run the PX4 SITL simulation
    "cd /home/rodrigo/SOFT-PX4-SIMULATION/PX4/my-px4-autopilot && PX4_GZ_WORLD=grasping_world PX4_GZ_MODEL_POSE='0,0,1' make px4_sitl gz_x500_spiral_cobra",

    # Run QGroundControl
    "cd ~/Downloads && ./QGroundControl.AppImage"
]

# Loop through each command in the list
for command in commands:
    # Each command is run in a new tab of the gnome-terminal
    subprocess.run(["gnome-terminal", "--tab", "--", "bash", "-c", command + "; exec bash"])
    
    # Pause between each command
    time.sleep(1)