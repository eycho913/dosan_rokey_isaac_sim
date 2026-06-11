import os

with open('/home/rokey/dev_ws/coupang_ws/scripts/coupang_sh5_bringup_v2.py', 'r') as f:
    lines = f.readlines()

# Extract VRDemonstrationLogger (lines 112 to 206)
logger_lines = lines[112:206]
with open('/home/rokey/dev_ws/coupang_ws/scripts/sh5_logger.py', 'w') as f:
    f.write("import os\nimport time\nimport h5py\nimport numpy as np\n\n")
    f.writelines(logger_lines)

# Extract TerminalKeyboard (lines 974 to 1010)
keyboard_lines = lines[974:1010]
with open('/home/rokey/dev_ws/coupang_ws/scripts/sh5_keyboard.py', 'w') as f:
    f.write("import select\nimport sys\nimport termios\nimport tty\n\n")
    f.writelines(keyboard_lines)

# Now remove them from original and add imports
new_lines = []
i = 0
while i < len(lines):
    if i == 112:
        new_lines.append("from sh5_logger import VRDemonstrationLogger\n")
        new_lines.append("from sh5_keyboard import TerminalKeyboard\n")
        i = 206
        continue
    if i == 974:
        i = 1010
        continue
    new_lines.append(lines[i])
    i += 1

with open('/home/rokey/dev_ws/coupang_ws/scripts/coupang_sh5_bringup_v2.py', 'w') as f:
    f.writelines(new_lines)

print("Split completed.")
