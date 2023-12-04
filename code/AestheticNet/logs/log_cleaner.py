import os

# open log.log
log = open('/home/zerui/SSIRA/code/AestheticNet/logs/log.log', 'r')
print("log.log opened")

# read log.log
log_lines = log.readlines()
print("log.log read")

# for each line, delete line with text "Phase Pretext"
for line in log_lines:
    if "ERROR Error" in line:
        print(line)
        log_lines.remove(line)
        print("line removed")

# save the changes
log = open('/home/zerui/SSIRA/code/AestheticNet/logs/log.log', 'w')
print("log.log opened")
log.writelines(log_lines)
print("log.log written")

# close log.log
log.close()
