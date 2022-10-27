import time
import os
import subprocess

def job_num():
    total_job = 0
    lines = str(subprocess.check_output(['qstat','-u','khs'])).split('\\n')
    for line in lines:
        if 'khs' in line:
            if ' Q ' in line or ' R ' in line:
                if 'cal' in line:
                    total_job +=1
    return total_job


with open('jobscript.x') as f:
    lines = f.readlines()

for i in range(10,11):
    new_lines = []
    for line in lines:
        if 'num'in line:
            new_lines.append(line.replace('num',str(int(i*10))))
        else:
            new_lines.append(line)

    with open('jobscript_.x','w') as f:
        for line in new_lines:
            f.write(line)
    os.system('qsub jobscript_.x')
    time.sleep(20)
    #while job_num() > 4:
    #    time.sleep(5)
