import subprocess
import sys
import os
from tqdm import tqdm

# set the folder in which the cli.py is located
rerun_folder = os.path.abspath(os.path.dirname(__file__))

# the first one is the rerun.py script, the last is the output
used_arguments = sys.argv[3:-1]

# this sets the amount of runs, which are performed
amount_of_runs = int(sys.argv[1])

# init runs
starting_run_id = int(sys.argv[2])

print("starting at ", starting_run_id)

output_location = os.path.abspath(sys.argv[-1])
for run_id in tqdm(range(amount_of_runs)):
    # in each run, the arguments are reused
    cmd_render = ["python", os.path.join(rerun_folder, "cli.py")]
    cmd_render.extend(["run"])
    cmd_render.extend(used_arguments)
    # the only exception is the output, which gets changed for each run, so that the examples are not overwritten
    new_location = os.path.join(output_location, str(run_id + starting_run_id))
    cmd_render.append(new_location)
    cmd_render.append("> /dev/null")
    print(" ".join(cmd_render))
    subprocess.call(" ".join(cmd_render), shell=True)

        
    #cmd_vis = ["python", os.path.join(rerun_folder, "cli.py")]
    #cmd_vis.extend(["vis", "hdf5", new_location+"/*.hdf5", "--depth_max", "2", "--save", new_location])

    #print(" ".join(cmd_vis))

    # execute one BlenderProc run
    #subprocess.call(" ".join(cmd_vis), shell=True)




    






