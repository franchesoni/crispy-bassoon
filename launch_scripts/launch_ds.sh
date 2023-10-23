#!/bin/bash

ds=$1
maxtime=$2
# a100 or v100
gputype=$3

echo "Launching job for ds $ds for ${maxtime} with gpu ${gputype}"
if [ "$gputype" == "a100" ]; then
   optional_sbatch="#SBATCH --partition=gpu_p4"
elif [ "$gputype" == "v100" ]; then
   optional_sbatch=""
else
   echo "Invalid gpu type, should be a100 or v100"
   exit 1
fi

sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=$ds
#SBATCH --time=$maxtime
#SBATCH --output=logs/${ds}.out
#SBATCH --error=logs/${ds}.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
$(echo -e "$optional_sbatch")

# Set up pre-defined Python env
module purge
module load pytorch-gpu/py3/1.10.1

cd ${SCRATCH}/cvpr/crispy-bassoon/
python -u -m mess.experiments.iis.precompute_masks sam --dev --ds=${ds}

EOT
