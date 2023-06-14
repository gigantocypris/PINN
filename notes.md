Testing experimental/DDP_example2.py:

1 GPU
Total training time was: 82.07545876502991

2 GPU
Total training time was: 49.86513566970825

3 GPU
Total training time was: 35.786556243896484

4 GPU
Total training time was: 30.676435232162476

Running the following:

(PINN) vidyagan@login31:/pscratch/sd/v/vidyagan/output_PINN> sbatch /pscratch/sd/v/vidyagan/PINN/slurm_tr
ain.sh 256
Submitted batch job 10212612
(PINN) vidyagan@login31:/pscratch/sd/v/vidyagan/output_PINN> sbatch /pscratch/sd/v/vidyagan/PINN/slurm_train.sh 512
Submitted batch job 10212615
(PINN) vidyagan@login31:/pscratch/sd/v/vidyagan/output_PINN> sbatch /pscratch/sd/v/vidyagan/PINN/slurm_train.sh 1024
Submitted batch job 10212616
(PINN) vidyagan@login31:/pscratch/sd/v/vidyagan/output_PINN> sbatch /pscratch/sd/v/vidyagan/PINN/slurm_train.sh 2048
Submitted batch job 10212617
(PINN) vidyagan@login31:/pscratch/sd/v/vidyagan/output_PINN> sbatch /pscratch/sd/v/vidyagan/PINN/slurm_train.sh 4096
Submitted batch job 10212619
(PINN) vidyagan@login31:/pscratch/sd/v/vidyagan/output_PINN> sbatch /pscratch/sd/v/vidyagan/PINN/slurm_train.sh 8192
Submitted batch job 10212621