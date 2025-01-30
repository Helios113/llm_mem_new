import os
import subprocess
import tempfile

TEMPLATE_FILE = "/nfs-share/pa511/new_work/scripts/slurm_template.template"

params_list = [
    {"script_name": "finetune.py", "params": "dataset.path=/nfs-share/pa511/new_work/data/amazonqa training.per_device_train_batch_size=16 simulation.use_lora=True", "node": "mauao" },
    {"script_name": "finetune.py", "params": "dataset.path=/nfs-share/pa511/new_work/data/aus_qa training.per_device_train_batch_size=16 simulation.use_lora=True", "node": "mauao" },
    {"script_name": "finetune.py", "params": "dataset.path=/nfs-share/pa511/new_work/data/finqa training.per_device_train_batch_size=16 simulation.use_lora=True", "node": "mauao" },
    {"script_name": "finetune.py", "params": "dataset.path=/nfs-share/pa511/new_work/data/medalpaca training.per_device_train_batch_size=16 simulation.use_lora=True", "node": "mauao" },
    {"script_name": "finetune.py", "params": "dataset.path=/nfs-share/pa511/new_work/data/triviaqa training.per_device_train_batch_size=16 simulation.use_lora=True", "node": "mauao" },
    # Add more parameter dictionaries here
]

last_job_id = "1657"

for param_dict in params_list:
    with open(TEMPLATE_FILE, 'r') as template_file:
        template_content = template_file.read()

    job_script_content = template_content.format(
        dependency=last_job_id,
        script_name=param_dict["script_name"],
        params=param_dict["params"],
        node=param_dict["node"]
    )

    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.sh') as job_script:
        job_script.write(job_script_content)
        job_script_path = job_script.name

    result = subprocess.run(['sbatch', job_script_path], capture_output=True, text=True)
    last_job_id = result.stdout.strip().split()[-1]

    os.remove(job_script_path)
