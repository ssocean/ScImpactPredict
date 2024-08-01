import subprocess
import time
from tqdm import tqdm
def get_current_node():

    command = ['hostname']
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception("Failed to get hostname:", result.stderr)
    return result.stdout.strip()

def get_user_job_ids(user, node):

    command = ['squeue', '-u', user, '--format=%i %N', '--noheader']
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception("Failed to execute squeue:", result.stderr)
    
    job_ids = []
    for line in result.stdout.splitlines():
        job_id, job_node = line.split()
        if job_node == node:
            job_ids.append(job_id)
    
    return job_ids

def cancel_jobs(job_ids):

    for job_id in job_ids:
        command = ['scancel', job_id]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Job {job_id} canceled successfully.")
        else:
            print(f"Failed to cancel job {job_id}:", result.stderr)
            
def auto_cancel(user):

    
    node = get_current_node()
    print(f"Current node: {node}")


    job_ids = get_user_job_ids(user, node)
    if job_ids:
        print("Job IDs found:", job_ids)

        cancel_jobs(job_ids)
    else:
        print("No jobs found for user", user, "on node", node)



if __name__ == '__main__':
    # 防止代码出错，瞬间kill掉当前节点。如果开始倒计时，可在60秒内Ctrl+C结束进程。
    for _ in tqdm(range(60), desc="Countdown Timer to shut down current node:", unit="s"):
        time.sleep(1)
        
    user = 'u1120220285'
    auto_cancel(user)