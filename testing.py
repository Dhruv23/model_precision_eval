import subprocess
import re
import pandas as pd

def run_trtexec(engine_path: str):
    # Run trtexec command
    cmd = [
        "trtexec",
        f"--loadEngine={engine_path}",
        "--shapes=images:1x3x640x640",
        "--avgRuns=100"
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return result.stdout

def parse_trtexec_output(output: str):
    # Patterns to extract values
    patterns = {
        "Throughput (qps)": r"Throughput:\s+([\d.]+)\s+qps",
        "Latency mean (ms)": r"mean = ([\d.]+) ms",
        "GPU Compute Time mean (ms)": r"GPU Compute Time:.*mean = ([\d.]+) ms",
        "H2D Latency mean (ms)": r"H2D Latency:.*mean = ([\d.]+) ms",
        "D2H Latency mean (ms)": r"D2H Latency:.*mean = ([\d.]+) ms",
    }

    # Extract matched values
    data = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        data[key] = float(match.group(1)) if match else None

    return pd.DataFrame([data])

if __name__ == "__main__":
    engine_file_fp16 = "yolov5s_fp16.engine"
    output = run_trtexec(engine_file_fp16)
    df_fp16 = parse_trtexec_output(output)
    df_fp16.to_csv("trtexec_results_fp16.csv", index=False)

    engine_file_int8 = "yolov5s_int8.engine"
    output = run_trtexec(engine_file_int8)
    df_int8 = parse_trtexec_output(output)
    df_int8.to_csv("trtexec_results_int8.csv", index=False)
    


