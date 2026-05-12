# main.py
import subprocess
import sys

def run(script):
    print(f"\n{'='*50}")
    print(f"  Running {script}")
    print(f"{'='*50}\n")
    result = subprocess.run([sys.executable, script], check=True)
    return result

if __name__ == "__main__":
    run("data/generator.py")
    run("benchmark.py")
    print("\n✓ Done. Results are in results/csv/ and results/plots/")