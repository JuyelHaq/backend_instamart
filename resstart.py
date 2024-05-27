import subprocess
import os
import signal
import sys

def start_flask():
    # Run the Flask app in a new process
    subprocess.Popen([sys.executable, 'app.py'])

def restart_flask():
    # Get the current process ID
    pid = os.getpid()
    print("Restarting Flask app...")

    # Terminate the current process
    os.kill(pid, signal.SIGTERM)

if __name__ == "__main__":
    start_flask()
    # Optionally, you can listen for a specific signal or event to trigger the restart
    # For example, a signal from another process or a specific HTTP request
    # For simplicity, let's assume the restart is triggered manually
    input("Press Enter to restart Flask app...")
    restart_flask()

