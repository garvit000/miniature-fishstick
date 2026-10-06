"""Root wrapper script to execute the stress-detection pipeline from workspace root."""
import os
import sys

# Forward to stress-detection/main.py
project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stress-detection")
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

os.chdir(project_dir)

import main

if __name__ == "__main__":
    main.main()
