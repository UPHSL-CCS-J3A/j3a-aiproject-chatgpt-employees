"""
Gender & Age Detection - Main Launcher
Run this file to start the desktop UI application
"""

import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Change to src directory for relative paths to work
os.chdir(os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the UI
import tkinter as tk
from ui import GenderAgeSkinDetectorUI

if __name__ == "__main__":
    root = tk.Tk()
    app = GenderAgeSkinDetectorUI(root)
    root.mainloop()
