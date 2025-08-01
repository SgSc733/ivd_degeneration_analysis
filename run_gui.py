import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from gui import IntegratedFeatureExtractorGUI

def main():
    root = tk.Tk()
    root.title("椎间盘退变分析系统")
    root.geometry("1280x720")
    
    app = IntegratedFeatureExtractorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
