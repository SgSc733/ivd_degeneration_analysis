import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import main
import sys

if __name__ == "__main__":
    sys.argv.append('--gui')
    main()