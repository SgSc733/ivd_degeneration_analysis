from .preprocessing_validator import PreprocessingValidator

__version__ = '1.0.0'
__author__ = 'Your Name'
__all__ = ['PreprocessingValidator']

def main():
    from .preprocessing_validator import main as validator_main
    validator_main()
