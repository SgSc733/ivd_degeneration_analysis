class Config:


    DISC_LABELS = {
        'L1-L2': {'disc': 3, 'upper': 2, 'lower': 4},
        'L2-L3': {'disc': 5, 'upper': 4, 'lower': 6},
        'L3-L4': {'disc': 7, 'upper': 6, 'lower': 8},
        'L4-L5': {'disc': 9, 'upper': 8, 'lower': 10},
        'L5-S1': {'disc': 11, 'upper': 10, 'lower': 12}
    }


    DURAL_SAC_LABEL = 20  

    
    DSCR_PARAMS = {
        'spline_smoothing': 0, 
        'spline_degree': 2,     
        'min_landmarks': 3    
    }


    DHI_PARAMS = {
        'central_ratio': 0.8,
        'calculate_dwr': True,
        'consider_bulging': True
    }
    
    ASI_PARAMS = {
        'n_components': 2,
        'scale_factor': 255.0
    }
    
    FD_PARAMS = {
        'threshold_percent': 0.65,
        'min_box_size': 1,
        'max_box_size': None
    }
    
    T2SI_PARAMS = {
        'roi_method': 'TARGET',
        'brightness_percentile': 75,
        'min_roi_size': 20
    }

    GABOR_PARAMS = {
        'wavelengths': [2, 4, 6, 8, 10],
        'orientations': None,  
        'frequency': 0.1,
        'sigma': None,
        'gamma': 0.5,
        'psi': 0
    }

    HU_MOMENTS_PARAMS = {}

    TEXTURE_PARAMS = {
        'lbp_radius': 1,
        'lbp_n_points': 8,
    }

    NUM_SLICES = 3 
    SLICE_AXIS = 0  

    VIS_PARAMS = {
        'style': 'seaborn',
        'dpi': 150,
        'save_intermediate': True
    }

    PREPROCESSING_PARAMS = {
        'target_size': [512, 512],  
        'texture': {
            'bin_width': 16,
            'normalize': True,
            'robust': False,  
            'exclude_percentile': 0.0  
        },
        'fractal': {
            'window_center': 128,
            'window_width': 255,
            'threshold_percentile': 65,
            'edge_method': 'canny'
        },
        'signal_intensity': {
            'interpolation': 'linear'
        }
    }

    FILTER_PARAMS = {
        'log': {
            'sigma_list': [1, 3, 5]
        },
        'wavelet': {
            'wavelet': 'db1',
            'level': 1
        }
    }

    OUTPUT_FORMATS = ['excel', 'json', 'csv']

    FEATURE_SETS = {
        'conventional': ['dhi', 'asi'],
        'shape': ['dhi', 'hu_moments', 'dscr'], 
        'texture': ['gabor', 'texture_features'],
        'fractal': ['fd'],
        'signal': ['t2si'],
        'stenosis': ['dscr'],  
        'all': ['dhi', 'asi', 'fd', 't2si', 'gabor', 'hu_moments', 'texture_features', 'dscr']  
    }

    PARALLEL_CONFIG = {
        'enabled': True,  
        'max_workers': None, 
        'chunk_size': 1,  
        'backend': 'multiprocessing',  
        'use_gpu': False,  
    }

    MEMORY_CONFIG = {
        'max_memory_gb': 8,  
        'cache_enabled': True,  
        'gc_interval': 10,  
    }

    CALCULATOR_PARALLEL = {
        'dhi': {'enabled': False},  
        'asi': {'enabled': True, 'max_workers': 2},
        'fd': {'enabled': False},   
        't2si': {'enabled': True, 'max_workers': 2},
        'gabor': {'enabled': True, 'max_workers': 4}, 
        'hu': {'enabled': False},   
        'texture': {'enabled': True, 'max_workers': 4}
    }

