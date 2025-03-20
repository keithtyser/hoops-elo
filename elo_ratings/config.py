"""
Configuration utilities for the NCAA Basketball ELO Rating System.

This module handles loading parameters from config files and providing defaults
when needed.
"""

import os
import json
import logging

logger = logging.getLogger('elo_ratings')

# Default configuration values
DEFAULT_CONFIG = {
    "men": {
        "initial_rating": 1500,
        "k_factor": 38,  
        "alpha": 10,     
        "carryover": True,
        "carryover_factor": 0.5,  
        "home_court": 100,  
        "slope": 0.01,   
        "mov_formula": "linear",
        "weighting_scheme": None
    },
    "women": {
        "initial_rating": 1500,
        "k_factor": 56,  
        "alpha": 10,     
        "carryover": True,
        "carryover_factor": 0.5,  
        "home_court": 100,  
        "slope": 0.01,    
        "mov_formula": "linear",
        "weighting_scheme": None
    }
}

def load_config(config_file="config.json"):
    """
    Load configuration from file.
    
    Parameters
    ----------
    config_file : str, optional
        Path to configuration file
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                
            # Update men's config
            if "men" in user_config:
                config["men"].update(user_config["men"])
                
            # Update women's config
            if "women" in user_config:
                config["women"].update(user_config["women"])
                
            logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_file}: {str(e)}")
            logger.warning("Using default configuration")
    else:
        logger.warning(f"Configuration file {config_file} not found, using defaults")
        # Create a template config file if it doesn't exist
        try:
            with open("config_template.json", 'w') as f:
                json.dump(DEFAULT_CONFIG, f, indent=4)
            logger.info("Created config_template.json with default values")
        except Exception as e:
            logger.error(f"Error creating config template: {str(e)}")
    
    return config

def get_gender_config(gender, config=None):
    """
    Get configuration for a specific gender.
    
    Parameters
    ----------
    gender : str
        'men' or 'women'
    config : dict, optional
        Configuration dictionary
        
    Returns
    -------
    dict
        Gender-specific configuration
    """
    if config is None:
        config = load_config()
        
    if gender.lower() not in ["men", "women"]:
        raise ValueError(f"Invalid gender: {gender}. Must be 'men' or 'women'")
        
    return config[gender.lower()] 