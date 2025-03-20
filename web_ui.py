#!/usr/bin/env python
"""
Web UI launcher for the NCAA Basketball ELO Rating System.

This script serves as the entry point for the web-based user interface.
"""

import os
import argparse
from elo_ratings.web_ui import app, load_data, run_web_ui

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="NCAA Basketball ELO Rating System Web UI")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--data-dir", default="data", help="Directory containing data files (default: data)")
    parser.add_argument("--gender", choices=["men", "women"], help="Preload data for the specified gender")
    
    return parser.parse_args()

def main():
    """Main entry point for the web UI."""
    args = parse_args()
    
    # Create the templates and static directories if they don't exist
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    # Preload data if gender is specified
    if args.gender:
        load_data(args.gender, args.data_dir)
    
    # Run the Flask application
    run_web_ui(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main() 