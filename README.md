# Postgraduate-graduation-project
This project presents a comprehensive graphical user interface (GUI) framework designed for systematic comparison and evaluation of static and dynamic Steady-State Visual Evoked Potential (SSVEP) brain-computer interface paradigms. The system serves as an integrated research platform that enables researchers to configure, execute, and analyze SSVEP experiments with flexible parameter control. Here is the introduction for each code file：

Core GUI Components
main.py - Application entry point that initializes the main window, manages page navigation, and handles audio playback for user feedback during experiments.  
main_page.py - Welcome screen.  
setting_page.py - Configuration interface for setting the number of visual stimuli in the experiment.  
customer.py - Defines reusable UI components (buttons, labels, input fields) and styling constants for consistent appearance across the application.  
Experiment Design
appearance_page.py - Visual stimulus design interface where users customize shape properties (type, color, size, position) for each stimulus. Includes save/load functionality for configurations.  
motion_page.py - Controls dynamic stimulus behavior, allowing users to set whether stimuli are static or move to specified target positions during experiments.  
blocks_page.py - Configures experimental timing parameters including cue duration, flashing time, pause intervals, and rest periods between blocks.  
Visual Stimuli System
shape.py - Implements various geometric shapes (squares, circles, triangles, etc.) with SSVEP properties like frequency and phase. Handles visual flashing and motion animation.  
shape_container.py - Manages collections of shapes within experimental displays, handling layout positioning, user interactions, and coordinate transformations.  
shape_dialog.py - Popup interface for editing individual stimulus properties (frequency, phase) through double-click interactions.  
Experiment Execution
experiment_page.py - Main experimental control interface that coordinates trial execution, displays real-time status, and integrates with TRCA recognition system.  
experiment_window.py - Full-screen experimental display window that presents visual stimuli to participants during data collection sessions.  
shapes_page.py - Base class for managing stimulus presentation, handling experimental state changes (cue, flash, pause phases).  
SSVEP Signal Processing
eeg_processor.py - Real-time EEG data acquisition and processing system with filtering, artifact removal, and TRCA-based classification algorithms.  
TRCA_benchmark.py - Implementation of Task-Related Component Analysis algorithm for SSVEP signal recognition, including model training and testing functions.  
trca_integration.py - Bridges the TRCA algorithm with the GUI system, managing data collection, real-time recognition, and result storage.  
Audio files and image files
All.wav - Play "All blocks finished".  
Twenty.wav - Play "Twenty minutes left, please get ready".  
icon.png - A logo image.
