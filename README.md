# Postgraduate-graduation-project
Development of a GUI-Based Analysis Framework for Comparative Evaluation of Static and Dynamic SSVEP Brain-Computer Interface Paradigms
Project Overview
This project presents a comprehensive graphical user interface (GUI) framework designed for systematic comparison and evaluation of static and dynamic Steady-State Visual Evoked Potential (SSVEP) brain-computer interface paradigms. The system serves as an integrated research platform that enables researchers to configure, execute, and analyze SSVEP experiments with flexible parameter control.
1.1Core Functionality
1. Paradigm Selection
The system supports two primary paradigms:
Static SSVEP: Traditional flickering visual stimuli with fixed frequency modulation
SSVEP + Motion: Dynamic paradigm incorporating movement patterns with SSVEP stimulation
2. Configurable Parameters and Their Functions
Stimulus Configuration Parameters
Number of Stimuli
Function: Defines the total number of visual targets in the experiment
Range: User-defined (e.g., 1-40 targets)
Impact: Affects classification complexity and information transfer rate potential
Frequency Settings
Function: Sets the flickering frequency for each stimulus target
Range: Typically 6-40 Hz
Impact: Determines the fundamental frequency of evoked neural responses and affects user comfort
Phase Settings
Function: Configures initial phase offset for each stimulus
Range: 0-2π radians (0-360°)
Impact: Enables phase-coded paradigms and reduces interference between adjacent stimuli
Visual Appearance Parameters
Shape Selection
Options: Square, Rectangle, Circle, Ellipse, Triangle, Free curve
Function: Defines the geometric form of visual stimuli
Impact: Affects visual attention and cortical activation patterns
Color Configuration
Options: White, Red, Blue, and additional color variants
Function: Sets stimulus color properties
Impact: Color-frequency interactions affect SSVEP amplitude (red more effective at lower frequencies, white at higher frequencies)
Size Parameters (a × b)
Function: Controls stimulus dimensions in pixels
Configuration: Width (a) and height (b) independently adjustable
Impact: Larger stimuli typically generate stronger SSVEP responses
Location Settings (X, Y coordinates)
Function: Positions stimuli on the display screen
Input Methods: 
oDirect coordinate input
oMouse-based dragging
oKeyboard arrow key adjustment
Impact: Spatial arrangement affects visual attention distribution and potential interference
Temporal Parameters
Cue Time
Function: Duration of pre-stimulus instruction display
Purpose: Allows subject preparation and attention direction
Typical Range: 0.5-2.0 seconds
Flash Time (Static SSVEP)
Function: Duration of stimulus presentation
Impact: Longer durations generally improve classification accuracy but may increase fatigue
Moving Speed (Dynamic SSVEP)
Function: Controls velocity of stimulus movement in dynamic paradigm
Impact: Affects motion-evoked responses and user tracking ability
Pause Time
Function: Inter-stimulus interval duration
Purpose: Prevents visual aftereffects and allows neural response reset
Rest Time Between Trials
Function: Extended break period between experimental trials
Purpose: Minimizes fatigue accumulation and maintains consistent performance
Number of Blocks
Function: Defines experimental session structure
Impact: Affects total data quantity and statistical reliability
3. Real-Time Monitoring and Control
Experiment Execution Interface
Real-time Progress Tracking: Displays current block and trial numbers
Performance Monitoring: Shows target identification and classification results
Experimental Control: Pause, continue, and restart functionality
Parameter Display: Real-time visualization of current experimental settings
Data Collection Capabilities
Signal Recording: Integrated EEG data acquisition support
Performance Metrics: Automatic calculation of accuracy and ITR
Export Functions: Data storage for offline analysis
1.2System Advantages
Flexibility and Customization
Comprehensive parameter control enables tailored experimental designs
Support for both static and dynamic paradigm configurations
Real-time parameter adjustment capabilities
User-Friendly Interface
Intuitive graphical controls reduce setup complexity
Visual feedback for stimulus positioning and configuration
Step-by-step configuration workflow
Research Applications
Comparative Studies: Direct comparison between static and dynamic paradigms
Parameter Optimization: Systematic evaluation of stimulus configurations
Algorithm Development: Platform for testing new SSVEP classification methods
User Experience Research: Assessment of comfort and fatigue factors
Educational Value
Training Platform: Ideal for introducing researchers to SSVEP-BCI concepts
Demonstration Tool: Visual representation of different paradigm characteristics
Protocol Standardization: Ensures consistent experimental procedures
1.3Technical Implementation
The system integrates Task-Related Component Analysis (TRCA) algorithms for signal processing and classification, providing standardized performance evaluation metrics across different paradigm configurations. The modular design allows for future expansion with additional analysis methods and stimulus paradigms.
This comprehensive GUI framework serves as both a research tool and educational platform, facilitating advancement in SSVEP-BCI technology through systematic comparative analysis and parameter optimization capabilities.
