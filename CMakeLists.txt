cmake_minimum_required(VERSION 3.8)
project(bbac_ics)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# 1. Install Python package (Library code)
# ========================================================
# ament_python_install_package(${PROJECT_NAME})
ament_python_install_package(${PROJECT_NAME}_core)

# 2. Install Executables (Nodes) - Renaming & Permissions
# ========================================================
install(
  PROGRAMS bbac_ics_core/nodes/bbac_main_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME bbac_main_node 
)

install(
  PROGRAMS bbac_ics_core/nodes/experiment_evaluator_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME experiment_evaluator_node
)

# Scripts experiments
install(
  PROGRAMS 
    bbac_ics_core/experiments/ablation_study.py
    bbac_ics_core/experiments/adaptive_eval.py
    bbac_ics_core/experiments/dynamic_rules_test.py
  DESTINATION lib/${PROJECT_NAME}
)

# 3. Generate ROS 2 Messages
# ========================================================
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AccessDecision.msg"
  "msg/AccessRequest.msg"          
  "msg/EmergencyAlert.msg"        
  "msg/LayerDecisionDetail.msg"
  "msg/LayerOutput.msg"         
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

# 4. Install Configs, Launch & Msg definitions
# ========================================================
install(DIRECTORY 
  launch 
  msg 
  config 
  data 
  DESTINATION share/${PROJECT_NAME} 
)

ament_package()
