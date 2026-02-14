cmake_minimum_required(VERSION 3.8)
project(bbac_ics)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 1. Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# Install Python package
# ===============================
#ament_python_install_package(${PROJECT_NAME})
ament_python_install_package(${PROJECT_NAME}_core)

# Install ROS node
# ===============================
install(
  PROGRAMS
    bbac_main_node
    experiment_evaluator_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    bbac_ics_core/experiments/ablation_study.py
    bbac_ics_core/experiments/adaptive_eval.py
    bbac_ics_core/experiments/dynamic_rules_test.py
  DESTINATION lib/${PROJECT_NAME}
)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AccessDecision.msg"
  "msg/AccessRequest.msg"          
  "msg/EmergencyAlert.msg"        
  "msg/LayerDecisionDetail.msg"
  "msg/LayerOutput.msg"         
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

# Install Configs & Launch
# ===============================
install(DIRECTORY 
  launch 
  msg 
  config 
  data 
  DESTINATION share/${PROJECT_NAME} 
)

ament_package()
