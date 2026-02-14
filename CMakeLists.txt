cmake_minimum_required(VERSION 3.8)
project(bbac_ics)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
# ===============================
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# Install Python package
# ===============================
#ament_python_install_package(${PROJECT_NAME})
ament_python_install_package(${PROJECT_NAME}_core)

# Make wrappers executable
# ===============================
file(CHMOD 
  ${CMAKE_CURRENT_SOURCE_DIR}/bbac_main_node
  ${CMAKE_CURRENT_SOURCE_DIR}/experiment_evaluator_node
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

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
  "msgs/AccessDecision.msg"
  "msgs/AccessRequest.msg"          
  "msgs/EmergencyAlert.msg"        
  "msgs/LayerDecisionDetail.msg"
  "msgs/LayerOutput.msg"         
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
