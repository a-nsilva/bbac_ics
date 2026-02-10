cmake_minimum_required(VERSION 3.8)
project(bbac_ics)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 1. Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# 3. Install Python Module (bbac_ics_core/)
### ament_python_install_package(bbac_ics_core)
install(DIRECTORY bbac_ics_core/
  DESTINATION lib/${PROJECT_NAME}/bbac_ics_core
)

# 4. Install Executables (Nós ROS)
install(PROGRAMS
  nodes/bbac_main_node.py
  nodes/baseline_manager_node.py
  nodes/evaluator_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# 5. Install Experiments (Scripts)
install(PROGRAMS
  bbac_ics_core/experiments/ablation_study.py
  bbac_ics_core/experiments/adaptive_eval.py
  bbac_ics_core/experiments/dynamic_rules_test.py
  DESTINATION lib/${PROJECT_NAME}
)
# 6. Install Configs & Launch
install(DIRECTORY msg DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY data DESTINATION share/${PROJECT_NAME}/)

# 7. Testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  list(APPEND AMENT_LINT_AUTO_EXCLUDE
    ament_cmake_copyright
    ament_cmake_flake8
    ament_cmake_pep257
  )
  ament_lint_auto_find_test_dependencies()
  
  find_package(ament_cmake_pytest REQUIRED)
  # Se os testes usam o código, garante que o path está correto nos imports
  ament_add_pytest_test(test_ingestion tests/test_ingestion.py)
endif()

ament_package()
