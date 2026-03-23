# Info about used grid Model
The IEEE 39-Bus New England System is a widely used benchmark in power system research and education. It represents a simplified model of the New England power grid in the United States, capturing the key characteristics of a medium-sized transmission network. The system consists of:

- 39 buses (nodes) representing substations or load points
- 10 generators providing electrical power at different locations
- 46 transmission lines connecting buses and forming the network topology
- 19 loads representing electrical demand at various buses

This test system is often used for studies of power flow analysis, state estimation, stability assessment, and control strategies. Its moderate size and realistic structure make it suitable for developing and benchmarking new computational methods, including machine learning and graph-based approaches such as Graph Neural Networks (GNNs).
In this project, the IEEE 39-bus system was used to generate training data for GNN-based state estimation, by simulating different operating conditions, varying generator outputs, and performing power flow calculations to capture a wide range of system states.

<img width="475" height="423" alt="image" src="https://github.com/user-attachments/assets/66360a8b-42f6-4165-8c5f-8152fdcbeee6" />

