# Application-of-machine-learning-to-power-system-state-estimation
This project is part of my master’s thesis and focuses on the application of Machine Learning methods to power system state estimation.
The work includes an overview of the theoretical foundations of both Machine Learning and classical state estimation techniques. It also highlights the key differences between Power Flow analysis and State Estimation, with emphasis on their roles in power system operation and analysis.

The practical part of the project consists of:
- building a power system model in PowerFactory,
- extracting simulation data used as a dataset,
- designing and implementing a neural network in Python using existing libraries (e.g., PyTorch, PyTorch Geometric),
- training the model and evaluating its performance.
  
The main objective of the project was to investigate whether Machine Learning-based approaches can serve as an effective alternative or complement to traditional state estimation methods, particularly in terms of accuracy and robustness to noisy or incomplete data.

You can find whole project in AP_mgr file, with table of content written in english in description

# Why is State Etimation important 
<img width="704" height="448" alt="image" src="https://github.com/user-attachments/assets/69ade6d3-fc56-41fe-bc57-747ef5aca673" />

Source "Smart Grid: The Electrical Grid of the Future Sharad Bhowmick January 10, 2022"


The electrical grid of the future is becoming increasingly complex, driven by the integration of renewable energy sources, distributed generation, and advanced monitoring technologies. Despite these transformations, one concept has remained constant: at the heart of every power system lies a supervisory and control framework that ensures reliable and secure operation. This control layer enables continuous monitoring, decision-making, and coordination, allowing operators to maintain full visibility and control over the system.

# What is State Estimation
“A mathematical process of determining the most probable values of variables describing the state of a power system based on redundant and error-prone measurements.”

State estimation is a fundamental tool in power system operation, aimed at reconstructing the actual state of the grid from imperfect measurement data. In practice, measurements collected from the system are often noisy, incomplete, or even inconsistent. State estimation algorithms process this data and provide the best possible approximation of key system variables, such as bus voltages and phase angles. By filtering out measurement errors and leveraging redundancy, the method delivers a coherent and reliable representation of the system, which is essential for monitoring, control, and decision-making processes.

<img width="942" height="229" alt="Zrzut ekranu 2026-03-23 o 18 17 39" src="https://github.com/user-attachments/assets/be3b2e3b-c270-4474-8593-c4ca43619f12" />


The difference between Power Flow and State Estimation lies primarily in the nature of the input data and the purpose of the analysis. Power Flow assumes that all necessary system parameters, such as loads and generation, are fully known and uses them to deterministically compute system states like bus voltages and power flows. In contrast, State Estimation operates in a more realistic environment, where measurements are incomplete, redundant, and affected by noise. Instead of relying on perfect input data, it uses available measurements (e.g., power flows, injections, voltages) to infer the most probable state of the system. As a result, while Power Flow is mainly used for planning and simulation, State Estimation is essential for real-time monitoring and control of the power grid.

# Electrical Grid as Graph
A power system can be naturally represented as a graph structure, where buses correspond to nodes and transmission lines to edges. This representation captures the physical topology of the network as well as the relationships between its components. Each node can store features such as voltage magnitude or phase angle, while edges can represent electrical parameters like impedance or power flow. 

<img width="1576" height="512" alt="image" src="https://github.com/user-attachments/assets/8fda429c-3941-4ef2-875c-9715f83f65c8" />
Węzeł = Bus,
Krawędź = Line,
Atrybut = Attriubute

Modeling the grid as a graph is particularly useful in the context of Machine Learning, as it allows the use of graph-based methods that explicitly take into account the connectivity and dependencies within the system, rather than treating the data as independent observations.

## Graph Neural Network
The use of neural networks for power system state estimation is motivated by their ability to learn complex nonlinear relationships and handle large volumes of measurement data. Traditional optimization-based methods may struggle with complex network topologies or incomplete data. In this context, Graph Neural Networks (GNNs) are particularly well-suited because they naturally model the graph structure of power systems, where nodes represent buses or measurement points and edges represent electrical connections. By aggregating information from neighboring nodes, GNNs capture both local and global dependencies, enabling more accurate and robust state estimation even under partial or noisy measurements.

<img width="801" height="243" alt="Zrzut ekranu 2026-03-23 o 19 11 08" src="https://github.com/user-attachments/assets/321761ac-234c-46b7-a904-66055c102828" />

Source "A Crash Course on Graph Neural Networks Avi Chawla 25 Aug 2024"


Furthermore, by learning better representations of the network topology and measurement interactions, GNNs provide a more expressive model, adhering to the principle that better representation leads to better models, which is crucial for modern monitoring, control, and optimization of electrical networks.

# Methodology

<img width="800" height="183" alt="Zrzut ekranu 2026-03-23 o 19 17 00" src="https://github.com/user-attachments/assets/2e1ad3d5-74f2-4275-9fa4-6297ee881b74" />

The research methodology consisted of several sequential steps. First, a power system model was created in PowerFactory, including all relevant buses, generators, and loads. Within this model, a simulation loop was implemented, which iteratively modified generator outputs, performed power flow calculations, and recorded the resulting system states. This process generated a dataset capturing a wide range of operating conditions and network behaviors. Next, the collected data was used to construct and train a Graph Neural Network (GNN) using the PyTorch Geometric (PyG) library. The network leveraged the graph structure of the power system, with nodes representing buses and edges representing electrical connections, to learn the mapping from generator settings and network topology to the resulting system states. This methodology enabled the creation of a data-driven model capable of accurate state estimation under varying operating conditions.
