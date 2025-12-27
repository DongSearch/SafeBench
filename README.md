## 🎯 Goal: 
Enhance planning defense in SafeBench benchmarking by leveraging FlowMatching and Signed Distance Functions (SDF). The system takes vehicle state information as input, predicts the next planning step, and detects out-of-distribution (OOD) scenarios.

## 🛠 Algorithms Used:
- **RL (Soft Actor-Critic)**: Learns normal driving behavior.
- **Flow Matching**: Focuses on vehicle trajectory, steering, speed, and lane-out information. During unexpected turns (left/right), it recognizes OOD scenarios and reduces vehicle speed to handle them safely.
- **SDF (Signed Distance Function)**: Learns collision-related responses, detecting front objects and sudden appearances regardless of normal driving patterns (side objects ignored). range is between 0 to 1. 0 is dangerous, 1 is safe.
- **Training Data**: Scenario 1 with random sampling (straight driving with sudden pedestrian appearances).
- **Proaction**: Incorporates soft/hard braking based on scores.
- **Workflow**: After RL learns normal driving, Flow Matching and SDF networks are trained. Flow Matching emphasizes trajectory control, while SDF handles sudden collision threats, improving planning defense.

## ✨ Edited Files
- **`safebench/agent/rl/FFM.py`**  
  - Added Flow Matching functionality
- **`safebench/agent/rl/sac_fm.py`**  
  - Integrated Flow Matching  
  - Added OOD evaluation on top of basic SAC.py
- **`safebench/agent/config/sac_fm.yaml`**  
  - Added YAML configuration for SAC + Flow Matching
- **`safebench/gym_carla/envs/carla_env.py`**  
  - Added human detection (previously only vehicles were detected)  
  - Edited reward function

## 📅 Key milestone
- **Sep 30:** Established CARLA and SafeBench environments on a remote server.
- **Oct 10:** Set up a simulation pipeline to evaluate scenarios via recorded videos (offline rendering).
- **Oct 15:** Optimized driving performance by transitioning from Single-Q to **Double-Q SAC**. 
    > *Note: Training basic driving from scratch was challenging due to the simultaneous tuning of rewards, parameters, and scores.*
- **Oct 30:** **[Trial & Error]** Realized that SAC and Flow Matching must be trained separately; joint training led to model divergence.
- **Nov 05:** **[Trial & Error]** Realized that SAC must not be trained with steering information, which lead to change bias when car turns and can't detect the OOD.
- **Nov 10:** Successfully stabilized the SAC model for basic driving maneuvers.
- **Nov 20:** Implemented **Pro-active Action** logic to enhance safety responses.
- **Nov 29:** **[Optimization]** Trained Flow Matching independently. Identified that training with pro-active actions simultaneously blurred the distinction between normal and abnormal driving.
- **Dec 10:** Fine-tuned hyperparameters and score thresholds based on evaluation metrics.
- **Dec 20:** Conducted comprehensive result analysis and visualization.
- **Dec 30:** Finalized the project report.

## Result
### 🎬 Scenario 01: DynamicObjectCrossing ✅
| Normal | with FM & SDF |
| :---: | :---: |
| <video src="https://github.com/user-attachments/assets/19076e98-29f2-469a-9443-73c1b8906e47" width="100%"></video> | <video src="https://github.com/user-attachments/assets/bcf5fc9c-2d08-4d1d-8e5d-2fc4a3d0576c" width="100%"></video> |

<img width="1200" height="600" alt="report_epbatch_2" src="https://github.com/user-attachments/assets/653cdc99-6491-4af3-a8d2-22cc0b97a6fb" />

#### 📈 Result Analysis
When a pedestrian crosses the road, the SDF score drops below 0.5, triggering an immediate stop. Once the pedestrian has cleared the path, the vehicle resumes its movement.

### 🎬 Scenario 02: VehicleTurningRoute ⚠️
| Normal | with FM & SDF |
| :---: | :---: |
| <video src="https://github.com/user-attachments/assets/a3ff2497-3b26-468d-a215-640af410d28b" width="100%"></video> | <video src="https://github.com/user-attachments/assets/0e7d0b9c-6ffd-4196-8351-37ee79f58ed6" width="100%"></video> |

<img width="1200" height="600" alt="report_epbatch_6" src="https://github.com/user-attachments/assets/a637ca0c-7aa4-423b-8a44-ddf507e7b960" />


#### 📈 Result Analysis
While turning is unsmooth, the FM score ensures functional deceleration. However, the discrepancy between the car's orientation and the pedestrian's direction limits the reliability of the SDF score in this specific scenario.

### 🎬 Scenario 03: OtherLeadingVehicle ✅
| Normal | with FM & SDF |
| :---: | :---: |
| <video src="https://github.com/user-attachments/assets/fed797f1-bce8-4a82-9c2a-adbcf716c629" width="100%"></video> | <video src="https://github.com/user-attachments/assets/ab0626d7-6a1d-4f5a-a0f0-56a01471c246" width="100%"></video> |

<img width="1200" height="600" alt="report_epbatch_10" src="https://github.com/user-attachments/assets/5b345f3e-3a4d-4755-9d1e-4695687bb3f5" />



#### 📈 Result Analysis
The FM score is maintained under normal conditions, but an SDF score below 0.5 signals danger and initiates braking. The vehicle resumes normal operation once the SDF score recovers (> 0.5) as the front car moves away.

### 🎬 Scenario 04: ManeuverOppositeDirection ✅
| Normal | with FM & SDF |
| :---: | :---: |
| <video src="https://github.com/user-attachments/assets/6ebd8224-b7e2-47a6-b9c6-580b2145368a" width="100%"></video> | <video src="https://github.com/user-attachments/assets/0656e968-4496-4f59-8645-44c7023402ab" width="100%"></video> |

<img width="1200" height="600" alt="report_epbatch_14" src="https://github.com/user-attachments/assets/0d1e2cd0-ee60-430e-a896-e32684fe35d8" />

#### 📈 Result Analysis
The FM score is maintained under normal conditions, but an SDF score below 0.5 signals danger and initiates braking.

### 🎬 Scenario 05: OppositeVehicleRunningRedLight ⚠️
| Normal | with FM & SDF | 
| :---: | :---: |
| <video src="https://github.com/user-attachments/assets/b74c443e-ac8c-4d84-95ac-3623e9c8c53c" width="100%"></video> | <video src="https://github.com/user-attachments/assets/fb19ce05-1ab5-4f82-9c10-8c0fe7fda07d" width="100%"></video> |

<img width="1200" height="600" alt="report_epbatch_18" src="https://github.com/user-attachments/assets/3c9c8312-310b-4c7f-8fdd-89995ae3304c" />


#### 📈 Result Analysis
Due to the state data being limited to frontal input, SDF fluctuations (below 0.5) are only triggered when a vehicle from the left merges into the front trajectory of the ego car

### 🎬 Scenario 06: TurnLeftAtSignalizedJunction ❌
| Normal | with FM & SDF |
| :---: | :---: |
| <video src="https://github.com/user-attachments/assets/4ced841a-8a12-486d-9d85-d121c49e66fb" width="100%"></video> | <video src="https://github.com/user-attachments/assets/29fba73f-d8d0-4911-a89f-745247f992a8" width="100%"></video> |

<img width="1200" height="600" alt="report_epbatch_22" src="https://github.com/user-attachments/assets/24ce2f24-9c03-4e3c-9271-a7f14a5f1985" />

#### 📈 Result Analysis
SDF-based braking is restricted by the frontal-only data. Fortunately, the FM score initiates a slowdown during left turns, providing a safety buffer even when the SDF score remains unresponsive

### 🎬 Scenario 07: TurnLeftAtSignalizedJunction ✅✅
| Normal | with FM & SDF |
| :---: | :---: |
| <video src="https://github.com/user-attachments/assets/1a25a961-2422-43e6-bbd6-23d1ac102088" width="100%"></video> | <video src="https://github.com/user-attachments/assets/c3a21867-8eed-4511-b4a1-c540de59b13e" width="100%"></video> |

<img width="1200" height="600" alt="report_epbatch_26" src="https://github.com/user-attachments/assets/db2069cf-1750-4efa-9be7-e167a450a90d" />

#### 📈 Result Analysis
This scenario provides a clear example for evaluating both SDF and FM scores. As shown in the logs, the ego car first detects the obstacle and initiates emergency braking. Once the path is clear, it successfully executes a right turn.

### 🎬 Scenario 08: NoSignalJunctionCrossingRoute ⚠️
| Normal | with FM & SDF |
| :---: | :---: |
| <video src="https://github.com/user-attachments/assets/7eaf11e6-92c4-48a4-8cf2-b4904ff4bb41" width="100%"></video> | <video src="https://github.com/user-attachments/assets/f0ff0f26-9a5c-456f-9674-0e326623881f" width="100%"></video> |

<img width="1200" height="600" alt="report_epbatch_30" src="https://github.com/user-attachments/assets/51de3576-29ed-4487-913a-458423c621d4" />

#### 📈 Result Analysis
Due to the state data being limited to frontal input, SDF fluctuations (below 0.5) are only triggered when a vehicle from the left merges into the front trajectory of the ego car


### Overall result

The Flow Matching (FM) algorithm proved remarkably efficient. Even when trained on only 10 episodes using a single random scenario (straight driving with a pedestrian), it successfully learned to follow the intended route. A key contribution of this work is the decoupling of SDF and FM: using SDF to prioritize collision avoidance while leveraging FM for precise route following, allowing each module to specialize in its respective safety and navigation tasks.

