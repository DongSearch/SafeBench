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
- **`safebench/agent/model_ckpt/sac_fm`**  
  - add trained model

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
| <video src="https://github.com/user-attachments/assets/8041c3c6-fedd-4b0b-ae7a-875deb7a7c66" width="100%"></video> | <video src="https://github.com/user-attachments/assets/b02197e1-b7ad-4231-ab43-2aa8501a0823" width="100%"></video> |

<img width="1200" height="600" alt="report_epbatch_2" src="https://github.com/user-attachments/assets/7093cd74-6914-4f64-8dcf-0893ad832a84" />





#### 📈 Result Analysis
When a pedestrian crosses the road, the SDF score drops below 0.5, triggering an immediate stop. Once the pedestrian has cleared the path, the vehicle resumes its movement.

### 🎬 Scenario 02: VehicleTurningRoute ⚠️
| Normal | with FM & SDF |
| :---: | :---: |
| <video src="https://github.com/user-attachments/assets/c699a938-be8e-471a-8b3a-ccb32424ab50" width="100%"></video> | <video src="https://github.com/user-attachments/assets/740ef2db-3fee-4549-98e0-6d0697baf101" width="100%"></video> |

<img width="1200" height="600" alt="report_epbatch_6" src="https://github.com/user-attachments/assets/392641e9-deb2-45c6-9760-764f2599cc49" />






#### 📈 Result Analysis
While turning is unsmooth, the FM score ensures functional deceleration. However, the discrepancy between the car's orientation and the pedestrian's direction limits the reliability of the SDF score in this specific scenario.

### 🎬 Scenario 03: OtherLeadingVehicle ✅
| Normal | with FM & SDF |
| :---: | :---: |
| <video src="https://github.com/user-attachments/assets/5847f8de-f87d-4296-a8b7-65208e5e1e49" width="100%"></video> | <video src="https://github.com/user-attachments/assets/786980c4-3dd7-4413-a396-94d6c8dde79c" width="100%"></video> |

<img width="1200" height="600" alt="report_epbatch_10" src="https://github.com/user-attachments/assets/92e9b53a-9230-4081-a637-06281a90a132" />





#### 📈 Result Analysis
The FM score is maintained under normal conditions, but an SDF score below 0.5 signals danger and initiates braking. The vehicle resumes normal operation once the SDF score recovers (> 0.5) as the front car moves away.

### 🎬 Scenario 04: ManeuverOppositeDirection ✅
| Normal | with FM & SDF |
| :---: | :---: |
| <video src="https://github.com/user-attachments/assets/e63fa5d5-4ee7-4dda-bf7b-83f31ba6562c" width="100%"></video> | <video src="https://github.com/user-attachments/assets/67d814f4-c59f-481e-b93b-882209704129" width="100%"></video> |

<img width="1200" height="600" alt="report_epbatch_14" src="https://github.com/user-attachments/assets/71fd9584-507c-475f-b53f-c167e1468468" />





#### 📈 Result Analysis
The FM score is maintained under normal conditions, but an SDF score below 0.5 signals danger and initiates braking.

### 🎬 Scenario 05: OppositeVehicleRunningRedLight ⚠️
| Normal | with FM & SDF | 
| :---: | :---: |
| <video src="https://github.com/user-attachments/assets/0d8b1d9f-1056-4083-b0b3-96ea74a6839c" width="100%"></video> | <video src="https://github.com/user-attachments/assets/0c13c264-62f2-483b-abdb-0ed299ff4bd1" width="100%"></video> |

<img width="1200" height="600" alt="report_epbatch_18" src="https://github.com/user-attachments/assets/021438ba-0a46-487b-97ac-2e7064024cf7" />









#### 📈 Result Analysis
Due to the state data being limited to frontal input, SDF fluctuations (below 0.5) are only triggered when a vehicle from the left merges into the front trajectory of the ego car

### 🎬 Scenario 06: TurnLeftAtSignalizedJunction ❌
| Normal | with FM & SDF |
| :---: | :---: |
| <video src="https://github.com/user-attachments/assets/90d03d3b-baa5-4524-a5f6-095b89359b8c" width="100%"></video> | <video src="https://github.com/user-attachments/assets/b1abb4fa-7160-48d8-9aa1-d43bafb7c9fd" width="100%"></video> |

<img width="1200" height="600" alt="report_epbatch_22" src="https://github.com/user-attachments/assets/c39834ba-57c7-491a-95e6-ec12fdb25810" />










#### 📈 Result Analysis
SDF-based braking is restricted by the frontal-only data. Fortunately, the FM score initiates a slowdown during left turns, providing a safety buffer even when the SDF score remains unresponsive

### 🎬 Scenario 07: TurnLeftAtSignalizedJunction ✅✅
| Normal | with FM & SDF |
| :---: | :---: |
| <video src="https://github.com/user-attachments/assets/ff1cf6dc-3511-4e24-83dc-08a8c0fca882" width="100%"></video> | <video src="https://github.com/user-attachments/assets/1141b8c9-c600-44b9-a93e-df591ab30f83" width="100%"></video> |

<img width="1200" height="600" alt="report_epbatch_26" src="https://github.com/user-attachments/assets/ffa99fa8-863d-44f3-a7c5-0caf7a46753e" />










#### 📈 Result Analysis
This scenario provides a clear example for evaluating both SDF and FM scores. As shown in the logs, the ego car first detects the obstacle and initiates emergency braking. Once the path is clear, it successfully executes a right turn.

### 🎬 Scenario 08: NoSignalJunctionCrossingRoute ⚠️
| Normal | with FM & SDF |
| :---: | :---: |
| <video src="https://github.com/user-attachments/assets/7f3afe95-8e58-443f-ad5a-8b78e9d57f89" width="100%"></video> | <video src="https://github.com/user-attachments/assets/2163cee5-2f2e-4849-a5cb-a297f087e460" width="100%"></video> |

<img width="1200" height="600" alt="report_epbatch_30" src="https://github.com/user-attachments/assets/3bb3fb32-6425-47ff-9155-8ef01a40db8d" />





#### 📈 Result Analysis
Due to the state data being limited to frontal input, SDF fluctuations (below 0.5) are only triggered when a vehicle from the left merges into the front trajectory of the ego car


### Overall result

The Flow Matching (FM) algorithm proved remarkably efficient. Even when trained on only 10 episodes using a single random scenario (straight driving with a pedestrian), it successfully learned to follow the intended route. A key contribution of this work is the decoupling of SDF and FM: using SDF to prioritize collision avoidance while leveraging FM for precise route following, allowing each module to specialize in its respective safety and navigation tasks.

