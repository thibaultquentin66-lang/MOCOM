import traci
import torch
import os
import sys
from brain import TrafficBrain

MODEL_PATH = "models/policy_ep50.pth" 
YELLOW_DURATION = 3
SIM_DURATION = 3000  

# --- SAFETY LOGIC ---
def set_safe_phase(junction_id, target_phase):
    current_phase = traci.trafficlight.getPhase(junction_id)
    if current_phase == target_phase: return 0 

    traci.trafficlight.setPhase(junction_id, current_phase + 1)
    for _ in range(YELLOW_DURATION): traci.simulationStep()
    traci.trafficlight.setPhase(junction_id, target_phase)
    return YELLOW_DURATION

# --- MAX PRESSURE LOGIC ---
def get_max_pressure_action():
    # 1. Inputs (Waiting Cars)
    n_in = traci.edge.getLastStepHaltingNumber("n_in")
    s_in = traci.edge.getLastStepHaltingNumber("s_in")
    e_in = traci.edge.getLastStepHaltingNumber("e_in")
    w_in = traci.edge.getLastStepHaltingNumber("w_in")
    
    # 2. Outputs (Leaving Cars)
    n_out = traci.edge.getLastStepVehicleNumber("n_out")
    s_out = traci.edge.getLastStepVehicleNumber("s_out")
    e_out = traci.edge.getLastStepVehicleNumber("e_out")
    w_out = traci.edge.getLastStepVehicleNumber("w_out")
    
    # 3. Pressure = In - Out
    pres_ns = (n_in + s_in) - (n_out + s_out)
    pres_ew = (e_in + w_in) - (e_out + w_out)
    
    return 0 if pres_ns > pres_ew else 2

# --- SIMULATION RUNNER ---
def run_simulation(mode="AI", output_file="tripinfo_ai.xml"):
    print(f"Running Simulation in {mode} mode...")
    sumo_cmd = ["sumo-gui", "-c", "ff_heterogeneous.sumocfg", "--tripinfo-output", output_file]
    traci.start(sumo_cmd)
    
    brain = None
    if mode == "AI":
        brain = TrafficBrain()
        if os.path.exists(MODEL_PATH):
            brain.load(MODEL_PATH)
            brain.eval()
        else:
            print("ERROR: Model not found!")
            sys.exit()
    
    while traci.simulation.getTime() < SIM_DURATION:
        target_phase = -1
        
        # --- DECISION MAKING ---
        if mode == "AI":
            q_n = traci.edge.getLastStepHaltingNumber("n_in")
            q_s = traci.edge.getLastStepHaltingNumber("s_in")
            q_e = traci.edge.getLastStepHaltingNumber("e_in")
            q_w = traci.edge.getLastStepHaltingNumber("w_in")
            state = torch.FloatTensor([q_n/50, q_s/50, q_e/50, q_w/50])
            probs = brain(state)
            action = torch.argmax(probs).item()
            target_phase = 0 if action == 0 else 2
            
        elif mode == "MaxPressure":
            target_phase = get_max_pressure_action()
            
        # --- ACT ---
        if target_phase != -1:
            set_safe_phase("C", target_phase)
            
        traci.simulationStep()
    
    traci.close()
    print(f"Finished {mode}. Saved to {output_file}")

if __name__ == "__main__":
    run_simulation(mode="Normal", output_file="tripinfo_normal.xml")
    run_simulation(mode="MaxPressure", output_file="tripinfo_maxpressure.xml")
    run_simulation(mode="AI", output_file="tripinfo_ai.xml")
    print("\n✅ Done! Now run 'compare_results.py'")