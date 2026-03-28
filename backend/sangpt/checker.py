import os

files_to_check = [
    "DATA/security/security_best_practices.txt",
    "DATA/security/ai_security_training_dataset.csv",
    "DATA/conversational/training_data.csv",
    "DATA/security/adversarial_red_team_scenarios.txt",
    "DATA/security/The AI Risk Repository V4_03_12_2025 - Contents.csv",
]

print("--- SANCTA FILE AUDIT ---")
for file_path in files_to_check:
    status = "[FOUND]" if os.path.exists(file_path) else "[MISSING]"
    print(f"{file_path:<70} {status}")
