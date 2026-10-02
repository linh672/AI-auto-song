import json
from pathlib import Path

# Import the configured OUTPUT_DIR from config so it's always accurate
from automation.config import OUTPUT_DIR

def update_preferences() -> None:
    pref_file = Path(r"C:\FlowBotProfile\Default\Preferences")
    if pref_file.exists():
        data = json.loads(pref_file.read_text("utf-8"))
        
        if "download" not in data:
            data["download"] = {}
            
        data["download"]["prompt_for_download"] = False
        data["download"]["default_directory"] = str(OUTPUT_DIR.resolve())
        data["download"]["directory_upgrade"] = True
        
        pref_file.write_text(json.dumps(data, indent=2), "utf-8")
        print(f"Updated Preferences successfully. Default directory set to: {OUTPUT_DIR}")
    else:
        print("Preferences file not found!")

if __name__ == "__main__":
    update_preferences()
