import subprocess
import datetime
from dotenv import load_dotenv
import os
import json
import time
from datetime import datetime
import openai

load_dotenv()

def load_config():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(script_dir, 'config.json')
        with open(config_file, 'r') as f:
            config = json.load(f)
            
            if 'tree' not in config or 'system_commands' not in config:
                raise KeyError("Missing required keys in config.json")
            
            return config
    except (FileNotFoundError, KeyError):
        print("⚠️ config.json not found or invalid, using default settings")
        return {
            "tree": {
                "max_depth": 3,
                "ignore_patterns": ["venv", "__pycache__", "node_modules", "build", "public", "dist", ".git"],
                "ignore_extensions": ["*.pyc", "*.pyo", "*.pyd", "*.so", "*.dll", "*.class"]
            },
            "system_commands": {
                "disk_usage_threshold": 80
            }
        }

config = load_config()

ignore_patterns = '|'.join(config['tree']['ignore_patterns'])
ignore_extensions = '|'.join(config['tree']['ignore_extensions'])
combined_ignore = f"{ignore_patterns}|{ignore_extensions}"

system_commands = [
    "uname -a",
    "python --version",
    "pip list",
    "python -c 'import sys; print(sys.path)'",
    f"df -h | awk '(NR==1) || ($5+0 >= {config['system_commands']['disk_usage_threshold']})'",
    "vm_stat | awk '/Pages free:|Pages active:|Pages inactive:|Pages wired down:/ {print}'",
    "netstat -an | grep LISTEN | awk '{print $4}' | sort -u",
    "printenv | grep -v 'KEY\\|SECRET\\|PASS\\|TOKEN'",
]

project_commands = [
    f"tree -d -L {config['tree']['max_depth']} -I '{combined_ignore}'",
]

def get_project_directories():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        projects_file = os.path.join(script_dir, 'containers-list.md')
        
        with open(projects_file, 'r') as f:
            projects = [
                line.strip('- \n') 
                for line in f.readlines() 
                if line.strip().startswith('-')
            ]
        return projects
    except FileNotFoundError:
        print("❌ Error: containers-list.md not found in .cursorboost directory")
        return []

def run_commands(commands_list, project_dir=None):
    print(f"📊 Collecting {'project' if project_dir else 'system'} information...")
    output = []
    output.append(f"Timestamp: {datetime.utcnow().isoformat()}Z\n")
    
    if project_dir:
        original_dir = os.getcwd()
        base_project_path = config.get('base_project_path')
        if not base_project_path or base_project_path == "<your_project_path>":
            raise ValueError(
                "❌ Error: 'base_project_path' is not defined or is still set to '<your_project_path>' in the configuration file (config.json). "
                "Please update it to your project's root directory."
            )
        
        project_path = os.path.join(os.path.expanduser(base_project_path), project_dir)
        try:
            os.chdir(project_path)
            output.append(f"\n### Project: {project_dir} ###\n")
        except FileNotFoundError:
            print(f"  ❌ Project directory not found: {project_path}")
            return ""

    for command in commands_list:
        print(f"  ⚡ Running: {command}")
        try:
            result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            output.append(f"{command}:\n{result.decode('utf-8')}\n")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed: {command}")
            output.append(f"{command} (FAILED):\n{e.output.decode('utf-8')}\n")
    
    if project_dir:
        os.chdir(original_dir)
    
    return "\n".join(output)

def write_snapshot(snapshot):
    with open("snapshot.txt", "w") as f:
        f.write(snapshot)

def generate_cursorrules(snapshot):
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        return None
    
    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that injects a user's system information and parses out the most important details. Only respond with the complete file contents, without any additional explanation or commentary."},
            {"role": "user", "content": f"You are a coding assistant optimizing text files for LLM-based applications. Using the following system snapshot, generate a text file that highlights the most relevant details for coding context. Only output the complete file contents, without explanations or extra text:\n\n{snapshot}"}
        ],
        temperature=1,
    )
    return response.choices[0].message.content

def write_cursorrules(cursorrules):
    project_description = ""
    try:
        with open("project-description.txt", "r") as f:
            project_description = f"# Project Description:\n{f.read()}\n\n"
    except FileNotFoundError:
        print("Warning: project-description.txt not found")
    
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cursorrules_path = os.path.join(parent_dir, '.cursorrules')
    
    with open(cursorrules_path, "w") as f:
        f.write(f"{project_description}{cursorrules}")

def find_requirements_files(root_dir):
    requirements_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if '.cursorboost' in dirpath:
            continue
        
        if 'requirements.txt' in filenames:
            requirements_files.append(os.path.join(dirpath, 'requirements.txt'))
    
    return requirements_files

root_directory = os.getcwd()
requirements_files = find_requirements_files(root_directory)
print("Found requirements files:", requirements_files)

if __name__ == "__main__":
    print("🚀 Starting Cursor Boost in continuous mode...")
    
    while True:
        print(f"\n⏰ Running update at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        projects = get_project_directories()
        if not projects:
            print("⚠️ No projects found in containers-list.md")
        
        system_snapshot = run_commands(system_commands)
        
        all_snapshots = [system_snapshot]
        for project in projects:
            print(f"📂 Processing project: {project}")
            project_snapshot = run_commands(project_commands, project)
            if project_snapshot:
                all_snapshots.append(project_snapshot)
        
        complete_snapshot = "\n\n".join(all_snapshots)
        
        write_snapshot(complete_snapshot)
        
        cursorrules = generate_cursorrules(complete_snapshot)
        if cursorrules:
            write_cursorrules(cursorrules)
            print("✅ .cursorrules updated successfully")
            print("✅ Snapshot saved to snapshot.txt")
        else:
            print("❌ Failed to generate .cursorrules")
            
        time.sleep(60)