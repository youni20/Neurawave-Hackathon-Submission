import os

# IGNORE THESE FOLDERS
IGNORE_DIRS = {
    "node_modules", "venv", ".git", "__pycache__", "dist", "build", ".idea", ".vscode"
}

# IGNORE THESE FILES
IGNORE_FILES = {
    "package-lock.json", "yarn.lock", ".DS_Store", "neurawave.db", "users.json", "predictions.json", "full_context.txt"
}

# ONLY READ THESE EXTENSIONS
ALLOWED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".css", ".html", ".json", ".md", ".txt"
}

def generate_snapshot():
    output = ["# 🚀 NEURAWAVE PROJECT SNAPSHOT\n"]
    
    # Walk through the current directory
    for root, dirs, files in os.walk("."):
        # Filter out ignored directories in-place
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            if file in IGNORE_FILES:
                continue
            
            ext = os.path.splitext(file)[1]
            if ext not in ALLOWED_EXTENSIONS:
                continue

            # Avoid reading the snapshot script itself
            if file == "snapshot.py":
                continue

            file_path = os.path.join(root, file)
            
            # Add file header
            output.append(f"\n{'='*50}")
            output.append(f"\nFILE: {file_path}")
            output.append(f"\n{'='*50}\n")
            
            # Read and add file content
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    output.append(content)
            except Exception as e:
                output.append(f"Error reading file: {e}")

    # Save to a file for easy copying
    with open("full_context.txt", "w", encoding="utf-8") as f:
        f.write("".join(output))

    print("✅ Snapshot generated in 'full_context.txt'")
    print("📋 Copy the contents of that file and paste it into Gemini!")

if __name__ == "__main__":
    generate_snapshot()