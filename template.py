import os

# Directory and file structure
structure = {
    
        "requirements.txt": "",
        ".env.example": "MONGO_URI=\nOSM_API_URL=\n",
        "README.md": "# Cattle Identification Backend (Python + Flask)\n",
        "app": {
            "__init__.py": "# Flask app factory\n",
            "config.py": "# Configuration loader\n",
            "db.py": "# MongoDB connection setup\n",
            "models.py": "# MongoDB models and helpers\n",
            "routes.py": "# Flask API routes\n",
            "utils.py": "# Utility helpers (notifications)\n",
            "errors.py": "# Custom error handlers\n",
            "services": {
                "osm_service.py": "# OSM Overpass API service\n",
                "identification.py": "# Object identification and filtering logic\n",
            }
        },
        "run.py": "# Entry point\nfrom app import create_app\n\napp = create_app()\n\nif __name__ == '__main__':\n    app.run(debug=True)\n"
    
}

def create_structure(base_path, tree):
    for name, content in tree.items():
        path = os.path.join(base_path, name)

        if isinstance(content, dict):
            # Create directory
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            # Create file
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

# Run the structure creator
create_structure(".", structure)

print("✔ Project structure created successfully!")
