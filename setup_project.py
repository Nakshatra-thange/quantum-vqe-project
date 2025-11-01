"""
Create the project directory structure for your VQE project
Run this script once to set up all folders and initial files
"""

import os
import json
from pathlib import Path

# Define project structure
project_name = "quantum-vqe-project"
structure = {
    "data": ["Raw data from VQE runs"],
    "notebooks": ["Jupyter notebooks for interactive development"],
    "src": ["Source code modules"],
    "figures": ["Publication-quality plots"],
    "docs": ["Documentation and reports"],
    "results": ["Processed results and analysis"]
}

print(f"Creating project structure for: {project_name}")
print("=" * 60)

# Create main project directory
Path(project_name).mkdir(exist_ok=True)
os.chdir(project_name)

# Create subdirectories
for folder, description in structure.items():
    Path(folder).mkdir(exist_ok=True)
    print(f"✓ Created {folder}/ - {description[0]}")
    
    # Create .gitkeep to track empty folders
    with open(f"{folder}/.gitkeep", "w") as f:
        f.write("")

# Create requirements.txt
requirements = """# Core quantum computing packages
qiskit==1.0.0
qiskit-nature==0.7.2
qiskit-aer==0.13.3
pyscf==2.5.0

# Scientific computing
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0

# Optional: Jupyter support
notebook>=7.0.0
ipywidgets>=8.0.0

# Optional: Real hardware access
# qiskit-ibm-runtime>=0.15.0
"""

with open("requirements.txt", "w") as f:
    f.write(requirements)
print("✓ Created requirements.txt")

# Create README.md template
readme = """# VQE for Molecular Hamiltonians

Implementation of Variational Quantum Eigensolver (VQE) for computing ground state energies of molecular systems.

## Project Overview
This project demonstrates quantum algorithms for electronic structure calculations, comparing VQE performance against classical quantum chemistry methods.

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start
```python
# Run verification tests
python tests/verify_installation.py

# Day 1: Basic tests
python tests/day1_tests.py
```

## Project Structure
- `data/` - Raw experimental data
- `notebooks/` - Interactive Jupyter notebooks
- `src/` - Core implementation modules
- `figures/` - Generated plots and visualizations
- `docs/` - Documentation and reports
- `results/` - Processed results

## Progress Tracker
- [x] Day 1: Environment setup ✓
- [ ] Day 2: Classical benchmark
- [ ] Day 3: VQE implementation
- [ ] Day 4: Optimization
- [ ] Day 5: Noise analysis
- [ ] Day 6: Extensions
- [ ] Day 7: Documentation

## Author
[Your Name]

## License
MIT License
"""

with open("README.md", "w") as f:
    f.write(readme)
print("✓ Created README.md")

# Create .gitignore
gitignore = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Data files (large)
*.h5
*.hdf5
*.dat

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Results (keep structure but not large files)
data/*.json
results/*.pkl
"""

with open(".gitignore", "w") as f:
    f.write(gitignore)
print("✓ Created .gitignore")

# Create initial config file
config = {
    "project_name": "VQE Materials Hamiltonians",
    "version": "0.1.0",
    "target_systems": ["H2", "H4", "Hubbard"],
    "default_basis": "sto-3g",
    "default_shots": 1024,
    "simulators": ["statevector", "qasm"],
    "optimizers": ["COBYLA", "SLSQP", "SPSA"],
    "day1_completed": True
}

with open("config.json", "w") as f:
    json.dump(config, f, indent=2)
print("✓ Created config.json")

# Create source module templates
src_files = {
    "hamiltonians.py": '''"""
Hamiltonian generation and manipulation
"""

def generate_h2_hamiltonian(bond_length=0.735, basis='sto-3g'):
    """Generate H2 molecule Hamiltonian"""
    pass

def generate_hubbard_hamiltonian(num_sites, u_over_t=2.0):
    """Generate 1D Hubbard model Hamiltonian"""
    pass
''',
    "ansatz.py": '''"""
Ansatz circuit builders
"""

def hardware_efficient_ansatz(num_qubits, num_layers):
    """Build hardware efficient ansatz"""
    pass

def uccsd_ansatz(num_qubits, num_electrons):
    """Build UCCSD ansatz"""
    pass
''',
    "vqe_runner.py": '''"""
Main VQE execution engine
"""

def run_vqe(hamiltonian, ansatz, optimizer='COBYLA'):
    """Execute VQE algorithm"""
    pass
''',
    "analysis.py": '''"""
Analysis and visualization utilities
"""

def plot_convergence(energies, exact_energy):
    """Plot VQE convergence"""
    pass
'''
}

for filename, content in src_files.items():
    with open(f"src/{filename}", "w") as f:
        f.write(content)
    print(f"✓ Created src/{filename}")

# Create __init__.py for src package
with open("src/__init__.py", "w") as f:
    f.write('"""VQE implementation package"""\n')

print("\n" + "=" * 60)
print("Project structure created successfully!")
print("=" * 60)
print("\nNext steps:")
print("1. Navigate to project: cd quantum-vqe-project")
print("2. Copy day1_tests.py to this directory")
print("3. Run: python day1_tests.py")
print("4. Start working on Day 2!")
print("\nTo initialize git:")
print("  git init")
print("  git add .")
print('  git commit -m "Initial project structure"')