"""
Dependency Checker for Digital Twin City Simulation
=====================================================
Run this script to verify all dependencies are installed correctly.
"""

import sys

def check_dependencies():
    """Check if all required dependencies are installed"""
    
    print("🔍 Checking dependencies...\n")
    
    dependencies = {
        'torch': 'PyTorch',
        'torch_geometric': 'PyTorch Geometric',
        'networkx': 'NetworkX',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'plotly': 'Plotly',
        'streamlit': 'Streamlit',
        'pandas': 'Pandas',
        'flask': 'Flask',
        'flask_cors': 'Flask-CORS',
        'waitress': 'Waitress',
        'tqdm': 'TQDM',
        'matplotlib': 'Matplotlib'
    }
    
    all_good = True
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            version = __import__(module).__version__ if hasattr(__import__(module), '__version__') else 'installed'
            print(f"✅ {name:25s} - {version}")
        except ImportError:
            print(f"❌ {name:25s} - NOT INSTALLED")
            all_good = False
    
    print("\n" + "="*70)
    
    if all_good:
        print("✅ All dependencies are installed!")
        print("\nYou can now run:")
        print("  • Backend API: python backend/app.py")
        print("  • Streamlit GUI: streamlit run streamlit_gui.py")
    else:
        print("❌ Some dependencies are missing!")
        print("\nTo install missing dependencies, run:")
        print("  pip install -r requirements.txt")
    
    print("="*70)
    
    return all_good


if __name__ == "__main__":
    check_dependencies()
