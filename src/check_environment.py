import sys
import importlib

def check_environment():
    print(f"Python version: {sys.version}")
    
    packages_to_check = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),
        ('xgboost', 'xgboost'),
        ('rapidfuzz', 'rapidfuzz'),
        ('scipy', 'scipy')
    ]
    
    all_passed = True
    print("\nPackage Status:")
    for module_name, package_name in packages_to_check:
        try:
            importlib.import_module(module_name)
            print(f"{package_name}: PASS")
        except ImportError:
            print(f"{package_name}: FAIL")
            all_passed = False
            
    print("\nFinal Overall Environment Status:")
    if all_passed:
        print("PASS - All required packages are available.")
    else:
        print("FAIL - Some required packages are missing.")

if __name__ == "__main__":
    check_environment()
