from setuptools import setup, find_packages

# Read content from README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="box2read",  
    version="0.1.0",  
    author="Yann Terrom", 
    author_email="yannterrom@hotmail.fr",  
    description="Custom OCR framework.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yannTrm/Box2Read",  
    project_urls={ 
        "Bug Tracker": "https://github.com/yannTrm/Box2Read/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},  # Indique que les packages sont sous le dossier `src`
    packages=find_packages(where="src"),  # Trouve les packages dans `src/`
    python_requires=">=3.8",  # Version minimale de Python requise
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines() if line.strip()
    ],
    extras_require={  # Dépendances optionnelles
        "dev": ["pytest", "pytest-cov", "flake8", "black", "isort"],
        "docs": ["sphinx", "sphinx-rtd-theme"],
    },
    include_package_data=True, 
    entry_points={ 
        "console_scripts": [
            "ocr-train=src.training.trainer:main",
            "ocr-evaluate=src.evaluation.evaluator:main",
        ],
    },
)
