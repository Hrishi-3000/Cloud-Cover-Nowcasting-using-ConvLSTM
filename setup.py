from setuptools import setup, find_packages

setup(
    name="cloud_cover_nowcasting",
    version="0.1.0",
    author="Hrishikesh Shahane",
    description="A package for cloud cover nowcasting using ConvLSTM.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "tensorflow>=2.9.0",
        "numpy",
        "opencv-python",
        "matplotlib",
        "scikit-image",
        "scikit-learn",
    ],
)
