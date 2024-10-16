from pathlib import Path

import setuptools

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
    name="streamlitBackCameraInput",
    version="0.1.0",
    author="Pham Xuan Tien",
    author_email="phamxtien@gmail.com",
    description="Custom camera_input with rear camera as default, just click on video to capture!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    data_files=[
        (
            "frontend",
            [
                "./src/streamlit_back_camera_input/frontend/main.js",
                "./src/streamlit_back_camera_input/frontend/style.css",
                "./src/streamlit_back_camera_input/frontend/index.html",
                "./src/streamlit_back_camera_input/frontend/streamlit-component-lib.js",
            ],
        )
    ],
    include_package_data=True,
    classifiers=[],
    python_requires=">=3.10",
    install_requires=["streamlit>=1.15", "jinja2"],
    url="https://github.com/phamxtien/streamlit_back_camera_input",
)
