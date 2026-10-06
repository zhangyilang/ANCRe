import setuptools

setuptools.setup(
    name="ancre",
    version="0.0.1",
    author="Yilang Zhang, Bingcong Li, Niao He, Georgios B. Giannakis",
    description="Adaptive Neural Connection Reassignment (ANCRe)",
    long_description_content_type="text/markdown",
    url="",
    license="Apache-2.0",
    packages=setuptools.find_packages(include=["ancre", "ancre.*"]),
    install_requires=["torch"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)