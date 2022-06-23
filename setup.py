from setuptools import find_packages, setup

setup(
    name="domus_gym",
    version="0.1",
    packages=find_packages(
        where=".",
        include=["domus_gym"],
        exclude=["domus_mlsim"],
    ),
    install_requires=[
        "gym",
        "domus_mlsim",
        "scipy <= 1.5.2",
        "stable-baselines3",
    ],  # And any other dependencies domus-gym needs
)
