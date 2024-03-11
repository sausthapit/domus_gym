from setuptools import find_packages, setup

setup(
    name="domus_gym",
    version="0.2",
    packages=find_packages(
        where=".",
        include=["domus_gym"],
        exclude=["domus_mlsim"],
    ),
    install_requires=[
        "gym",
        "domus_mlsim",
        "scipy",
        "stable-baselines3",
    ],  # And any other dependencies domus-gym needs
)
