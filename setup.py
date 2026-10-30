from setuptools import find_packages, setup

setup(
    name="legged_gym",
    version="2.1.0",
    author="Nikita Rudin",
    license="BSD-3-Clause",
    license_files=["LICENSE", "licenses/*", "resources/robots/UNITREE_LICENSE"],
    packages=find_packages(
        include=["legged_gym", "legged_gym.*", "spectral_prior", "spectral_prior.*"],
        exclude=["tests", "tests.*", "*.tests", "*.tests.*"],
    )
    + ["legged_gym.resources"],
    package_dir={"legged_gym.resources": "resources"},
    author_email="rudinn@ethz.ch",
    description="Unofficial SPRINT spectral priors and residual PPO in legged_gym format",
    python_requires=">=3.10,<3.14",
    include_package_data=True,
    package_data={
        "spectral_prior": ["configs/*.json"],
        "legged_gym.resources": [
            "robots/UNITREE_LICENSE",
            "robots/provenance.json",
            "robots/g1/g1_29dof.xml",
            "robots/g1/meshes/*",
        ],
    },
    install_requires=[
        "mujoco==3.17.0",
        "rsl-rl-lib==5.5.2",
        "torch>=2.6.0",
        "numpy>=1.26",
        "scipy>=1.11",
        "tensordict>=0.7.0",
        "tensorboard",
        "viser>=1.1.1,<2",
        "mjviser==0.0.12",
    ],
    extras_require={"viewer": []},
    entry_points={
        "console_scripts": [
            "legged-train=legged_gym.scripts.train:main",
            "legged-play=legged_gym.scripts.play:main",
            "legged-evaluate=legged_gym.scripts.evaluate:main",
            "spectral-prior=spectral_prior.cli:main",
            "sprint-train=legged_gym.scripts.train:main",
            "sprint-play=legged_gym.scripts.play:main",
            "sprint-evaluate=legged_gym.scripts.evaluate:main",
            "sprint-reference=legged_gym.scripts.reference:main",
        ]
    },
)
