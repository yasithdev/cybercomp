# CyberCompute - An Extensible Platform for Computational Experiments

## Functionality

- Searching computational models and execution engines registered in cybercompute.
- Defining experiments by coupling computational models with execution engines, and defining their choice of parameter values.
- Defining larger experiments through composition of smaller experiments.
- Defining experiment collections by grouping a set of experiments with common observations.
- Comparing and contrasting observations within/across experiment collections.

## System Design

- System should find which systems out of N systems are compatible, and suggest them to users
- System should type-match parameters/observations when coupling models and engines, and when composing larger experiments.
- system should validate first (before execution) and throw error

## Terminology

### Model

### Engine

### Parameters

Inputs of scientific interest to an experiment

- initial values for ODE
- weights of inference-mode NNs

### Observations

Outputs of scientific interest from an experiment

- variables (e.g. time)
- constants (e.g., total energy)

## Examples

### Astrophysics

Q: Given a "solar system" with "parameters" and "initial conditions", what is the gravitational force X between planet X and planet Y at time T?

### Neuroscience

Q: Given a "neuronal network" with "parameters" "initial conditions", when does neuron X exhibit a membrane voltage < Y?

### Computational Physics

Q: Given a "double pendulum" with "parameters" and "initial conditions", what is the trajectory (X, Y) followed by the second bob?

## Roadmap

1. Neuroscience (Giri)
2. Deep Learning (Giri, Chris)
3. Quantum Chemistry (Sudhakar)
4. Geoscience (Dimuthu)
5. Molecular Dynamics (Sudhakar)
