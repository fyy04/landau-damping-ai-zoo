# LandauDamping-Zoo
A collection / zoo of AI modeling, literature and baseline implementations for Landau damping.
> This repo is currently for literature sorting and baseline code collection.

## Project Overview
This repository collects related papers, research routes and baseline model implementations for Landau damping plasma simulation.
We aim to summarize existing AI approaches for Landau damping and build benchmark test cases for plasma dynamics prediction.

## Research Routes
1. Traditional HP closure + Neural network approximation
2. Discover explicit PDE/closure from dynamics data
3. PINN for learning implicit closure within single spatio-temporal domain
4. Offline fitting of physical quantities (q, dq/dx) with FNO
5. Embedding FNO into multi-moment equations for autonomous closed-loop simulation
6. Physics-constrained neural operators to directly generate multi-field trajectories
7. Online closed-loop training with history-dependent closure + differentiable solver
8. Research on effective closure, cross-parameter generalization and long-time stability

## Baseline Models
- U-Net
- FNO (Fourier Neural Operator)
- CNO

## Repository Structure
