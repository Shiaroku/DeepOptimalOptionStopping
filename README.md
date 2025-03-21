# DeepOptimalOptionStopping
This is a repository showcasing deep learning approaches for pricing American options. It implements two optimal stopping methods :

- **Deep Longstaff-Schwartz:** Uses a neural network in a Monte Carlo simulation to estimate continuation values.
- **Deep Optimal Stopping:** Applies reinforcement learning to learn the optimal exercise strategy.

---

## Project Structure

DeepOptimalOptionPricing/  
├── README.md             # This file  
├── LICENSE               # Project license (MIT)  
├── requirements.txt      # Project dependencies  
├── data/                 # Data or simulation outputs  
├── notebooks/            # Jupyter notebooks for experiments  
├── src/                  # Source code (models, data generator, etc.)  
└── tests/                # Unit tests

---

## Setup

1. **Clone the repository:**

   git clone https://github.com/Shiaroku/DeepOptimalOptionPricing.git  
   cd DeepOptimalOptionPricing

2. **Install dependencies:**

   pip install -r requirements.txt

---

## Usage

- **Notebooks:** Explore the implementations and experiments in the `notebooks/` directory.
- **Source Code:** Find the core implementations in the `src/` directory.
- **Tests:** Run unit tests with `pytest tests/` to verify functionality.

---

## License

This project is licensed under the MIT License.

