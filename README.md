# DeepOptimalOptionStopping
This is a repository showcasing deep learning approaches for pricing American options. It implements two optimal stopping methods :

- **Deep Longstaff-Schwartz:** Uses a neural network in a Monte Carlo simulation to estimate continuation values.
- **Deep Optimal Stopping:** Applies reinforcement learning to learn the optimal exercise strategy.

---

## Bibliography 

- [Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American options by simulation: a simple least-squares approach. *The Review of Financial Studies, 14*(1), 113-147.](https://people.math.ethz.ch/~hjfurrer/teaching/LongstaffSchwartzAmericanOptionsLeastSquareMonteCarlo.pdf)
- [Becker, S., Cheridito, P., & Jentzen, A. (2019). Deep optimal stopping. *Mathematical Finance, 29*(3), 606-620.](https://jmlr.org/papers/volume20/18-232/18-232.pdf)

---

## Setup

1. **Clone the repository:**

   `git clone https://github.com/Shiaroku/DeepOptimalOptionPricing.git`  
   `cd DeepOptimalOptionPricing`

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

