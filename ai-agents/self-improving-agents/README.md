# self-improving-agents

## Overview
This project demonstrates self-improving agents using three core machine learning paradigms:
- **Supervised Learning:** Classification of handwritten digits (MNIST).
- **Unsupervised Learning:** Clustering of customer data (KMeans).
- **Reinforcement Learning:** Training an agent to solve the CartPole environment.

## Architecture
- Modular codebase with separate folders for each paradigm.
- Shared utilities for data handling and evaluation.
- Jupyter Notebooks for interactive exploration and visualization.

## Tech Stack
- Python 3.9+
- numpy, pandas, matplotlib, seaborn
- scikit-learn
- torch or tensorflow (for deep learning, optional)
- gymnasium or openai-gym (for RL)
- jupyter (optional, for notebooks)

## Getting Started

### 1. Clone the Repository
```sh
git clone <repo-url>
cd self-improving-agents
```

### 2. Set Up Virtual Environment
```sh
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Mac/Linux
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

### 4. Run Example Notebooks or Scripts
- Supervised: `python supervised/main.py` or open `notebooks/supervised.ipynb`
- Unsupervised: `python unsupervised/main.py` or open `notebooks/unsupervised.ipynb`
- Reinforcement: `python reinforcement/main.py` or open `notebooks/reinforcement.ipynb`

## Use Cases

### 1. Supervised Learning: MNIST Digit Classification
- **Goal:** Train a model to classify handwritten digits.
- **Dataset:** MNIST (downloaded automatically).
- **Model:** Logistic Regression or simple Neural Network.
- **Steps:**
  1. Load and preprocess data.
  2. Train model.
  3. Evaluate accuracy.
  4. Save and load model.

### 2. Unsupervised Learning: Customer Segmentation
- **Goal:** Cluster customers based on purchasing behavior.
- **Dataset:** Synthetic or UCI customer data.
- **Model:** KMeans clustering.
- **Steps:**
  1. Load and preprocess data.
  2. Apply KMeans.
  3. Visualize clusters.
  4. Analyze cluster characteristics.

### 3. Reinforcement Learning: CartPole Balancing
- **Goal:** Train an agent to balance a pole on a cart.
- **Environment:** OpenAI Gym CartPole-v1.
- **Algorithm:** Q-Learning or Deep Q-Network (DQN).
- **Steps:**
  1. Initialize environment and agent.
  2. Train agent over episodes.
  3. Track rewards and performance.
  4. Save and reload agent.

## Performance Metrics
- **Supervised:** Accuracy, confusion matrix.
- **Unsupervised:** Silhouette score, cluster visualization.
- **Reinforcement:** Average reward per episode, success rate.

## Documentation (Please refer notebooks)
- [Supervised Learning Details](docs/supervised.md)
- [Unsupervised Learning Details](docs/unsupervised.md)
- [Reinforcement Learning Details](docs/reinforcement.md)

## Contributing
Pull requests are welcome. For major changes, please open an issue first.

## License
[MIT](LICENSE)
