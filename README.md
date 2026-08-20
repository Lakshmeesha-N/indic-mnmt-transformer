# MMT (Multimodal Translation & Modeling)

A modular repository designed for Multimodal Machine Translation (MMT), deep learning workflows, and data processing pipelines.

## 📁 Repository Structure

```text
MMT/
├── data/              # Datasets (raw, preprocessed, and train/val splits)
├── models/            # Model checkpoints, saved weights, and configs
├── scripts/           # Training, evaluation, and inference scripts
├── README.md          # Project documentation
└── requirements.txt   # Python dependency specifications
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- `pip` package manager
- (Optional) NVIDIA GPU with CUDA support for accelerated model training/inference

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/MMT.git
   cd MMT
   ```

2. **Set up a virtual environment:**
   ```bash
   # On Windows:
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux:
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Development Workflow

- **Data Management:** Place raw data and dataset pre-processing tools inside `data/`.
- **Model Checkpoints:** Store trained weights and architecture configurations in `models/`.
- **Execution Scripts:** Place pipeline execution, training loops, and evaluation code inside `scripts/`.

## 🧪 Running Tests & Formatting

Run tests with `pytest`:
```bash
pytest
```

Format code using `black`:
```bash
black .
```

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
