### Prerequisites

-   Python 3.8+
-   Git
-   **Git LFS:** This project uses Git LFS to manage the large model file. You must install it before cloning the repository. You can download it from [git-lfs.github.com](https://git-lfs.github.com/).

### Installation

After installing Git LFS, you can clone the repository and install the Python libraries:

```bash
# Clone the repository (Git LFS will automatically download the model)
git clone [your-github-repo-url]
cd [your-repo-name]

# Install Python dependencies
pip install "fastapi[all]" torch transformers sentencepiece scikit-learn numpy

Automated Content Moderation: A Transformer-based Toxicity Classifier
This repository contains the source code and trained model for an AI-powered web application designed to detect and classify multiple types of toxicity in text. The project serves as a complete, end-to-end example of a modern NLP pipeline, from data exploration and model fine-tuning to final deployment as a local web service.

This README documents the baseline model, which was trained on a 20,000-comment sample to establish initial performance and identify key challenges.

🚀 Features
Multi-Label Classification: Classifies text across 7 different toxicity categories: Toxic, Severe Toxicity, Obscene, Threat, Insult, Identity Attack, and Sexual Explicit.

Transformer-Powered: Utilizes the power of the pre-trained DistilBERT model for nuanced language understanding.

Interactive Web UI: A simple but effective frontend built with FastAPI and HTML that allows for real-time text analysis.

Ready to Run: Includes a self-contained local server setup for easy demonstration and testing.

🛠️ Project Workflow & Methodology
The project followed a structured machine learning workflow:

Data Exploration & Sampling: We began by analyzing the google/civil_comments dataset. Due to its large size (1.8M+ comments), we created a smaller, representative sample of 20,000 training and 2,000 validation examples to ensure rapid development.

Preprocessing & Tokenization: The text was converted into a model-readable format using the official distilbert-base-uncased tokenizer. A crucial data formatting step involved combining the multiple float-based label columns into a single labels tensor required by PyTorch for multi-label training.

Model Fine-Tuning: We fine-tuned the pre-trained DistilBERT model for 3 epochs on a T4 GPU. After encountering environment issues, we implemented a robust manual PyTorch training loop for greater stability and control.

Evaluation: The model's performance was measured using the micro-averaged F1 Score, a suitable metric for multi-label classification tasks, especially with imbalanced data.

🎯 Performance & Results (Baseline Model)
The primary goal of this baseline was to establish a working model and analyze its performance on unseen data.

Final Test F1 Score: 0.5800

This is a strong baseline score, proving the effectiveness of the fine-tuning approach. Analysis revealed that the model performed very well on common categories like toxicity and insult. Its main limitation was the severe class imbalance in the training data, which resulted in lower performance on rare categories like threat and identity_attack. This insight was key to planning future improvements (e.g., training on more data).

🔧 Key Challenges & Solutions
During development, we navigated several common machine learning challenges:

Challenge: The high-level Hugging Face Trainer API was unstable in the cloud environment.

Solution: We pivoted to a more fundamental and reliable manual PyTorch training loop, which gave us full control over the training process.

Challenge: Initial training runs were extremely slow.

Solution: We diagnosed the issue as the notebook defaulting to CPU. By correctly configuring the runtime to use a T4 GPU, we achieved a massive performance increase.

Challenge: The evaluation metric failed due to a data type mismatch.

Solution: We updated our compute_metrics function to binarize both the true labels and the model's predictions before calculating the F1 score, ensuring a fair comparison.

⚙️ Local Setup and Usage
Follow these steps to run the application on your local machine.

1. Prerequisites
Python 3.8+

pip for package installation

2. Installation
First, clone or download this repository. Then, navigate to the project directory in your terminal and install all the required libraries:

pip install "fastapi[all]" torch transformers sentencepiece scikit-learn numpy

3. Folder Structure
Ensure your project folder is organized exactly as follows for the application to work correctly:

toxicity_app/
|
|-- my-toxic-classifier/   <-- FOLDER containing the trained model files
|   |-- config.json
|   |-- model.safetensors
|   |-- vocab.txt
|   |-- ... (and other model files)
|
|-- templates/             <-- FOLDER for the website template
|   |-- index.html
|
|-- main.py                <-- The Python backend script
|
|-- README.md              <-- This file

4. Running the Application
Open a terminal and navigate to the root of your project folder (toxicity_app/).

Run the following command to start the web server:

uvicorn main:app --reload

Once the server is running, open your web browser and go to the following address:
http://127.0.0.1:8000

You can now enter any text into the text box and click "Analyze" to see the model's predictions.
