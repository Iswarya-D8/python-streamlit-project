🤖 Code Mentor AI

Code Mentor AI is an AI-powered web application that helps you explain code, fix errors, analyze complexity, and optimize Python code.

It uses Streamlit for the frontend and Hugging Face Transformers (with PyTorch) for the backend.

---

🚀 Features

- Explain Code – Line-by-line simple explanations
- Fix Errors – Detects syntax and logic errors with corrections
- Analyze Complexity – Provides Time & Space complexity (Big-O)
- Optimize Code – Suggests cleaner and faster implementations
- Dark Theme UI – Clean editor-style interface

---

🧠 Model Used

- "deepseek-coder-1.3b-instruct" (runs locally)

---

🛠️ Tech Stack

- Python
- Streamlit
- PyTorch
- Hugging Face Transformers

---

📦 Prerequisites

Make sure you have:

- Python 3.8+
- Git

«Note: If you have an NVIDIA GPU, the app will use it for faster performance.»

---

⚙️ Setup Instructions

1. Install dependencies

pip install streamlit torch transformers accelerate

(For GPU users, install CUDA-enabled PyTorch from the official PyTorch website.)

2. Run the application

streamlit run app.py

---

▶️ Usage

- The app opens in your browser (usually http://localhost:8501)
- The AI model downloads automatically on first run
- Paste your Python code into the editor
- Click any feature button to analyze your code

---

👩‍💻 Author

Iswarya
