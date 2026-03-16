<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/Flower-Federated_Learning-5C2D91?style=for-the-badge" />
  <img src="https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/AWS-EC2-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white" />
</p>

<h1 align="center">FedMed-LLM</h1>

<p align="center">
  <strong>Privacy-Preserving Medical Q&A via Federated LLM Fine-Tuning</strong>
</p>

<p align="center">
  A full-stack medical question-answering chatbot powered by <strong>Microsoft Phi-2 (2.7B)</strong>,<br/>
  fine-tuned with <strong>Federated Learning</strong> and <strong>LoRA</strong> on the MedQuAD dataset,<br/>
  deployed as a streaming React + FastAPI application on AWS EC2.
</p>

<p align="center">
  <a href="#demo">Demo</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#key-features">Features</a> •
  <a href="#tech-stack">Tech Stack</a> •
  <a href="#federated-learning-pipeline">FL Pipeline</a> •
  <a href="#evaluation-results">Evaluation</a> •
  <a href="#getting-started">Setup</a> •
  <a href="#deployment">Deployment</a>
</p>

---



<p align="center">
  <img src="docs/screenshots/chat_demo.gif" alt="FedMed-LLM Chat Demo" width="700"/>
</p>

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FedMed-LLM Architecture                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐         │
│  │  Hospital A  │     │  Hospital B  │     │  Hospital C  │         │
│  │  (Client 1)  │     │  (Client 2)  │     │  (Client 3)  │         │
│  │              │     │              │     │              │         │
│  │  Local Data  │     │  Local Data  │     │  Local Data  │         │
│  │  MedQuAD     │     │  MedQuAD     │     │  MedQuAD     │         │
│  │  Partition 1 │     │  Partition 2 │     │  Partition 3 │         │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘         │
│         │  LoRA Weights      │  LoRA Weights      │  LoRA Weights   │
│         └─────────────┬──────┴──────┬─────────────┘                 │
│                       ▼              ▼                              │
│              ┌────────────────────────────┐                         │
│              │    Flower FL Server        │                         │
│              │    FedAvg Aggregation      │                         │
│              │    + DP Noise Injection    │                         │
│              │    (σ = 0.01 Gaussian)     │                         │
│              └────────────┬───────────────┘                         │
│                           ▼                                         │
│              ┌────────────────────────────┐                         │
│              │    Merged Phi-2 + LoRA     │                         │
│              │    Production Model        │                         │
│              └────────────┬───────────────┘                         │
│                           ▼                                         │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │                   Deployment Stack                       │       │
│  │                                                          │       │
│  │  ┌─────────────┐    ┌─────────────┐   ┌──────────────┐   │       │
│  │  │   React +   │──▶│   FastAPI    │──▶│  Phi-2 +    │   │       │
│  │  │   Tailwind  │   │   (SSE)      │   │  LoRA Model  │   │       │
│  │  │   Frontend  │◀──│   Backend    │◀──│  Inference  │   │       │
│  │  └─────────────┘   └──────┬───────┘   └──────────────┘   │       │
│  │                           │                              │       │
│  │                    ┌──────▼───────┐                      │       │
│  │                    │   Supabase   │                      │       │
│  │                    │   (Auth + DB)│                      │       │
│  │                    └──────────────┘                      │       │
│  └──────────────────────────────────────────────────────────┘       │
│                                                                     │
│                    Docker Compose on AWS EC2                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 🧠 Federated Learning Training Pipeline
- **3 simulated hospital nodes** each with private MedQuAD data partitions
- **Flower framework** orchestrating FedAvg aggregation across 15 federated rounds
- **LoRA fine-tuning** (r=32, alpha=64) on Microsoft Phi-2 (2.7B parameters)
- **4-bit quantization** (BitsAndBytes NF4) for memory-efficient training on Colab GPUs
- **Round-by-round checkpointing** to Google Drive with automatic resume logic

### 🔒 Privacy Preservation
- **Federated Learning**: Raw medical data never leaves individual hospital nodes
- **Differential Privacy**: Gaussian noise injection (σ=0.01) in FedAvg aggregation
- **Gradient Clipping**: `max_grad_norm=1.0` to bound sensitivity
- Simulates **HIPAA-compliant** distributed training architecture

### 💬 Streaming Medical Chatbot
- **Server-Sent Events (SSE)** for real-time token-by-token streaming responses
- **JWT authentication** with secure user registration and login
- **Conversation history** with sidebar navigation
- **Suggested medical questions** for quick interaction
- **Automatic medical disclaimers** appended to every response

### 🎨 Modern Frontend
- **Dark theme UI** with gray-950 background and emerald accent palette
- **Fully responsive** — optimized for desktop, tablet, and mobile
- **Chat history sidebar** with conversation management
- Built with **React 18 + Vite + Tailwind CSS**

### 🚀 Production Deployment
- **Docker Compose** with multi-stage builds (backend ~2GB, frontend ~50MB)
- **Nginx reverse proxy** for static asset serving and API routing
- **AWS EC2** deployment with configurable instance types
- **Rate limiting** (10 req/min per user), structured JSON logging, input validation

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM Base Model** | Microsoft Phi-2 (2.7B) |
| **Fine-Tuning** | LoRA (PEFT 0.10.0), 4-bit QLoRA via BitsAndBytes |
| **Federated Learning** | Flower (flwr), FedAvg strategy |
| **ML Framework** | PyTorch, HuggingFace Transformers 4.40.0, Accelerate 0.29.0 |
| **Dataset** | MedQuAD (Medical Question-Answer Dataset) |
| **Backend** | FastAPI, Uvicorn, Server-Sent Events (SSE) |
| **Auth & Database** | Supabase (PostgreSQL), JWT (PyJWT + bcrypt) |
| **Frontend** | React 18, Vite, Tailwind CSS |
| **Containerization** | Docker, Docker Compose, Nginx |
| **Cloud** | AWS EC2, Google Colab (training), Google Drive (checkpoints) |
| **Testing** | pytest |

---

## Federated Learning Pipeline

### Training Configuration

```
Model:             Microsoft Phi-2 (2.7B parameters)
Quantization:      4-bit NF4 (BitsAndBytes)
LoRA Rank:         r = 32
LoRA Alpha:        64
LoRA Target:       q_proj, k_proj, v_proj, dense
Federated Rounds:  15
Clients per Round: 3
Samples per Node:  6,000
Local Epochs:      2 per round
Learning Rate:     Tuned via hyperparameter sweep (Day 2)
Scheduler:         Cosine with warmup
Optimizer:         AdamW (paged, 8-bit)
Batch Size:        4 (with gradient accumulation = 4 → effective 16)
Max Sequence Len:  512 tokens
```

### Critical Training Fix

A gradient checkpointing bug was identified and resolved during development. The fix requires calling `model.enable_input_require_grads()` **after** `prepare_model_for_kbit_training()` with `use_gradient_checkpointing=False`:

```python
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
model.enable_input_require_grads()  # Critical — must come after prepare
```

Without this fix, training loss stalls around ~2.6 and the model fails to converge.

### Hyperparameter Sweep

Three configurations were tested over 3 federated rounds each:

| Config | Learning Rate | Batch Size | Final Loss (3 rounds) |
|--------|--------------|------------|----------------------|
| A | 2e-4 | 8 | *See sweep_results.json* |
| B | 5e-5 | 4 | *See sweep_results.json* |
| C (baseline) | 1e-4 | 4 | *See sweep_results.json* |

The winning configuration was used for the final 15-round production run.

### Loss Curve

<p align="center">
  <img src="docs/loss_curve_v2.png" alt="Training Loss Curve — 15 Federated Rounds" width="600"/>
</p>

---

## Evaluation Results

### Automated Metrics

Evaluated on 200 held-out test samples from MedQuAD, comparing fine-tuned FedMed-LLM against base Phi-2:

| Metric | Base Phi-2 | FedMed-LLM (15 rounds) | Δ Improvement |
|--------|-----------|------------------------|---------------|
| **ROUGE-1** | — | — | — |
| **ROUGE-2** | — | — | — |
| **ROUGE-L** | — | — | — |
| **BLEU** | — | — | — |

> 📝 *Fill in after running `evaluation_results.json` from the Day 4 evaluation notebook.*

### Manual Quality Assessment

20 randomly sampled medical questions scored by human reviewers:

| Criterion | Base Phi-2 (avg) | FedMed-LLM (avg) |
|-----------|-----------------|-------------------|
| **Relevance** (1-5) | — | — |
| **Accuracy** (1-5) | — | — |

> ⚠️ **Disclaimer:** FedMed-LLM is a research prototype. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

---

## Privacy Analysis

FedMed-LLM implements a two-layer privacy architecture:

### Layer 1 — Federated Learning
Patient medical data is **never centralized**. Each hospital node trains locally on its own partition of MedQuAD, sharing only model weight updates (LoRA adapter deltas) with the central server.

### Layer 2 — Differential Privacy
Gaussian noise (σ=0.01) is injected during FedAvg aggregation to prevent reconstruction of individual training samples from aggregated weights. Combined with gradient clipping (`max_grad_norm=1.0`), this bounds the maximum influence of any single data point.

### Threat Model
| Threat | Mitigation |
|--------|-----------|
| Curious central server inspecting updates | DP noise masks individual contributions |
| Model inversion attacks | Gradient clipping + noise injection |
| Data leakage via memorization | LoRA limits trainable parameters (~1.5% of model) |
| Raw data exposure | Data never leaves client nodes |

> See [`privacy_analysis.md`](docs/privacy_analysis.md) for the full threat model documentation.

---

## Project Structure

```
fedmed-llm/
├── frontend/                     # React frontend application
│   ├── src/
│   │   ├── pages/
│   │   │   ├── RegisterPage.jsx  # User registration & login
│   │   │   └── ChatPage.jsx      # Main chat interface with SSE streaming
│   │   ├── App.jsx               # Router setup
│   │   └── main.jsx              # Entry point
│   ├── vite.config.js            # Vite config with /api proxy
│   ├── tailwind.config.js        # Dark theme + emerald accents
│   ├── Dockerfile                # Multi-stage: node build → nginx serve
│   └── package.json
│
├── backend/                      # FastAPI backend
│   ├── main.py                   # App entry — all endpoints
│   ├── requirements.txt          # Python dependencies
│   ├── backend_tests.py          # pytest test suite
│   ├── Dockerfile                # python:3.11-slim multi-stage
│   └── .env.example              # Template for environment variables
│
├── training/                     # Colab training notebooks
│   ├── colab_day1_v2_training.ipynb    # 10-round FL training (r=32)
│   ├── colab_day2_sweep.ipynb          # Hyperparameter sweep (3 configs)
│   ├── colab_day3_final_training.ipynb # 15-round production training
│   ├── evaluation_notebook.ipynb       # ROUGE/BLEU evaluation
│   └── dp_simulation.ipynb             # Differential privacy analysis
│
├── docs/
│   ├── privacy_analysis.md       # Threat model & DP documentation
│   ├── loss_curve_v3.png         # Training loss visualization
│   ├── architecture_diagram.png  # System architecture
│   └── screenshots/              # UI screenshots & demo GIF
│
├── docker-compose.yml            # Full stack orchestration
├── nginx.conf                    # Reverse proxy configuration
├── sweep_results.json            # Hyperparameter sweep outcomes
├── evaluation_results.json       # ROUGE/BLEU metric results
└── README.md                     # This file
```

---

## Getting Started

### Prerequisites

- **Python 3.11+**
- **Node.js 20+**
- **Docker & Docker Compose** (for containerized deployment)
- **Supabase account** (free tier works)
- **Google Colab** with L4/A100 GPU (for training only)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/fedmed-llm.git
cd fedmed-llm
```

### 2. Set Up Supabase

Create a new Supabase project and run the SQL schema to set up auth tables. Then create your `.env` file:

```bash
cp backend/.env.example backend/.env
```

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
JWT_SECRET=your-strong-secret-key
MODEL_PATH=./model/merged_phi2_fedmed
```

### 3. Run the Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be live at `http://localhost:8000`. Test it:

```bash
curl http://localhost:8000/api/health
```

### 4. Run the Frontend

```bash
cd frontend
npm install
npm run dev
```

The app opens at `http://localhost:5173`. Vite automatically proxies `/api` requests to `localhost:8000`.

### 5. Run Tests

```bash
cd backend
pytest backend_tests.py -v
```

---

## Docker Deployment

### Build & Run Locally

```bash
docker-compose up --build
```

This starts three containers:

| Container | Description | Port |
|-----------|-------------|------|
| `fedmed-backend` | FastAPI + Phi-2 inference | 8000 |
| `fedmed-frontend` | React app served via Nginx | 80 |
| `fedmed-nginx` | Reverse proxy | 80 (public) |

### Verify

```bash
curl http://localhost/api/health
# → {"status": "healthy", "model_loaded": true, "uptime": "..."}
```

---

## Deployment

### AWS EC2

1. **Launch instance**: `g4dn.xlarge` (GPU inference) or `t2.medium` (CPU-only demo)
2. **Security groups**: Open ports 80 (HTTP), 443 (HTTPS), 22 (SSH)
3. **Install Docker**:
   ```bash
   sudo apt update && sudo apt install -y docker.io docker-compose
   sudo usermod -aG docker $USER
   ```
4. **Upload model** (from Google Drive):
   ```bash
   scp -r merged_phi2_fedmed/ ec2-user@<EC2-IP>:~/fedmed-llm/model/
   ```
5. **Deploy**:
   ```bash
   docker-compose up -d
   ```
6. **Access**: `http://<EC2-PUBLIC-IP>/`

> 💡 **Cost tip**: Use Spot Instances during development (70% savings). Switch to On-Demand for recruiter demos.

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/api/health` | ❌ | Server health, model status, uptime |
| `POST` | `/api/register` | ❌ | Create new user account |
| `POST` | `/api/login` | ❌ | Authenticate and receive JWT |
| `POST` | `/api/chat` | ✅ | Send question, receive SSE stream |
| `GET` | `/api/history` | ✅ | Retrieve conversation history |

### Example: Streaming Chat

```bash
curl -N -H "Authorization: Bearer <token>" \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the symptoms of diabetes?"}' \
     http://localhost:8000/api/chat
```

Response streams as SSE events:

```
data: {"token": "Diabetes"}
data: {"token": " symptoms"}
data: {"token": " include"}
...
data: {"token": "[DONE]"}
```

---

## Training Notebooks

All training is designed to run on **Google Colab** with L4 or A100 GPUs.

| Notebook | Purpose | Duration |
|----------|---------|----------|
| `colab_day1_v2_training.ipynb` | 10-round FL training with improved config | ~8–9 hrs |
| `colab_day2_sweep.ipynb` | Hyperparameter sweep (3 configs × 3 rounds) | ~7.5 hrs total |
| `colab_day3_final_training.ipynb` | 15-round production training | ~11 hrs (2 sessions) |

### Colab Tips

- **Idle timeout protection**: Use the browser console keep-alive snippet included in each notebook
- **Resume logic**: Every notebook saves after each round to Google Drive — resume from any round if the session disconnects
- **NumPy compatibility**: Requires `numpy==1.26.4` (NumPy 2.0 crashes the `datasets` library)
- **Required packages**:
  ```
  transformers==4.40.0
  peft==0.10.0
  accelerate==0.29.0
  datasets==2.19.0
  bitsandbytes==0.45.5
  flwr (Flower)
  ```

---

## Key Learnings

1. **Gradient checkpointing bug**: `enable_input_require_grads()` must be called after `prepare_model_for_kbit_training()` — without it, loss plateaus at ~2.6
2. **NumPy 2.0 incompatibility**: Pin to `numpy==1.26.4` to avoid crashes in HuggingFace `datasets`
3. **Colab session management**: Round-by-round checkpointing to Google Drive is essential; sessions disconnect unpredictably
4. **LoRA efficiency**: Only ~1.5% of model parameters are trained, making federated communication lightweight
5. **Medical AI responsibility**: Always append disclaimers — this is both ethically required and a strong portfolio talking point

---

## Scopus-Indexed Publication

> **Federated Learning for Medical Applications** — Published in Springer (July 2025), Scopus-indexed.

This project builds on the research foundations from the above publication, extending the work into a full-stack deployable system with differential privacy simulation.

---

## Author

**Purushotham Reddy Tiyyagura**

B.Tech Computer Science — Amrita Vishwa Vidyapeetham


---

## Resume Bullets

For quick copy-paste into your resume:

- **Fine-tuned Microsoft Phi-2 (2.7B) via Federated Learning** using Flower framework — 15 rounds across 3 simulated hospital nodes with LoRA (r=32), achieving ROUGE-L improvement of +X% over base model on MedQuAD medical Q&A
- **Implemented differential privacy noise injection** (σ=0.01 Gaussian) in FedAvg aggregation with gradient clipping to simulate HIPAA-compliant distributed training for sensitive medical data
- **Deployed streaming medical chatbot** with FastAPI (SSE) + React on AWS EC2 via Docker Compose, featuring JWT authentication, rate limiting, conversation history, and automatic medical safety disclaimers

---

## License

This project is for educational and portfolio purposes. The MedQuAD dataset and Phi-2 model are subject to their respective licenses.

---

<p align="center">
  <strong>FedMed-LLM</strong> — Where privacy meets medical AI🔒
</p>



## Training Results

| Round | Avg Loss |
|-------|----------|
| 1     | 1.2416   |
| 2     | 1.1487   |
| 3     | 1.1150   |
| 4     | 1.0887   |
| 5     | 1.0658   |
| 6     | 1.0453   |
| 7     | 1.0257   |
| 8     | 1.0070   |
| 9     | 0.9887   |
| 10    | 0.9710   |

**Total improvement: 21.8% over 10 federated rounds**
