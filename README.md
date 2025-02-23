# EfficienGPT 🚀

## Overview 🌟

EfficienGPT is a **quick-learning engine** designed to optimize knowledge acquisition using the **Pareto Principle (80/20 Rule)**. By leveraging **GPT-4o**, **MongoDB Atlas**, and **Streamlit**, it provides **fast, insightful, and actionable learning experiences** tailored to individual needs.  

EfficienGPT allows users to:  
- Retrieve **focused insights** without getting overwhelmed by unnecessary details.  
- Store and access knowledge instantly with **MongoDB Atlas**.  
- Interact through a **clean, intuitive UI** powered by **Streamlit**.  

## Features ⚡

- **AI-powered Learning:** Uses GPT-4o to generate **precise and actionable insights**.  
- **Efficient Knowledge Storage:** Stores and retrieves insights securely via **MongoDB Atlas**.  
- **Streamlit UI:** Provides a **simple and distraction-free** interface for users.  
- **Pareto Principle Optimization:** Ensures that only the **most important 20% of knowledge** is delivered.  
- **Customizable Inputs:** Users define their **topic, available time, and use case** for tailored learning.  

---


## Setup & Installation 🛠️  

### 1. Clone the Repository  

```bash
git clone https://github.com/your-username/EfficienGPT.git
cd EfficienGPT
```

### 2. Create a Virtual Environment  

```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies  

```bash
pip install -r requirements.txt
```

---

## API Key Configuration 🔑  

EfficienGPT requires an **OpenAI API Key** to function properly.  

### 4. Generate an OpenAI API Key  

1. Go to the **[OpenAI API Keys page](https://platform.openai.com/signup/)**.  
2. Sign up or log in to your OpenAI account.  
3. Navigate to **API Keys** and create a new API key.  

**⚠️ Note:** Using OpenAI’s API **requires payment** after free-tier usage is exhausted. Be mindful of costs.  

### 5. Store the API Key  

Create a `.env` file in the project directory and add the following line:  

```plaintext
OPENAI_API_KEY=your_api_key_here
```

Alternatively, you can set it as an environment variable:  

```bash
export OPENAI_API_KEY="your_api_key_here"  # macOS/Linux
set OPENAI_API_KEY="your_api_key_here"    # Windows
```

---

## Running EfficienGPT  

### 6. Start the Streamlit App  

```bash
streamlit run Home.py
```

This will launch the web interface where you can interact with EfficienGPT.  

---

## Free Local Version (Using Ollama)  

If you want to run EfficienGPT **without incurring OpenAI API costs**, you can use **Ollama**, a local LLM runtime.  

### 1. Install Ollama  

Follow the installation guide here: **[Ollama Installation](https://ollama.ai/)**.  

### 2. Download a Local Model  

```bash
ollama pull mistral  # Or any other model you prefer
```

### 3. Run the Local Version  

Modify your `.env` file to use the local model:  

```plaintext
USE_LOCAL_MODEL=True
OLLAMA_MODEL=mistral
```

Then, run the Streamlit app as usual:  

```bash
streamlit run app.py
```

This will use **Ollama’s locally stored model** instead of the OpenAI API.  

---

## Contributing  

EfficienGPT is an evolving project. Contributions, feature requests, and bug reports are welcome!  

### Steps to Contribute:  
1. **Fork** the repository.  
2. Create a **new branch** for your feature/fix.  
3. **Commit** your changes.  
4. Open a **pull request**.  

---

## License  

This project is licensed under the **MIT License**. See the `LICENSE` file for details.  

---

## Credits  

EfficienGPT was built with ❤️ using:  
- **GPT-4o** for AI-powered insights.  
- **MongoDB Atlas** for efficient and secure knowledge storage.  
- **Streamlit** for an intuitive user experience.  
- **LangChain** for smart prompt engineering.  
```

This README provides a complete guide to setting up, running, and contributing to EfficienGPT. Let me know if you need any modifications! 🚀
