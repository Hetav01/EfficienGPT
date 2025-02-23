from dotenv import load_dotenv
from warnings import filterwarnings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel, RunnableSequence
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from bson.objectid import ObjectId
import os
from utils.db_functions import *

#ignore warnings
filterwarnings("ignore")

# Load environment variables from .env or HOME PATH file
load_dotenv()

#clear the database before running the code


# New MongoDB connection for storing embeddings
client = get_mongo_client()
rag_db = client[os.getenv("MONGO_RAG_DB")]
collection = rag_db["embeddings"]

embedding_model = OllamaEmbeddings(model="nomic-embed-text")

#generate embeddings for a piece of text

def generate_embeddings(text):
    embedding = embedding_model.embed_query(text)
    return embedding

#store the embeddings in the database
def store_embeddings(text, embedding):
    entry = {
        "text": text,
        "embedding": embedding
    }
    result = collection.insert_one(entry)
    return str(result.inserted_id)


#retrieve input text from another database
text= """
    Response for topic 'Day 1: Foundations and Basics of BERT':
**Day 1: Foundations and Basics of BERT**

**1. Overview of Transformers:**
   - Understand the central role of transformers, introduced by the Vaswani et al. paper "Attention is All You Need."
   - Key Concepts:
     - **Self-Attention Mechanism:** Calculate the importance of each word in a sequence relative to other words.
     - **Multi-head Attention:** Allow the model to focus on different words at different times.

   **Formula:**
   ```
   Attention(Q, K, V) = softmax((QK^T) / sqrt(d_k)) V
   ```
   Where:
   - `Q` (Query), `K` (Key), `V` (Value) are derived from inputs.
   - `d_k` is the dimension of key/query.

**2. Basics of BERT:**
   - BERT stands for Bidirectional Encoder Representations from Transformers.
   - Key Features of BERT:
     - **Bidirectional:** Understands the context from both left and right of a word (as opposed to directional models which read text sequentially).
     - **Pre-training Tasks:** 
       - **Masked Language Model (MLM):** Randomly masks a proportion of input tokens and predicts them, fostering contextual understanding.
       - **Next Sentence Prediction (NSP):** Helps in understanding sentence relationships.

   **Code Snippet to Use BERT from Hugging Face:**
   ```python
   from transformers import BertTokenizer, BertForMaskedLM
   import torch

   # Load pre-trained model tokenizer (vocabulary)
   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

   # Encode text
   text = "[CLS] I want to [MASK] the cheerful world [SEP]"
   tokenized_text = tokenizer.tokenize(text)
   indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

   # Convert inputs to PyTorch tensors
   tokens_tensor = torch.tensor([indexed_tokens])

   # Load pre-trained model
   model = BertForMaskedLM.from_pretrained('bert-base-uncased')
   model.eval()

   # Predict all tokens
   with torch.no_grad():
       outputs = model(tokens_tensor)
       predictions = outputs[0]
       
   # Decode the predicted token
   predicted_index = torch.argmax(predictions[0, 3]).item()
   predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
   print("Predicted token:", predicted_token)
   ```

**3. BERT Architecture:**
   - Consists of layers of encoders (12 for `bert-base`, 24 for `bert-large`).
   - Each encoder layer has:
     - **Multi-head Self Attention Sub-layer.**
     - **Feed Forward Neural Network Sub-layer.**

**4. Importance of Fine-Tuning:**
   - BERT is pre-trained on generic data. Fine-tuning on domain-specific data significantly improves model accuracy and application relevance.
   - **Steps for Fine-Tuning:**
     - Choose a domain-specific dataset.
     - Define a specific NLP task (e.g., text classification).
     - Train the model using transfer learning.

**5. Practical Application: Text Classification Task**
   - BERT can be fine-tuned for various tasks like sentiment analysis.
   - Code snippet for fine-tuning:
   ```python
   from transformers import BertForSequenceClassification, Trainer, TrainingArguments

   model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
   
   training_args = TrainingArguments(
       output_dir='./results',          
       num_train_epochs=3,              
       per_device_train_batch_size=8,  
       per_device_eval_batch_size=16,  
       warmup_steps=500,               
       weight_decay=0.01,              
       logging_dir='./logs',          
   )

   trainer = Trainer(
       model=model,                        
       args=training_args,                 
       train_dataset=train_dataset,        
       eval_dataset=eval_dataset            
   )

   trainer.train()
   ```

**6. Understanding Pre-training and Transfer Learning**
   - Distinction between pre-training and fine-tuning, where pre-training involves learning generic language features while fine-tuning adapts the model to specific tasks.

**7. Emerging BERT Variants:**
   - **DistilBERT, ALBERT, RoBERTa:** These are variants introduced to optimize BERT in terms of efficiency and/or performance in certain contexts.

**Resources:**
- Read Vaswani et al. “Attention is All You Need” paper for foundational concepts.
- Hugging Face’s Transformers documentation for practical code implementations.
- Explore BERT’s official GitHub repository and tutorials for a deeper technical dive.

These foundations and key principles will prepare you to understand BERT's intricate workings, fulfill your role effectively, and handle related interview questions competently.

Response for topic 'Day 2: Generalizing to New Tasks with BERT':
**Day 2: Generalizing to New Tasks with BERT**

### 8:00 AM - 9:00 AM: Overview of BERT and Transfer Learning
- **Key Concept**: BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model pre-trained on a large corpus, efficiently understanding context via bidirectional training of transformers.
- **Transfer Learning**: Using pre-trained BERT models as a starting point to apply to related tasks, significantly reducing computational costs and time. The essential idea is improving model performance and generalization by leveraging existing learned weights.

### 9:00 AM - 10:00 AM: Understanding Fine-Tuning in BERT
- **Fine-Tuning**: Process of customizing a pre-trained BERT model for a specific NLP task by adjusting specific model parameters with a smaller, task-specific dataset. Key in generalizing BERT for new tasks.
- **Steps**:
  1. Download a pre-trained BERT model.
  2. Prepare your task dataset (e.g., sentiment analysis).
  3. Train the model on this dataset, adjusting parameters slightly.

#### Code Snippet: Fine-Tuning with Hugging Face
```python
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # Assuming 3 classes

# Prepare data
train_texts, train_labels = ["example sentence"], [0]
train_encodings = tokenizer(train_texts, truncation=True, padding=True)

# Define Trainer
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir='./results',
        num_train_epochs=3
    ),
    train_dataset=train_encodings
)

# Fine-tune the model
trainer.train()
```

### 10:00 AM - 11:00 AM: Task-Specific Applications
- **Text Classification**: Classifying text into predefined categories. BERT improves the accuracy with its deep context understanding.
- **Entity Recognition**: Recognizing and categorizing key entities within a text which could be people, locations, etc.
- **Key Steps**: Tokenization, Encoding, Model training, and Evaluation.

### 11:00 AM - 12:00 PM: Implementing Real-World Applications
- Deploy BERT models to solve real-world tasks using frameworks like TensorFlow Serving, FastAPI, or Flask.
- Understand how BERT's contextual embeddings improve the tasks' performance.

#### Code Snippet: Inferencing with a Fine-Tuned Model
```python
from transformers import pipeline

# Load fine-tuned model
nlp_pipeline = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Run inference
result = nlp_pipeline("I love working with transformers!")
print(result)
```

### 1:00 PM - 2:00 PM: Optimization Techniques for BERT
- **Quantization**: Reducing the model size by lowering precision of model weights, which speeds up inference and decreases memory usage.
- **Pruning**: Removing parts of the model that are not critical to form results, optimizing the BERT model for low-end devices.

### 2:00 PM - 3:00 PM: Handling Large-Scale Datasets
- Learn efficient ways to preprocess data in batches to improve computational resource management.
  
### 3:00 PM - 4:00 PM: Experimentation and Performance Analysis
- Setting up experiments to test different model configurations or datasets.
- Initializing metrics for performance tracking (accuracy, precision, recall).

### 4:00 PM - 5:00 PM: Interview Preparation and Insights
- Review common interview questions:
  - Explain how BERT generalizes differently from traditional ML models.
  - Discuss challenges you might face when fine-tuning BERT for different tasks.
- Discuss examples of real projects or contributions made with BERT.

**Resources for In-Depth Learning:**
- Hugging Face Transformers documentation for BERT usage and tweaks.
- Research papers on BERT applications in various domains.
- Interactive courses on platforms like Coursera (e.g., DeepLearning.AI's NLP course) for practical understanding.

Response for topic 'Day 3: Fine-Tuning BERT for Specific Domains':
**Fine-Tuning BERT for Specific Domains**

**Concept Overview:**
Fine-tuning BERT for specific domains involves adapting a pre-trained BERT model to work effectively on a domain-specific task by training it further using relevant data. This process is essential when generic pre-trained models do not perform well on specific datasets due to vocabulary, context, or nuances peculiar to that domain.

**Key Components:**
1. **Understanding BERT Architecture:**
   - BERT uses transformer architecture, which includes self-attention mechanisms to capture the context from surrounding words in both left and right directions (bidirectional learning).
   - The attention mechanism allows BERT to understand contextual relationships. 

2. **Fine-Tuning Process:**
   - Start with a pre-trained BERT model.
   - Prepare clean, domain-specific data for training.
   - Modify the final layers of the BERT model to suit the task (e.g., adding a classification head for text classification tasks).
   - Implement the training loop: Forward propagation, loss calculation using a suitable loss function (e.g., cross-entropy loss for classification), backpropagation, optimizer step.
  
3. **Optimization Techniques:**
   - Use learning rate schedules like warm-up decay to stabilize training.
   - Employ dropout to prevent overfitting.
   - Gradient clipping to handle exploding gradients in models.

**Code Snippets:**

**Setting Up Environment:**
```python
!pip install transformers torch
```

**Loading BERT with Hugging Face Transformers and Fine-Tuning:**
```python
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader, Dataset
import torch

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)}

# Model Initialization
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Hyperparameters and Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Trainer Setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=CustomDataset(train_texts, train_labels),
    eval_dataset=CustomDataset(val_texts, val_labels)
)

# Training
trainer.train()
```

**Optimization:**
- **Learning Rate Scheduler:** Used within the Transformers library; typically suggested to use a linear schedule with warm-up.
- **Data Augmentation:** Enhance your dataset with techniques such as synonym replacement or back-translation to combat limited domain data.

**Metrics and Evaluation:**
- Post-training, evaluate the model using metrics such as accuracy, F1 score, precision, and recall specific to your task.
- Perform cross-validation if the dataset is small.

**Resources:**
- [Hugging Face Documentation](https://huggingface.co/transformers/)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
  
**Practice Questions for Interview:**
1. How does BERT's attention mechanism work, and why is it crucial for understanding context?
2. Describe the steps you’d take to prepare data for fine-tuning BERT on a domain-specific corpus.
3. What are some techniques you would employ to prevent overfitting when fine-tuning BERT?
4. Explain how you would use a learning rate schedule and why it may be beneficial during training.

Understanding these core ideas, steps, and code implementations will help to tackle domain-specific challenges effectively in BERT fine-tuning.

Response for topic 'Day 4: Advanced Techniques and Tools for BERT':
**Advanced Techniques and Tools for BERT**

1. **Understanding Variants of BERT:**
   - **BERT Variants**: RoBERTa, DistilBERT, and ALBERT are popular BERT variants. RoBERTa optimizes the training process by removing the Next Sentence Prediction task, while DistilBERT reduces the model size to increase speed and reduce resource requirements. ALBERT incorporates parameter sharing and factorized embedding parameterization for efficiency.
   - **Code Example: Loading BERT Variants**
     ```python
     from transformers import AutoModel, AutoTokenizer

     model_name = "roberta-base" # Can be replaced with 'distilbert-base-uncased', 'albert-base-v2', etc.
     model = AutoModel.from_pretrained(model_name)
     tokenizer = AutoTokenizer.from_pretrained(model_name)
     ```

2. **Fine-Tuning Pre-trained BERT Models:**
   - Fine-tuning involves adapting a pre-trained BERT model to a specific task or domain to improve performance. Essential steps include adjusting hyperparameters, using a custom dataset, and employing transfer learning.
   - **Code Example: Fine-tuning with PyTorch**
     ```python
     from transformers import Trainer, TrainingArguments

     training_args = TrainingArguments(
         output_dir='./results',
         num_train_epochs=3,
         per_device_train_batch_size=16,
         per_device_eval_batch_size=64,
         warmup_steps=500,
         weight_decay=0.01,
         logging_dir='./logs',
     )

     trainer = Trainer(
         model=model,
         args=training_args,
         train_dataset=train_dataset,
         eval_dataset=eval_dataset
     )
     trainer.train()
     ```

3. **Preprocessing Large-scale Text Datasets:**
   - Cleaning and preprocessing text data is crucial for building successful NLP models. This includes tokenization, removing stopwords, lowercasing, and stemming/lemmatization if necessary.
   - **Code Example: Tokenization using Hugging Face**
     ```python
     encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
     ```

4. **Deploying BERT Models:**
   - Deploy models using frameworks like TensorFlow, PyTorch, or Hugging Face's Transformers. Optimize for speed by considering quantization and pruning techniques to reduce the model size and improve latency.
   - **Code Example: Model Deployment with TorchScript**
     ```python
     scripted_model = torch.jit.script(model)
     torch.jit.save(scripted_model, "model_scripted.pt")
     ```

5. **Inference Pipelines:**
   - Real-time and batch processing require efficient pipelines for model inference. Batch processing can exploit parallelism to speed up predictions, while real-time applications might utilize caching and queue systems to handle incoming requests gracefully.
   - **Code Example: Batch Inference with DataLoader**
     ```python
     from torch.utils.data import DataLoader

     dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
     model.eval()
     with torch.no_grad():
         for batch in dataloader:
             inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)
             outputs = model(**inputs)
     ```

6. **Performance Optimization:**
   - Techniques like transfer learning (fine-tuning on specific tasks), hyperparameter tuning (e.g., learning rate, batch size), and using smaller model versions (DistilBERT, ALBERT) can significantly enhance model performance.
   - **Strategies**:
     - Use smaller batch sizes if memory is a constraint.
     - Employ learning rate schedulers, like linear decay or cosine annealing, for better convergence.

7. **Experimentation and Analysis:**
   - Conduct systematic experiments to evaluate model performance. Use metrics such as accuracy, F1-score, precision, and recall based on the task.
   - **Tools**:
     - TensorBoard for visualization.
     - sklearn.metrics for comprehensive evaluation.

8. **Collaboration and Continuous Learning:**
   - Stay updated with the latest advancements through literature, forums, and involvement in NLP communities. Collaboration with peers can improve problem-solving and bring in fresh perspectives.
   - **Resources**:
     - Hugging Face's blog and forums.
     - Research papers from conferences like ACL, EMNLP.

**Interview Preparation Focus:**
- Emphasize practical experience with BERT's fine-tuning, deployment nuances, and real-world problem-solving.
- Be prepared to discuss optimization techniques like quantization, pruning, and model scaling.
- Highlight any projects or contributions to open-source NLP tools or papers. 

Ensure readiness on these core areas to tackle questions and practical scenarios that may arise in the interview.

Response for topic 'Day 5: Practical Applications and Case Studies with BERT':
### Day 5: Practical Applications and Case Studies with BERT

**Objective:** Gain insights into how BERT can be applied to solve real-world NLP problems and prepare practical solutions for expected industry use cases such as text classification, entity recognition, sentiment analysis, and question answering.

#### Hourly Breakdown

**Hour 1-2: Overview of BERT Applications**
- **Focus:** Understand where BERT excels in NLP tasks.
- **Core Applications:**
  - **Text Classification:** Predicting categories for documents (e.g., spam detection).
  - **Named Entity Recognition (NER):** Identifying entities in text (e.g., people, organizations).
  - **Sentiment Analysis:** Determining sentiment expressed in text (e.g., positive, negative).
  - **Question Answering:** Extracting answers from passages for given questions.

**Hour 3-4: Fine-Tuning BERT Models**
- **Concept:** Fine-tuning involves adapting a pre-trained BERT model to a specific task by continuing the training process with task-specific data.
- **Key Steps:**
  - **Data Preparation:** Preprocess domain-specific datasets.
  - **Model Training:**
    ```python
    from transformers import BertForSequenceClassification, Trainer, TrainingArguments
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    training_args = TrainingArguments(
        output_dir='./results',          
        num_train_epochs=3,              
        per_device_train_batch_size=8,  
        per_device_eval_batch_size=8,   
        warmup_steps=500,                
        evaluation_strategy="epoch",
        logging_dir='./logs',            
    )
    trainer = Trainer(
        model=model,                         
        args=training_args,                  
        train_dataset=train_dataset,         
        eval_dataset=eval_dataset            
    )
    trainer.train()
    ```

**Hour 5-6: Case Studies Exploration**
- **Focus on Industry-Relevant Applications:**
  - **Financial Sector:** Text classification for processing financial documents.
  - **Healthcare:** NER for extracting medical terms from healthcare reports.
  - **E-commerce:** Sentiment analysis to gauge customer satisfaction from reviews.
- Discuss the impact and effectiveness of BERT in each case.

**Hour 7-8: Optimizing and Deploying BERT**
- **Model Optimization Techniques:**
  - **Quantization:** Reducing model size by decreasing precision of weights.
  - **Pruning:** Removing redundant weights.
- **Code Example for Quantization with PyTorch:**
  ```python
  from torch.quantization import quantize_dynamic
  model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
  quantized_model = quantize_dynamic(
      model, {torch.nn.Linear}, dtype=torch.qint8
  )
  ```

**Hour 9-10: Deploying BERT Models**
- **Deployment Options:** Deploy with Docker for consistency across environments.
- **Setup Docker Deployment:**
  ```dockerfile
  FROM pytorch/pytorch:latest
  COPY . /app
  WORKDIR /app
  RUN pip install -r requirements.txt
  CMD ["python", "app.py"]
  ```
- Discuss integrating models into scalable systems using Kubernetes and cloud services like AWS.

**Hour 11-12: Real-Time Inference Pipelines**
- **Building Scalable Pipelines:**
  - Use frameworks like TensorFlow Serving or Flask for real-time applications.
- **Example using Flask:**
  ```python
  from flask import Flask, request
  app = Flask(__name__)

  @app.route('/predict', methods=['POST'])
  def predict():
      data = request.json
      # Perform prediction with BERT model
      return {'prediction': model.predict(data)}

  if __name__ == '__main__':
      app.run()
  ```
  
#### Additional Learning Resources
- **Books:** "Deep Learning for Natural Language Processing" by Palash Goyal.
- **Courses:** Coursera's NLP Specialization by deeplearning.ai.
- **Research Papers:** Original BERT paper by Google AI Language (Devlin et al., 2019).

This breakdown ensures you grasp the critical 20% of knowledge to confidently discuss BERT applications and structures your day for optimal preparation and understanding of industry applications.

Response for topic 'Day 6: Reviewing and Refining BERT for Improved Accuracy':
### Day 6: Reviewing and Refining BERT for Improved Accuracy (8-hour Breakdown)

The focus is on exploring techniques to review and improve the accuracy of BERT models, which are crucial for a Machine Learning Engineer role in improving performance in NLP tasks.

#### Hour 1: Understanding BERT and Evaluation Metrics
- **Objective**: Grasp core concepts of BERT and key evaluation metrics.
- **Key Concepts**:
  - BERT's architecture: Bidirectional training of Transformers for language understanding.
  - Fine-tuning: Adjusting a pre-trained BERT to specific tasks like text classification.
  - Evaluation metrics: Precision, recall, F1-score for qualitative assessment.
- **Formula**: F1-score = 2 * (Precision * Recall) / (Precision + Recall)

#### Hour 2: Preparing Domain-Specific Datasets
- **Objective**: Learn the methods for preparing datasets tailored to specific applications.
- **Key Concepts**:
  - Data cleaning: Removal of noise and irrelevant parts.
  - Tokenization: Splitting text into understandable tokens for BERT.
  - Example Code (Using Hugging Face Transformers for tokenization in Python):
  ```python
  from transformers import BertTokenizer
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  tokens = tokenizer.tokenize("Sample text for tokenization.")
  ```

#### Hour 3: Model Fine-Tuning Techniques
- **Objective**: Master strategies to improve model accuracy through fine-tuning.
- **Key Concepts**:
  - Learning rate scheduling: Adjusting learning rates during training for stability.
  - Freeze certain layers: Training specific parts of the model, useful for smaller datasets.
- **Example Code**:
  ```python
  from transformers import BertForSequenceClassification
  model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
  for param in model.bert.parameters():
      param.requires_grad = False  # Freeze BERT layers
  optimizer = AdamW(model.parameters(), lr=2e-5)
  ```

#### Hour 4: Hyperparameter Optimization
- **Objective**: Implement and optimize hyperparameters.
- **Key Concepts**:
  - Identify the right batch size: Helps in controlling memory and affects convergence.
  - Experiment with learning rate: Critical for convergence speed.
- **Example Code**:
  ```python
  from transformers import Trainer, TrainingArguments
  training_args = TrainingArguments(
      per_device_train_batch_size=16,
      learning_rate=2e-5,
      num_train_epochs=3,
  )
  ```

#### Hour 5: Real-Time and Batch Inference Pipelines
- **Objective**: Develop scalable inference solutions for production environments.
- **Key Concepts**:
  - Inference optimization: Techniques to make the model inference efficient.
  - Frameworks: Utilizing TensorFlow, PyTorch, and deployment tools such as TorchScript.
- **Example Code**: Deployment (PyTorch)
  ```python
  import torch
  model.eval()
  input_ids = torch.tensor(tokenizer.encode("Sample input for inference")).unsqueeze(0)
  with torch.no_grad():
      outputs = model(input_ids)
  ```

#### Hour 6: Evaluating Model Performance
- **Objective**: Learn to evaluate and analyze model outputs for better accuracy.
- **Key Concepts**:
  - Importance of a validation set: Ensures model generalization.
  - Comparison metrics: Cross-validation and confusion matrix analysis.
- **Analytical Resources**: Use tools like `scikit-learn` for detailed evaluation.
  ```python
  from sklearn.metrics import classification_report
  print(classification_report(y_true, y_pred))
  ```

#### Hour 7: Iterative Refinement and Experimentation
- **Objective**: Understand iterative approaches to improve models over time.
- **Key Concepts**:
  - Experiment tracking: Using MLFlow or Weights & Biases for tracking.
  - Model iterations: Adjust based on errors and shortcomings.
- **Practice**: Consistently measure and record model performances between iterations.

#### Hour 8: Staying Updated with NLP Advancements
- **Objective**: Keep abreast of latest developments in BERT and NLP.
- **Key Concepts**:
  - Networking: Follow top researchers and papers from ACL, NeurIPS.
  - Community: Participate in forums like Hugging Face community and Reddit's ML channels.

Leverage these core principles to demonstrate proficiency in refining BERT models for optimal performance, aligning well with a machine learning engineer role focused on NLP.

"""

#split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600, 
    chunk_overlap=200,
    length_function=len,
    add_start_index= True
)

chunks = text_splitter.split_text(text)
print(f"Split text into {len(chunks)} chunks.") 


# Generate embeddings for the retrieved text and store them in the database
embedding = generate_embeddings(text)
store_embeddings(text, embedding)

# Retrieve the stored embeddings from the database
def retrieve_embeddings():
    all_entries = collection.find()
    return [(entry['text'], entry['embedding']) for entry in all_entries]


