import spacy
import numpy as np
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize

# Load NLP model (disable NER as it's not needed)
nlp = spacy.load("en_core_web_sm", disable=["ner"])

# Load sentence transformer model (runs on CPU)
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2", device="cpu")

# Company details used as knowledge base
company_info = """
AI Solutions Inc. is a leading technology company specializing in artificial intelligence, 
machine learning, and data analytics. Founded in 2015, our company is headquartered in San Francisco, USA. 
We help businesses integrate AI into their workflows to optimize efficiency and decision-making.

Our core services include AI consulting, custom machine learning model development, natural language 
processing (NLP) solutions, and data-driven insights. We work with industries such as finance, healthcare, 
retail, and cybersecurity to provide intelligent automation and predictive analytics.

We have a global team of over 500 AI researchers, engineers, and data scientists, working remotely 
and in our offices worldwide. Our clients range from startups to Fortune 500 companies.

At AI Solutions Inc., we prioritize ethical AI practices and transparency. We ensure that our AI models 
are fair, unbiased, and explainable. Our research team actively contributes to the AI community by 
publishing papers, open-source tools, and participating in AI conferences.

Customer support is available 24/7 via email at support@aisolutions.com and through our online chat system. 
For enterprise clients, we provide dedicated account managers for personalized assistance.

Our mission is to make artificial intelligence accessible and beneficial to businesses of all sizes. 
We believe AI should empower human creativity rather than replace it. Our vision is to build the future 
of AI-driven automation, making processes smarter and more efficient.

AI Solutions Inc. has received multiple awards for innovation in AI and has been recognized as one of the 
top AI startups by industry leaders. We partner with major tech companies like Google, Microsoft, and 
IBM to push the boundaries of AI development.

Our company offers internship programs, AI training bootcamps, and online certification courses to 
help individuals upskill in AI and machine learning. We also provide AI research grants for university students.

The company is expanding its operations to Europe and Asia, opening new offices in London and Singapore 
by the end of 2025.

For more details, visit our website at www.aisolutions.com.
"""

# Split company information into sentences
sentences = sent_tokenize(company_info)

# Generate embeddings for each sentence
sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

# Keyword-based categories for relevance boosting
question_keywords = {
    "company services": {"ai", "services", "machine learning", "automation", "nlp", "analytics"},
    "company location": {"headquarters", "san francisco", "office", "location", "expanding", "global"},
    "contact support": {"customer support", "email", "contact", "help"},
    "partnerships": {"google", "microsoft", "ibm", "partner"},
    "internships and training": {"internship", "training", "bootcamp", "certification", "course", "grant"},
    "awards and recognition": {"award", "innovation", "recognized", "top ai startup"},
    "ethical ai": {"ethical", "transparency", "fair", "bias", "explainable", "ai community"},
}

# Chatbot loop
print("\nAI Chatbot (Type 'exit' to quit)\n")
while True:
    user_input = input("You: ").strip().lower()
    if user_input == "exit":
        print("Chatbot: Goodbye!")
        break

    # Encode user input
    user_embedding = model.encode(user_input, convert_to_tensor=True)

    # Compute similarity scores
    similarity_scores = util.pytorch_cos_sim(user_embedding, sentence_embeddings)[0].cpu().numpy()

    # Find the best and second-best matches
    best_idx = np.argmax(similarity_scores)
    second_best_idx = np.argsort(similarity_scores)[-2]

    best_sentence = sentences[best_idx]
    second_best_sentence = sentences[second_best_idx]

    best_score = similarity_scores[best_idx]
    second_best_score = similarity_scores[second_best_idx]

    # Extract keywords from user input
    extracted_keywords = {token.text.lower() for token in nlp(user_input) if token.is_alpha}

    # Boost confidence score if keywords match predefined categories
    if any(extracted_keywords & keywords for keywords in question_keywords.values()):
        best_score += 0.2

    # Respond if confidence is high enough
    if best_score >= 0.4:
        response = f"Chatbot: {best_sentence}"
        if second_best_score >= 0.35 and second_best_sentence != best_sentence:
            response += f"\n(Alternative: {second_best_sentence})"
    else:
        response = "Chatbot: Sorry, I couldn't understand."

    print(response)
