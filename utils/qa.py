from transformers import AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

def create_qa_pipeline(model_name="Intel/dynamic_tinybert"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=512)
    question_answerer = pipeline("question-answering", model=model_name, tokenizer=tokenizer, return_tensors='pt')
    llm = HuggingFacePipeline(pipeline=question_answerer, model_kwargs={"temperature": 0.7, "max_length": 512})
    return llm
