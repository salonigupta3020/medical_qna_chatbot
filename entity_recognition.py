import re

def extract_entities(question):
    """Basic entity recognition for symptoms, diseases, treatments."""
    # Simple example using regex
    symptoms = re.findall(r'\b(symptoms|signs|issues|problems)\b', question, re.IGNORECASE)
    diseases = re.findall(r'\b(fever|cold|flu|headache)\b', question, re.IGNORECASE)
    treatments = re.findall(r'\b(treatment|medication|therapy)\b', question, re.IGNORECASE)
    
    return {
        'symptoms': symptoms,
        'diseases': diseases,
        'treatments': treatments
    }

