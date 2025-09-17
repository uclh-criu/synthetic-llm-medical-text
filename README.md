# synthetic-llm-medical-text

Simple tools for generating synthetic medical text using OpenAI's LLMs.

## Features
- Generate medical notes using any OpenAI model
- Match statistical properties of real text data
- Mark up entities and their relationships

## Setup

1. Create a `.env` file in the project root:
```
OPENAI_API_KEY=your-api-key-here
```

2. Install dependencies:
```
pip install openai python-dotenv pandas numpy
```

## Usage

```python
import pandas as pd
from generate_note import generate_note, generate_batch
from enhance_prompt import analyze_data, enhance_prompt_with_stats, add_markup_instructions

# Define base prompt and model
prompt = "Generate a clinical note about a patient with diabetes. Include their symptoms, medications, and treatment plan."
model = "gpt-3.5-turbo"  # or any other OpenAI model

# Basic generation
note = generate_note(prompt, model=model)

# Batch generation (multiple notes in one call)
notes = generate_batch(prompt, n_samples=3, model=model)

# Enhance prompt with statistics from real data
df = pd.read_csv("your_data.csv")
stats = analyze_data(df, text_col='note_text')
stats_prompt = enhance_prompt_with_stats(prompt, stats)

# Enhance stats_prompt with entity markup
markup_prompt = add_markup_instructions(
    prompt=stats_prompt,
    entity_type=["medical event", "date"],
    relation_name="occurred on"
)

# Generate note with both statistical and markup enhancements
enhanced_note = generate_note(
    prompt=markup_prompt['user_prompt'],
    system_prompt=markup_prompt['system_prompt'],
    model=model
)

print(enhanced_note)
```

See `examples.ipynb` for more detailed examples.