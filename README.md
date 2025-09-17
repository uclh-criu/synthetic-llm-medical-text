# synthetic-llm-medical-text

Simple tools for generating synthetic medical text using LLMs.

## Features
- Generate medical notes with GPT-4
- Match statistical properties of real text data
- Mark up entities and their relationships

## Usage

```python
from generate_note import generate_note
from enhance_prompt import enhance_prompt_with_stats, add_markup_instructions

# Basic generation
note = generate_note("Generate a clinical note about diabetes")

# With statistical matching
stats = analyze_data(real_data_df, text_col='note')
enhanced_prompt = enhance_prompt_with_stats(prompt, stats)

# With entity markup
prompts = add_markup_instructions(
    prompt="Generate a clinical note",
    entity_type=["medical event", "date"],
    relation_name="occurred on"
)
marked_note = generate_note(
    prompt=prompts['user_prompt'],
    system_prompt=prompts['system_prompt']
)
```

See `notebook.ipynb` for more examples.
