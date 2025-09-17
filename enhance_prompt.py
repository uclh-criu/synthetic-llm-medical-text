from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
from collections import Counter

def get_basic_text_stats(df: pd.DataFrame, text_col: str = 'note') -> Dict[str, float]:
    """
    Compute basic statistics about text length and structure.
    
    Args:
        df: DataFrame containing the text data
        text_col: Column name containing the text
    
    Returns:
        Dict with basic text statistics
    """
    # Length statistics
    lengths = df[text_col].str.len()
    
    # Sentence statistics (approximate using periods)
    sentences_per_doc = df[text_col].str.count('\.')
    
    return {
        'length_stats': {
            'mean': lengths.mean(),
            'std': lengths.std(),
            'min': lengths.min(),
            'max': lengths.max(),
            'median': lengths.median()
        },
        'sentence_stats': {
            'mean': sentences_per_doc.mean(),
            'std': sentences_per_doc.std(),
            'min': sentences_per_doc.min(),
            'max': sentences_per_doc.max(),
            'median': sentences_per_doc.median()
        }
    }

def analyze_vocabulary(df: pd.DataFrame, text_col: str = 'note') -> Dict[str, Any]:
    """
    Analyze vocabulary and word usage patterns.
    
    Args:
        df: DataFrame containing the text data
        text_col: Column name containing the text
    
    Returns:
        Dict with vocabulary statistics
    """
    # Combine all text
    all_text = ' '.join(df[text_col].fillna(''))
    
    # Basic word statistics
    words = all_text.split()
    word_counts = Counter(words)
    
    return {
        'vocabulary_size': len(word_counts),
        'avg_word_length': np.mean([len(w) for w in words]),
        'common_words': dict(word_counts.most_common(50))
    }

def analyze_data(
    df: pd.DataFrame,
    text_col: str = 'note',
    analysis_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run configurable analysis on text data.
    
    Args:
        df: DataFrame containing the text data
        text_col: Column name containing the text
        analysis_types: List of analysis types to perform. 
                       If None, performs all available analyses.
                       Options: ['basic', 'vocabulary']
    
    Returns:
        Dictionary of requested analyses
    """
    if analysis_types is None:
        analysis_types = ['basic', 'vocabulary']
        
    results = {}
    
    if 'basic' in analysis_types:
        results['basic'] = get_basic_text_stats(df, text_col)
        
    if 'vocabulary' in analysis_types:
        results['vocabulary'] = analyze_vocabulary(df, text_col)
        
    return results

def create_stats_guidance(stats: Dict[str, Any]) -> str:
    """
    Create a human-readable guidance string from statistics.
    
    Args:
        stats: Dictionary of statistics from analyze_data()
    
    Returns:
        Formatted string with statistical guidance
    """
    guidance = "\nStatistical properties to match:\n"
    
    if 'basic' in stats:
        basic_stats = stats['basic']
        length_stats = basic_stats['length_stats']
        guidance += f"- Target length: {length_stats['mean']:.0f} characters "
        guidance += f"(range: {length_stats['min']:.0f}-{length_stats['max']:.0f})\n"
        
        sentence_stats = basic_stats['sentence_stats']
        guidance += f"- Target sentences: {sentence_stats['mean']:.1f} "
        guidance += f"(range: {sentence_stats['min']:.0f}-{sentence_stats['max']:.0f})\n"
    
    if 'vocabulary' in stats:
        vocab_stats = stats['vocabulary']
        guidance += f"- Vocabulary size: {vocab_stats['vocabulary_size']} unique words\n"
        guidance += f"- Average word length: {vocab_stats['avg_word_length']:.1f} characters\n"
        
        # Add some common words as examples
        common_words = list(vocab_stats['common_words'].keys())[:5]
        guidance += f"- Common words: {', '.join(common_words)}\n"
    
    return guidance

def enhance_prompt_with_stats(
    prompt: str,
    stats: Dict[str, Any]
) -> str:
    """
    Add statistical guidance to prompt.
    
    Args:
        prompt: Original prompt text
        stats: Dictionary of statistics from analyze_data()
    
    Returns:
        Enhanced prompt with statistical guidance
    """
    stats_guidance = create_stats_guidance(stats)
    return f"{prompt}\n\n{stats_guidance}"

def add_markup_instructions(
    prompt: str,
    entity_type: Union[str, List[str]],  # Can be single type or list of types
    relation_name: Optional[str] = None,
) -> Dict[str, str]:
    """Add entity markup and optional relationship instructions to a prompt."""
    
    # Handle single or multiple entity types
    if isinstance(entity_type, str):
        entity_types = [entity_type]
    else:
        entity_types = entity_type

    # Build system prompt with entity markup instructions
    if len(entity_types) == 1:
        # Single entity type (original behavior)
        system_prompt = (
            f"You are a clinical note generator.\n"
            f"Mark each {entity_types[0]} mentioned in the text with [E] tags.\n"
            f"Example: The patient takes [E]aspirin[/E]."
        )
    else:
        # Multiple entity types (e.g., medical events and dates)
        type_instructions = []
        for i, etype in enumerate(entity_types):
            tag = f"[{chr(65+i)}]"  # [A], [B], etc.
            type_instructions.append(f"Mark each {etype} with {tag} tags")
        
        system_prompt = (
            f"You are a clinical note generator.\n"
            f"{'. '.join(type_instructions)}.\n"
            f"Example: {type_instructions[0]}: The patient has [{chr(65)}]diabetes[/{chr(65)}]\n"
            f"         {type_instructions[1]}: Diagnosed on [{chr(66)}]January 2020[/{chr(66)}]"
        )

    # Add relationship instructions if requested
    if relation_name:
        relationship_instructions = (
            f"\n\nAfter the note, list any {relation_name} "
            f"relationships between marked entities:\n"
            f"[RELATIONS]\n"
            f"{entity_types[0]}, {entity_types[1] if len(entity_types) > 1 else entity_types[0]}\n"
            f"[/RELATIONS]"
        )
        system_prompt += relationship_instructions

    return {
        'system_prompt': system_prompt,
        'user_prompt': prompt
    }