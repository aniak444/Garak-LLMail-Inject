from garak.probes.base import Probe
from datasets import load_dataset

class LLMailInject(Probe):
    """
    Probe for Agentic Indirect Prompt Injection based on the 
    microsoft/llmail-inject-challenge dataset (IEEE SaTML 2025).
    """
    
    b_type = "injection"
    name = "LLMailInject"
    description = "Tests if an agentic LLM executes unauthorized tool calls from untrusted emails."
    primary_detector = "agentic.ActionDetector"
    
    def __init__(self):
        super().__init__()
        self.prompts = []
        self._load_data()

    def _load_data(self):
        """
        Loads the Microsoft dataset and dynamically finds the correct column.
        Uses Phase1 as identified in the project requirements.
        """
        from datasets import load_dataset
        
        dataset = load_dataset("microsoft/llmail-inject-challenge")
        
        target_split = 'Phase1'
        
        columns = dataset[target_split].column_names
        print(f"DEBUG: Available columns in {target_split}: {columns}")

        if 'body' in columns:
            text_col = 'body'
        elif 'objectives' in columns:
            text_col = 'objectives'
        else:
            text_col = columns[0]

        print(f"DEBUG: Using '{text_col}' as the source for attack prompts.")

        for item in dataset[target_split]:
            self.prompts.append(str(item[text_col]))

    def get_prompts(self):
        return self.prompts