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
        Loads the Microsoft LLMail-Inject dataset 
        and extracts attack prompts from Phase1.
        """
        dataset = load_dataset("microsoft/llmail-inject-challenge")
        target_split = "Phase1"

        columns = dataset[target_split].column_names
        print(f"[INFO] Available columns in '{target_split}': {columns}")

        if "body" in columns:
            text_col = "body"
        elif "objectives" in columns:
            text_col = "objectives"
        else:
            text_col = columns[0]

        print(f"[INFO] Using '{text_col}' as attack prompt source.")

        self.prompts = [str(item[text_col]) for item in dataset[target_split]]
        print(f"[INFO] Loaded {len(self.prompts)} attack prompts.")

    def generate(self):
        return self.prompts