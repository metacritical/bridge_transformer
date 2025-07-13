import os
import json
import random
import torch
import numpy as np
import tiktoken
import requests
from torch.utils.data import Dataset, DataLoader

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# -----------------------------------------------------------------------------
# Data Generation Options
# -----------------------------------------------------------------------------

class LLMDataGenerator:
    """Generate dataset for bridge functionality using a local LLM"""
    
    def __init__(self, lm_studio_endpoint="http://localhost:1234/v1"):
        self.endpoint = lm_studio_endpoint
        self.headers = {
            "Content-Type": "application/json"
        }
        
        # Categories of knowledge-requiring questions
        self.question_categories = [
            "history", "science", "geography", "literature", 
            "politics", "art", "sports", "technology"
        ]
        
        # Conversation categories that don't require knowledge
        self.conversation_categories = [
            "greetings", "personal", "creative", "hypothetical", 
            "advice", "opinions", "preferences"
        ]
    
    def generate_dataset(self, size=1000, include_non_bridge=True):
        """Generate a dataset using a local LLM"""
        dataset = []
        
        # Determine split between bridge and non-bridge examples
        bridge_count = size if not include_non_bridge else int(size * 0.7)
        non_bridge_count = 0 if not include_non_bridge else size - bridge_count
        
        print(f"Generating {bridge_count} bridge examples...")
        bridge_examples = self._generate_bridge_examples(bridge_count)
        dataset.extend(bridge_examples)
        
        if include_non_bridge:
            print(f"Generating {non_bridge_count} non-bridge examples...")
            non_bridge_examples = self._generate_non_bridge_examples(non_bridge_count)
            dataset.extend(non_bridge_examples)
        
        # Shuffle dataset
        random.shuffle(dataset)
        return dataset
    
    def _generate_bridge_examples(self, count):
        """Generate examples that require external knowledge"""
        examples = []
        
        # Create prompts for different categories
        for _ in range(count):
            category = random.choice(self.question_categories)
            
            # Generate question using LLM
            question_prompt = f"Generate a specific factual question about {category} that would require looking up information to answer accurately. The question should be straightforward and have a clear factual answer."
            question = self._generate_from_llm(question_prompt)
            
            # Extract entity and domain from question using LLM
            extraction_prompt = f"""
            For this question: "{question}"
            
            1. What is the main entity being asked about?
            2. What category of knowledge is needed (person, place, date, event, etc.)?
            
            Answer in this format:
            Entity: [entity]
            Category: [category]
            """
            
            extraction_result = self._generate_from_llm(extraction_prompt)
            
            # Parse extraction result
            entity = ""
            category = ""
            
            for line in extraction_result.split('\n'):
                if line.startswith("Entity:"):
                    entity = line.replace("Entity:", "").strip()
                elif line.startswith("Category:"):
                    category = line.replace("Category:", "").strip()
            
            # Create query representation
            query_repr = f"{category}:{entity}" if entity and category else f"query:{question}"
            
            # Generate answer using LLM
            answer_prompt = f"Answer this factual question accurately and concisely: {question}"
            answer = self._generate_from_llm(answer_prompt)
            
            example = {
                "input": question,
                "knowledge_boundary": True,
                "query_representation": query_repr,
                "retrieved_information": answer,
                "expected_output": answer
            }
            
            examples.append(example)
            print(f"Generated bridge example: {question}")
        
        return examples
    
    def _generate_non_bridge_examples(self, count):
        """Generate examples that don't require external knowledge"""
        examples = []
        
        for _ in range(count):
            category = random.choice(self.conversation_categories)
            
            # Generate conversation starter
            if category == "greetings":
                prompt = "Generate a casual greeting or conversation starter."
            elif category == "personal":
                prompt = "Generate a personal question that doesn't require factual knowledge to answer."
            elif category == "creative":
                prompt = "Generate a request for creative content like a story, poem, or description."
            elif category == "hypothetical":
                prompt = "Generate a hypothetical 'what if' question."
            elif category == "advice":
                prompt = "Generate a request for general advice or guidance."
            elif category == "opinions":
                prompt = "Generate a question asking for an opinion on a topic."
            else:  # preferences
                prompt = "Generate a question about preferences or favorites."
            
            question = self._generate_from_llm(prompt)
            
            example = {
                "input": question,
                "knowledge_boundary": False,
                "query_representation": "",
                "retrieved_information": "",
                "expected_output": "[RESPONSE THAT DOESN'T REQUIRE EXTERNAL KNOWLEDGE]"
            }
            
            examples.append(example)
            print(f"Generated non-bridge example: {question}")
        
        return examples
    
    def _generate_from_llm(self, prompt):
        """Generate text from local LLM using LM Studio API"""
        try:
            data = {
                "model": "local-model",  # The model ID in LM Studio
                "prompt": prompt,
                "max_tokens": 150,
                "temperature": 0.7
            }
            
            response = requests.post(
                f"{self.endpoint}/completions",
                headers=self.headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["text"].strip()
            else:
                print(f"Error: {response.status_code}")
                print(f"Response: {response.text}")
                return f"[Error generating text: {response.status_code}]"
        except Exception as e:
            print(f"Exception when calling LLM: {e}")
            return f"[Error: {str(e)}]"
    
    def save_dataset(self, filepath):
        """Save dataset to disk"""
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Dataset saved to {filepath}")
    
    def load_dataset(self, filepath):
        """Load dataset from disk"""
        with open(filepath, 'r') as f:
            dataset = json.load(f)
        return dataset


class BridgeDataset(Dataset):
    """PyTorch dataset for bridge model training"""
    
    def __init__(self, data, tokenizer, block_size=1024):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = data
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize input
        input_ids = self.tokenizer.encode(example["input"])
        
        # For training the bridge detection:
        # 1 if this example requires bridge, 0 otherwise
        bridge_label = 1 if example["knowledge_boundary"] else 0
        
        # Tokenize expected output
        output_ids = self.tokenizer.encode(example["expected_output"])
        
        # Tokenize retrieved information (if any)
        if example["retrieved_information"]:
            retrieval_ids = self.tokenizer.encode(example["retrieved_information"])
        else:
            retrieval_ids = []
        
        # Create query representation embedding
        query_repr = example["query_representation"]
        
        # Make sure sequences don't exceed block size
        if len(input_ids) > self.block_size:
            input_ids = input_ids[:self.block_size]
        
        # Create attention mask (1 for tokens to attend to, 0 for padding)
        attn_mask = [1] * len(input_ids)
        
        # Convert to tensors
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        bridge_label_tensor = torch.tensor(bridge_label, dtype=torch.float)
        output_tensor = torch.tensor(output_ids, dtype=torch.long)
        attn_mask_tensor = torch.tensor(attn_mask, dtype=torch.long)
        
        return {
            "input_ids": input_tensor,
            "attention_mask": attn_mask_tensor,
            "bridge_label": bridge_label_tensor,
            "output_ids": output_tensor,
            "query_repr": query_repr,
            "retrieval_ids": retrieval_ids if retrieval_ids else None
        }


def create_dataloaders(dataset, batch_size=16, split=[0.8, 0.1, 0.1]):
    """Split dataset and create dataloaders"""
    assert sum(split) == 1.0, "Split ratios must sum to 1"
    
    # Determine split sizes
    train_size = int(len(dataset) * split[0])
    val_size = int(len(dataset) * split[1])
    test_size = len(dataset) - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Bridge Dataset')
    parser.add_argument('--generator', type=str, default='llm',
                        help='Type of generator to use')
    parser.add_argument('--size', type=int, default=1000,
                        help='Number of examples to generate')
    parser.add_argument('--output', type=str, default='bridge_dataset.json',
                        help='Output file path')
    parser.add_argument('--lm_endpoint', type=str, default='http://localhost:1234/v1',
                        help='LM Studio API endpoint (for LLM generator)')
    
    args = parser.parse_args()
    
    # Use LLM data generator
    print(f"Using LLM data generator with endpoint {args.lm_endpoint} to create {args.size} examples")
    generator = LLMDataGenerator(lm_studio_endpoint=args.lm_endpoint)
    
    # Generate dataset
    dataset = generator.generate_dataset(size=args.size)
    
    # Save dataset
    generator.save_dataset(dataset, args.output)
    
    print(f"Dataset with {len(dataset)} examples saved to {args.output}")
