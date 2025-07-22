#!/usr/bin/env python3
"""
Helper utilities for preparing training data for context-aware expressions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import csv
import re
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class TrainingExample:
    """Represents a single training example"""
    text: str
    label: int  # 1 for positive, 0 for negative
    domain: str
    description: str = ""
    
    def to_dict(self):
        return {
            'text': self.text,
            'label': self.label,
            'domain': self.domain,
            'description': self.description
        }


class TrainingDataPreparer:
    """Utility class for preparing and managing training data"""
    
    def __init__(self):
        self.examples = []
    
    def add_example(
        self,
        text: str,
        label: int,
        domain: str,
        description: str = ""
    ):
        """Add a training example"""
        example = TrainingExample(text, label, domain, description)
        self.examples.append(example)
    
    def add_examples_from_dict(
        self,
        domain: str,
        positive_examples: List[str],
        negative_examples: List[str]
    ):
        """Add multiple examples from lists"""
        for text in positive_examples:
            self.add_example(text, 1, domain, "positive example")
        
        for text in negative_examples:
            self.add_example(text, 0, domain, "negative example")
    
    def load_from_csv(self, filepath: str):
        """Load training data from CSV file"""
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.add_example(
                    text=row['text'],
                    label=int(row['label']),
                    domain=row['domain'],
                    description=row.get('description', '')
                )
    
    def save_to_csv(self, filepath: str):
        """Save training data to CSV file"""
        with open(filepath, 'w', newline='') as f:
            fieldnames = ['text', 'label', 'domain', 'description']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for example in self.examples:
                writer.writerow(example.to_dict())
    
    def load_from_json(self, filepath: str):
        """Load training data from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            for item in data:
                self.add_example(
                    text=item['text'],
                    label=item['label'],
                    domain=item['domain'],
                    description=item.get('description', '')
                )
    
    def save_to_json(self, filepath: str):
        """Save training data to JSON file"""
        data = [example.to_dict() for example in self.examples]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_examples_by_domain(self, domain: str) -> Tuple[List[str], List[int]]:
        """Get examples for a specific domain"""
        domain_examples = [ex for ex in self.examples if ex.domain == domain]
        texts = [ex.text for ex in domain_examples]
        labels = [ex.label for ex in domain_examples]
        return texts, labels
    
    def get_positive_negative_by_domain(self, domain: str) -> Tuple[List[str], List[str]]:
        """Get positive and negative examples separately for a domain"""
        domain_examples = [ex for ex in self.examples if ex.domain == domain]
        positive = [ex.text for ex in domain_examples if ex.label == 1]
        negative = [ex.text for ex in domain_examples if ex.label == 0]
        return positive, negative
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the training data"""
        domains = set(ex.domain for ex in self.examples)
        stats = {
            'total_examples': len(self.examples),
            'domains': len(domains),
            'domain_breakdown': {}
        }
        
        for domain in domains:
            domain_examples = [ex for ex in self.examples if ex.domain == domain]
            positive = len([ex for ex in domain_examples if ex.label == 1])
            negative = len([ex for ex in domain_examples if ex.label == 0])
            
            stats['domain_breakdown'][domain] = {
                'total': len(domain_examples),
                'positive': positive,
                'negative': negative,
                'balance_ratio': positive / (positive + negative) if (positive + negative) > 0 else 0
            }
        
        return stats
    
    def validate_data(self) -> List[str]:
        """Validate training data and return issues found"""
        issues = []
        
        # Check for empty texts
        empty_texts = [i for i, ex in enumerate(self.examples) if not ex.text.strip()]
        if empty_texts:
            issues.append(f"Found {len(empty_texts)} examples with empty text")
        
        # Check label distribution by domain
        domains = set(ex.domain for ex in self.examples)
        for domain in domains:
            domain_examples = [ex for ex in self.examples if ex.domain == domain]
            positive = len([ex for ex in domain_examples if ex.label == 1])
            negative = len([ex for ex in domain_examples if ex.label == 0])
            
            if positive == 0:
                issues.append(f"Domain '{domain}' has no positive examples")
            if negative == 0:
                issues.append(f"Domain '{domain}' has no negative examples")
            
            # Check for severe imbalance
            total = positive + negative
            if total > 0:
                pos_ratio = positive / total
                if pos_ratio < 0.2 or pos_ratio > 0.8:
                    issues.append(f"Domain '{domain}' has imbalanced data (positive ratio: {pos_ratio:.2f})")
        
        # Check for very short examples
        short_examples = [ex for ex in self.examples if len(ex.text.split()) < 5]
        if short_examples:
            issues.append(f"Found {len(short_examples)} examples with fewer than 5 words")
        
        # Check for duplicate texts
        texts = [ex.text for ex in self.examples]
        unique_texts = set(texts)
        if len(texts) != len(unique_texts):
            issues.append(f"Found {len(texts) - len(unique_texts)} duplicate texts")
        
        return issues
    
    def augment_data(
        self,
        domain: str,
        augmentation_methods: List[str] = ['synonym_replacement', 'sentence_shuffle']
    ) -> int:
        """Augment training data using various techniques"""
        original_count = len(self.examples)
        domain_examples = [ex for ex in self.examples if ex.domain == domain]
        
        augmented = []
        
        for example in domain_examples:
            for method in augmentation_methods:
                if method == 'synonym_replacement':
                    augmented_text = self._synonym_replacement(example.text)
                elif method == 'sentence_shuffle':
                    augmented_text = self._sentence_shuffle(example.text)
                elif method == 'paraphrase':
                    augmented_text = self._simple_paraphrase(example.text)
                else:
                    continue
                
                if augmented_text and augmented_text != example.text:
                    augmented.append(TrainingExample(
                        text=augmented_text,
                        label=example.label,
                        domain=example.domain,
                        description=f"augmented ({method})"
                    ))
        
        self.examples.extend(augmented)
        return len(self.examples) - original_count
    
    def _synonym_replacement(self, text: str) -> str:
        """Simple synonym replacement (basic implementation)"""
        # This is a very basic implementation - in practice you'd use a proper thesaurus
        synonyms = {
            'patient': 'individual',
            'doctor': 'physician',
            'has': 'exhibits',
            'needs': 'requires',
            'transfer': 'move',
            'account': 'profile',
            'add': 'include',
            'cart': 'basket',
            'purchase': 'buy'
        }
        
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in synonyms:
                words[i] = synonyms[word.lower()]
        
        return ' '.join(words)
    
    def _sentence_shuffle(self, text: str) -> str:
        """Shuffle sentences in multi-sentence text"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) > 1:
            import random
            sentences_copy = sentences.copy()
            random.shuffle(sentences_copy)
            return '. '.join(sentences_copy) + '.'
        
        return text
    
    def _simple_paraphrase(self, text: str) -> str:
        """Simple paraphrasing (basic implementation)"""
        # Basic paraphrasing by changing sentence structure
        paraphrases = {
            r'patient has (.+)': r'the individual exhibits \1',
            r'transfer (.+) to (.+)': r'move \1 into \2',
            r'add (.+) to (.+)': r'include \1 in \2'
        }
        
        result = text
        for pattern, replacement in paraphrases.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result if result != text else text


def create_healthcare_training_data():
    """Create comprehensive healthcare training dataset"""
    preparer = TrainingDataPreparer()
    
    positive_examples = [
        "The doctor examined the patient who has severe headache and prescribed medication",
        "Medical records indicate the patient has diabetes and requires insulin therapy",
        "In the emergency room, the patient has chest pain and needs immediate attention",
        "The nurse noted that the patient has fever and administered antibiotics",
        "During the consultation, the patient has anxiety symptoms and was referred to therapy",
        "Clinical assessment shows the patient has hypertension and needs monitoring",
        "The specialist confirmed the patient has arthritis and recommended treatment",
        "Hospital admission notes: patient has pneumonia and requires oxygen support",
        "The surgeon explained that the patient has appendicitis and needs surgery",
        "After examination, the patient has migraine and was given pain relief",
        "The pediatrician noted the patient has asthma and prescribed inhaler",
        "Cardiology report states the patient has arrhythmia requiring medication",
        "The psychiatrist observed the patient has depression needing counseling",
        "Emergency physician confirmed the patient has fracture requiring casting",
        "The oncologist determined the patient has tumor needing chemotherapy"
    ]
    
    negative_examples = [
        "The computer has virus issues and needs antivirus software",
        "My car has engine problems and requires immediate repair",
        "The building has structural damage and needs renovation work",
        "The plant has yellow leaves and requires more water",
        "The software has bugs and needs debugging by developers",
        "The machine has mechanical failure and requires maintenance",
        "The project has budget issues and needs additional funding",
        "The website has performance problems and requires optimization",
        "The device has battery problems and needs replacement",
        "The system has security vulnerabilities and requires patches",
        "The application has loading issues and needs fixing",
        "The network has connectivity problems requiring troubleshooting",
        "The database has corruption issues needing restoration",
        "The server has overheating problems requiring cooling",
        "The router has configuration issues needing adjustment"
    ]
    
    preparer.add_examples_from_dict('healthcare', positive_examples, negative_examples)
    return preparer


def create_comprehensive_training_dataset():
    """Create a comprehensive multi-domain training dataset"""
    preparer = TrainingDataPreparer()
    
    # Healthcare domain
    healthcare_pos = [
        "The doctor examined the patient who has severe headache",
        "Medical records show the patient has diabetes requiring treatment",
        "Emergency physician noted the patient has cardiac symptoms",
        "The specialist confirmed the patient has arthritis",
        "Clinical assessment reveals the patient has hypertension"
    ]
    
    healthcare_neg = [
        "The computer has virus issues affecting performance",
        "The car has engine problems requiring repair",
        "The software has bugs needing fixes",
        "The building has structural damage",
        "The network has connectivity problems"
    ]
    
    # Financial domain
    financial_pos = [
        "Login to bank account and transfer funds to savings",
        "Complete wire transfer of money to checking account",
        "Use online banking to transfer payment to vendor",
        "PayPal allows you to transfer funds to merchant",
        "ATM transaction: transfer amount to credit account"
    ]
    
    financial_neg = [
        "Transfer files to backup folder location",
        "Transfer patient to ICU for treatment",
        "Transfer student to different class section",
        "Transfer ownership to new buyer",
        "Transfer call to support department"
    ]
    
    # E-commerce domain
    ecommerce_pos = [
        "Browse online store and add product to cart",
        "Shopping website: add items to basket now",
        "E-commerce platform allows adding products to cart",
        "Online marketplace: add books to order",
        "Digital store: add electronics to wishlist"
    ]
    
    ecommerce_neg = [
        "Recipe instructions: add sugar to mixture",
        "Database system: add user to records",
        "Meeting agenda: add topic to discussion",
        "Garden care: add fertilizer to soil",
        "Software: add feature to application"
    ]
    
    preparer.add_examples_from_dict('healthcare', healthcare_pos, healthcare_neg)
    preparer.add_examples_from_dict('financial', financial_pos, financial_neg)
    preparer.add_examples_from_dict('ecommerce', ecommerce_pos, ecommerce_neg)
    
    return preparer


def main():
    """Demonstrate training data preparation utilities"""
    
    print("Training Data Preparation Utilities")
    print("=" * 50)
    
    # Create comprehensive dataset
    print("\n1. Creating comprehensive training dataset...")
    preparer = create_comprehensive_training_dataset()
    
    # Show statistics
    stats = preparer.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"Total examples: {stats['total_examples']}")
    print(f"Domains: {stats['domains']}")
    
    for domain, info in stats['domain_breakdown'].items():
        print(f"\n{domain.title()}:")
        print(f"  Total: {info['total']}")
        print(f"  Positive: {info['positive']}")
        print(f"  Negative: {info['negative']}")
        print(f"  Balance: {info['balance_ratio']:.2f}")
    
    # Validate data
    print("\n2. Validating data...")
    issues = preparer.validate_data()
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ No issues found")
    
    # Augment data
    print("\n3. Augmenting healthcare data...")
    augmented_count = preparer.augment_data('healthcare', ['synonym_replacement', 'paraphrase'])
    print(f"Added {augmented_count} augmented examples")
    
    # Save to files
    print("\n4. Saving training data...")
    preparer.save_to_json('training_data.json')
    preparer.save_to_csv('training_data.csv')
    print("✓ Saved to training_data.json and training_data.csv")
    
    # Demonstrate loading
    print("\n5. Testing data loading...")
    new_preparer = TrainingDataPreparer()
    new_preparer.load_from_json('training_data.json')
    print(f"✓ Loaded {len(new_preparer.examples)} examples from JSON")
    
    # Get domain-specific data
    print("\n6. Domain-specific data extraction...")
    healthcare_pos, healthcare_neg = new_preparer.get_positive_negative_by_domain('healthcare')
    print(f"Healthcare: {len(healthcare_pos)} positive, {len(healthcare_neg)} negative")
    
    print("\nTraining data preparation complete!")
    print("Use the generated files with ContextAwareTrainer for model training.")


if __name__ == "__main__":
    main()