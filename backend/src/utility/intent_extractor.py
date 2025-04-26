from transformers import pipeline
import re
from typing import Dict, Any


class IntentExtractor:
    """
    A class to extract intent components from user queries using a pre-trained NER model.
    """

    def __init__(self):
        # Load the pre-trained NER model
        self.ner_model = pipeline(
            "ner",
            model="dslim/distilbert-NER",
            aggregation_strategy="simple",
        )

    def extract_intent_components(self, query: str) -> Dict[str, Any]:
        """
        Extract intent components from a user query.

        Args:
            query (str): The user query.

        Returns:
            Dict[str, Any]: Extracted intent components including primary intent, product type, desired attributes, and constraints.
        """
        # Define default intent components
        intent_components = {
            "primary_intent": None,
            "product_type": None,
            "desired_attributes": [],
            "constraints": [],
        }

        # Perform NER on the query
        ner_results = self.ner_model(query)

        # Process NER results to extract components
        for entity in ner_results:
            entity_text = entity["word"].strip()
            entity_label = entity["entity_group"]

            if entity_label == "MISC":
                # Assume MISC entities represent product types
                intent_components["product_type"] = entity_text
            elif entity_label == "ORG":
                # Assume ORG entities represent constraints (e.g., "in stock")
                intent_components["constraints"].append(entity_text)
            elif entity_label == "LOC":
                # Assume LOC entities represent desired attributes (e.g., "red")
                intent_components["desired_attributes"].append(entity_text)

        # Fallback: Use regex to extract price and color if NER fails
        if not intent_components["desired_attributes"]:
            color_match = re.search(
                r"\b(red|blue|green|black|white)\b", query, re.IGNORECASE
            )
            if color_match:
                intent_components["desired_attributes"].append(
                    color_match.group(0).lower()
                )

        if not intent_components["constraints"]:
            price_match = re.search(r"\bunder \$?\d+\b", query, re.IGNORECASE)
            if price_match:
                intent_components["constraints"].append(price_match.group(0).lower())

        # Infer primary intent based on keywords in the query
        if "buy" in query.lower():
            intent_components["primary_intent"] = "buy"
        elif "compare" in query.lower():
            intent_components["primary_intent"] = "compare"
        elif "find" in query.lower() or "similar" in query.lower():
            intent_components["primary_intent"] = "find similar"

        return intent_components


# Example usage
if __name__ == "__main__":
    extractor = IntentExtractor()
    query = "Find a red camera under $500 with free shipping"
    components = extractor.extract_intent_components(query)
    print(components)
