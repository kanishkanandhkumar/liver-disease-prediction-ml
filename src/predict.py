
# predict.py - Make predictions with trained model

import torch
import torch.nn as nn
import numpy as np

class ToxicityModel(nn.Module):
    def __init__(self, input_dim=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return torch.sigmoid(self.model(x)).squeeze()

def load_model():
    """Load trained model"""
    model = ToxicityModel(input_dim=32)
    model.load_state_dict(torch.load('../models/bbbp_toxicity_model.pth', 
                                     map_location='cpu'))
    model.eval()
    return model

def smiles_to_features(smiles):
    """Same function as in training"""
    chars = 'C N O H S P F Cl Br I ( ) [ ] = # - + . 1 2 3 4 5 6 7 8 9 0'.split()
    features = np.zeros(len(chars) + 3)
    
    for i, char in enumerate(chars):
        features[i] = smiles.count(char)
    
    features[-3] = len(smiles)
    features[-2] = smiles.count('=') + smiles.count('#')
    features[-1] = smiles.count('C') / max(1, len(smiles))
    
    return features.astype(np.float32)

def predict(smiles, model=None):
    """Predict toxicity for a SMILES string"""
    if model is None:
        model = load_model()
    
    features = smiles_to_features(smiles)
    features_tensor = torch.FloatTensor(features).unsqueeze(0)
    
    with torch.no_grad():
        prob = model(features_tensor).item()
    
    result = {
        'smiles': smiles,
        'probability': prob,
        'prediction': 'Toxic' if prob > 0.5 else 'Safe',
        'confidence': abs(prob - 0.5) * 2
    }
    
    return result

# Example usage
if __name__ == "__main__":
    # Test on aspirin
    result = predict("CC(=O)OC1=CC=CC=C1C(=O)O")
    print(f"Aspirin: {result['prediction']} (prob: {result['probability']:.3f})")
