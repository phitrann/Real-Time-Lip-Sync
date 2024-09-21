import torch
from models import Wav2Lip

class LipSyncModel:
    def __init__(self, checkpoint_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = Wav2Lip()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.model.to(self.device)

    def predict(self, mel, face):
        mel = torch.FloatTensor(mel.transpose(0, 2, 1)).unsqueeze(0).to(self.device)
        face = torch.FloatTensor(face.transpose(0, 3, 1, 2)).to(self.device)

        with torch.no_grad():
            pred = self.model(mel, face)

        return pred.cpu().numpy().transpose(0, 2, 3, 1)

# Usage
model = LipSyncModel('path/to/wav2lip_checkpoint.pth')