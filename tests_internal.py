import torch
import sys
import os

# Memastikan script bisa mengimport dari folder src
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

try:
    from models import OptoGNN
    from utils import RankMSELoss
except ImportError:
    # Fallback jika dijalankan dari root project
    sys.path.insert(0, os.path.join(os.getcwd(), 'opto_gnn_project/src'))
    from models import OptoGNN
    from utils import RankMSELoss

def test_model_forward():
    print("Testing OptoGNN Forward Pass...")
    model = OptoGNN(esm_dim=320, hidden_dim=64)
    
    # Mock data
    num_nodes = 50
    x = torch.randn((num_nodes, 320))
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    class MockData:
        pass
    data = MockData()
    data.x = x
    data.edge_index = edge_index
    data.batch = batch
    
    output = model(data)
    assert output.shape == (1, 1), f"Expected shape (1,1), got {output.shape}"
    print("✅ Forward Pass Success!")

def test_loss_backward():
    print("Testing RankMSELoss Backward Pass...")
    criterion = RankMSELoss(alpha=0.5)
    preds = torch.randn((5, 1), requires_grad=True)
    targets = torch.randn((5, 1))
    
    loss = criterion(preds, targets)
    loss.backward()
    
    assert preds.grad is not None, "Gradients should not be None"
    print("✅ Loss Gradient Success!")

if __name__ == '__main__':
    try:
        test_model_forward()
        test_loss_backward()
        print("\n--- SEMUA TES BERHASIL ---")
    except Exception as e:
        print(f"\n❌ Tes Gagal: {e}")
        sys.exit(1)
