import torch

def main():
    # Check GPU availability
    print(torch.cuda.is_available())

    # Use GPU for computations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    print(x ** 2)

main()
