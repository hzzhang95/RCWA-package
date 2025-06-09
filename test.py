import torch


# Step 1: Define the parameter to optimize
x = torch.tensor([0.0], requires_grad=True)  # starting point

# Step 2: Set up the optimizer
optimizer = torch.optim.Adam([x], lr=0.25)

# Step 3: Optimization loop
for step in range(100):
    optimizer.zero_grad()  # zero the gradients

    # Step 4: Define the function
    loss = (x - 3) ** 2 + 2  # this is the function to minimize

    loss.backward()  # compute gradients
    optimizer.step()  # update x

    if step % 10 == 0:
        print(f"Step {step}: x = {x.item():.4f}, loss = {loss.item():.4f}")

print(f"Minimum at x = {x.item():.4f}")
