import numpy as np
import matplotlib.pyplot as plt

# Load data
dataset_file = "/Users/somaiyaabdurahman/Documents/DVA493/LABS/LAB1/Diabetic.txt"
data_start_line = 24
data = np.genfromtxt(dataset_file, delimiter=',', skip_header=data_start_line)

# Separate features and labels
X = data[:, :-1]
y = data[:, -1].reshape(-1, 1)

# Normalize the features
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Split data into training, validation, and test sets
num_examples = X.shape[0]
num_train = int(0.75 * num_examples)
num_val = int(0.1 * num_examples)
num_test = num_examples - num_train - num_val

indices = np.random.permutation(num_examples)
X = X[indices]
y = y[indices]

X_train, y_train = X[:num_train], y[:num_train]
X_val, y_val = X[num_train:num_train + num_val], y[num_train:num_train + num_val]
X_test, y_test = X[num_train + num_val:], y[num_train + num_val:]

# Neural network parameters
input_size = 19  # 19 features
hidden_size_1 = 15
hidden_size_2 = 10
output_size = 1

# Initialize weights and biases
np.random.seed(42)
weights_input_hidden1 = np.random.randn(input_size, hidden_size_1)
weights_hidden1_hidden2 = np.random.randn(hidden_size_1, hidden_size_2)
weights_hidden2_output = np.random.randn(hidden_size_2, output_size)

bias_hidden1 = np.random.rand(hidden_size_1)
bias_hidden2 = np.random.rand(hidden_size_2)
bias_output = np.random.rand(output_size)

# Sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def forward_pass(X, weights_input_hidden1, bias_hidden1, weights_hidden1_hidden2, bias_hidden2, weights_hidden2_output, bias_output):
    hidden_layer_input1 = np.dot(X, weights_input_hidden1) + bias_hidden1
    hidden_layer_output1 = sigmoid(hidden_layer_input1)

    hidden_layer_input2 = np.dot(hidden_layer_output1, weights_hidden1_hidden2) + bias_hidden2
    hidden_layer_output2 = sigmoid(hidden_layer_input2)

    output_layer_input = np.dot(hidden_layer_output2, weights_hidden2_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    return predicted_output, hidden_layer_output1, hidden_layer_output2

learning_rate = 0.01
epochs = 200
val_accuracies = []

for epoch in range(epochs):
    # Forward pass for training data
    predicted_output, hidden_layer_output1, hidden_layer_output2 = forward_pass(X_train, weights_input_hidden1, bias_hidden1, weights_hidden1_hidden2, bias_hidden2, weights_hidden2_output, bias_output)

    # Compute training loss
    training_loss = np.mean(0.5 * (y_train - predicted_output) ** 2)

    # Backpropagation
    output_error = y_train - predicted_output
    output_delta = output_error * sigmoid_derivative(predicted_output)

    hidden_layer_error2 = output_delta.dot(weights_hidden2_output.T)
    hidden_layer_delta2 = hidden_layer_error2 * sigmoid_derivative(hidden_layer_output2)

    hidden_layer_error1 = hidden_layer_delta2.dot(weights_hidden1_hidden2.T)
    hidden_layer_delta1 = hidden_layer_error1 * sigmoid_derivative(hidden_layer_output1)

    # Update weights and biases
    weights_hidden2_output += learning_rate * hidden_layer_output2.T.dot(output_delta)
    bias_output += learning_rate * np.sum(output_delta, axis=0)

    weights_hidden1_hidden2 += learning_rate * hidden_layer_output1.T.dot(hidden_layer_delta2)
    bias_hidden2 += learning_rate * np.sum(hidden_layer_delta2, axis=0)

    weights_input_hidden1 += learning_rate * X_train.T.dot(hidden_layer_delta1)
    bias_hidden1 += learning_rate * np.sum(hidden_layer_delta1, axis=0)

    # Forward pass for validation data
    val_output, _, _ = forward_pass(X_val, weights_input_hidden1, bias_hidden1, weights_hidden1_hidden2, bias_hidden2, weights_hidden2_output, bias_output)

    # Compute validation accuracy
    val_accuracy = np.mean((val_output > 0.5) == y_val)
    val_accuracies.append(val_accuracy.item())


    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Training Loss: {training_loss}, Validation Accuracy: {val_accuracy * 100:.2f}%')


# Test the model
predicted_output_test, _, _ = forward_pass(X_test, weights_input_hidden1, bias_hidden1, weights_hidden1_hidden2, bias_hidden2, weights_hidden2_output, bias_output)

# Calculate test loss
test_loss = np.mean(0.5 * (y_test - predicted_output_test) ** 2)

# Calculate test accuracy
test_accuracy = np.mean((predicted_output_test > 0.5) == y_test)

print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy * 100:.2f}%')


plt.figure(figsize=(10, 5))
plt.plot(range(epochs), val_accuracies, label='Validation Accuracy', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy Changes Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

#do plot for training  