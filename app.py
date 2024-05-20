
from model.model_training import train_model
from model.predict import load_model, predict_news

# Paths to the datasets
fake_path = "/content/drive/MyDrive/true_fake_news/Fake.csv"
real_path = "/content/drive/MyDrive/true_fake_news/True.csv"

# Train the model
train_model(fake_path, real_path)

# Load the trained model
model = load_model()

# Function to manually test news
def manual_testing(news):
    result = predict_news(model, news)
    print(f"Prediction: {result}")

# Example usage
if __name__ == "__main__":
    news_input = input("Enter Your news headline: ")
    manual_testing(news_input)
