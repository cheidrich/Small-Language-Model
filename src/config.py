
data_path = "../data/deu_news_2024_10K-sentences.txt"
model_save_path = "../models/language_model.pth"

vocab_size = 8000

epochs = 30
learning_rate = 0.005
word_loss_weight = 0.4
class_loss_weight = 0.6

embedding_dim = 25
hidden_dim = 100

max_generate_len = 20