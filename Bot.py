import re
from typing import List, Tuple

import nltk
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

# Проверяем наличие необходимых данных NLTK и загружаем их при необходимости
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class FAQBot:
    def __init__(self, model_name: str = 'DeepPavlov/rubert-base-cased'):
        print(f"Инициализация модели: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Переключение модели в режим оценки на CPU
        self.model.eval()
        self.model.cpu()

        # Переменные для базы знаний
        self.questions_embeddings = None
        self.answers = None
        self.original_questions = None

    def preprocess_text(self, text: str) -> str:
        text = text.lower()  # Приведение к нижнему регистру
        text = re.sub(r'[^\w\s]', ' ', text)  # Удаление знаков препинания
        text = re.sub(r'\s+', ' ', text).strip()  # Удаление лишних пробелов
        return text

    def get_embedding(self, text: str) -> np.ndarray:
        processed_text = self.preprocess_text(text)
        tokens = self.tokenizer(
            processed_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        # Извлечение эмбеддинга с модели
        with torch.no_grad():
            output = self.model(**tokens)
            embedding = output.last_hidden_state.mean(dim=1).squeeze().numpy()

        return embedding

    def calculate_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        return cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))[0][0]

    def build_knowledge_base(self, questions: List[str], answers: List[str]):
        print("Создание базы знаний...")
        self.original_questions = questions
        embeddings = []

        for index, question in enumerate(questions):
            if index % 10 == 0:
                print(f"Обработка вопроса {index}/{len(questions)}")

            embeddings.append(self.get_embedding(question))

        self.questions_embeddings = np.stack(embeddings)
        self.answers = answers
        print("База знаний успешно создана!")

    def find_answer(self, question: str, top_k: int = 1, threshold: float = 0.5) -> List[Tuple[str, float]]:
        if self.questions_embeddings is None or self.answers is None:
            raise ValueError("База знаний не инициализирована.")

        question_embedding = self.get_embedding(question)
        similarities = np.array([
            self.calculate_similarity(question_embedding, emb)
            for emb in self.questions_embeddings
        ])

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [
            (self.answers[idx], float(similarities[idx]))
            for idx in top_indices if similarities[idx] >= threshold
        ]

        if not results:
            return [("Извините, я не смог найти подходящий ответ. Попробуйте задать вопрос иначе.", 0.0)]

        return results

    def save_model(self, path: str):
        torch.save({
            'questions_embeddings': self.questions_embeddings,
            'answers': self.answers,
            'original_questions': self.original_questions
        }, path)
        print(f"Данные успешно сохранены в файл: {path}")

    def load_model(self, path: str):
        data = torch.load(path, map_location='cpu')
        self.questions_embeddings = data['questions_embeddings']
        self.answers = data['answers']
        self.original_questions = data['original_questions']
        print(f"Данные успешно загружены из файла: {path}")
