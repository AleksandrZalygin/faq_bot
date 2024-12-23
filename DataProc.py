import json
from typing import List, Tuple
from sklearn.model_selection import train_test_split


class DataProcessor:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.data = None
        self.questions = None
        self.answers = None

    def load_data(self) -> None:
        with open(self.json_path, 'r', encoding='utf-8') as file:
            faq_data = json.load(file)
            self.data = faq_data.get('data', [])

        # Извлечение вопросов и ответов
        self.questions = [entry.get('question', '') for entry in self.data]
        self.answers = [entry.get('answer', '') for entry in self.data]

    def prepare_datasets(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[
        List[str], List[str], List[str], List[str]]:
        if self.questions is None or self.answers is None:
            self.load_data()

        train_questions, test_questions, train_answers, test_answers = train_test_split(
            self.questions,
            self.answers,
            test_size=test_size,
            random_state=random_state
        )

        return train_questions, test_questions, train_answers, test_answers
