from Bot import FAQBot
from DataProc import DataProcessor


def train():
    try:
        # Загрузка данных с использованием обработчика
        print("Инициализация загрузки данных...")
        data_processor = DataProcessor('data.json')
        questions, _, answers, _ = data_processor.prepare_datasets(test_size=0.1)

        # Создание экземпляра FAQ бота
        faq_bot = FAQBot()

        # Построение базы знаний из вопросов и ответов
        faq_bot.build_knowledge_base(questions, answers)

        # Сохранение базы знаний на диск
        faq_bot.save_model('faq_bot.pth')

        # Тестовые вопросы для демонстрации работы бота
        example_questions = [
            "Сколько бюджетных мест на факультете?",
            "Как получить повышенную стипендию?",
            "Есть ли на факультете военная кафедра?",
            "Как организована практика студентов?",
            "Есть ли на факультете магистратура?"
        ]

        print("\nРезультаты работы бота на примерах:")
        for example in example_questions:
            print(f"\nВопрос: {example}")
            answers = faq_bot.find_answer(example, top_k=1)
            for response, confidence in answers:
                print(f"Уверенность: {confidence:.2%}")
                print(f"Ответ: {response}")

    except Exception as error:
        print(f"Возникла ошибка: {str(error)}")
