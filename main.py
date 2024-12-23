import os

from Bot import FAQBot
from train import train


def run_interactive_bot():
    try:
        # Проверяем, существует ли файл сохранённой модели
        if not os.path.isfile('faq_bot.pth'):
            print("Модель не найдена. Пожалуйста, сначала выполните настройку, запустив setup.py.")
            return

        print("Загрузка сохранённой модели...")
        faq_bot = FAQBot()
        faq_bot.load_model('faq_bot.pth')
        print("Модель успешно загружена!")

        print("\nВведите свои вопросы (для выхода используйте команды 'выход', 'exit', 'quit' или 'q').")

        while True:
            # Чтение вопроса пользователя
            user_question = input("\nВаш вопрос: ").strip()

            # Проверяем команды выхода
            if user_question.lower() in ['выход', 'exit', 'quit', 'q']:
                print("Спасибо за использование бота! До свидания.")
                break

            # Проверяем на пустую строку
            if not user_question:
                print("Ошибка: Вопрос не может быть пустым. Попробуйте снова.")
                continue

            try:
                # Поиск ответа с помощью бота
                responses = faq_bot.find_answer(user_question, top_k=1)

                # Вывод найденного ответа
                for response, confidence in responses:
                    print(f"Ответ (уверенность: {confidence:.2%}): {response}")

                    # Уведомление о низкой уверенности
                    if confidence < 0.5:
                        print("\nПримечание: уверенность в ответе низкая. Попробуйте переформулировать ваш вопрос.")

            except Exception as error:
                print(f"Произошла ошибка при обработке вашего вопроса: {str(error)}")

    except Exception as error:
        print(f"Ошибка при запуске интерактивного бота: {str(error)}")


if __name__ == "__main__":
    # Если модель отсутствует, выполняем процесс её создания
    if not os.path.isfile('faq_bot.pth'):
        train()
    run_interactive_bot()
