import argparse

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data_loader import balance_train_dataset, load_archive, load_df
from models import DecisionTreeClassifier


def parse_arguments() -> argparse.Namespace:
    """
    Разбирает аргументы командной строки для запуска обучения.

    Parameters:
        None: Функция не принимает параметры. По умолчанию: None.

    Returns:
        argparse.Namespace: Пространство имён с параметрами запуска.

    Fallbacks:
        Если параметр не указан, используется режим обработки пропусков "none".
    """
    # Создаём парсер для настройки запуска из терминала.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pass-processing-type",
        default="none",
        choices=["none", "weight"],
        help="Тип обработки пропусков: none или weight.",
    )
    parser.add_argument(
        "--show-tree",
        default="no",
        choices=["no", "yes"],
        help="Печатать дерево в конце: no или yes.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Загружает данные, обучает дерево решений и выводит accuracy.

    Parameters:
        None: Функция не принимает параметры. По умолчанию: None.

    Returns:
        None: Результат оценки выводится в консоль.

    Fallbacks:
        Ошибки загрузки данных и обучения передаются вызывающему коду.
    """
    # Разбираем параметры запуска перед обучением модели.
    arguments = parse_arguments()

    # Загружаем архив и читаем таблицу с датасетом.
    load_archive()
    dataframe = load_df()

    # Разделяем признаки и целевую переменную по последнему столбцу.
    feature_names = dataframe.columns[:-1]
    label_name = dataframe.columns[-1]

    # Формируем стратифицированные обучающую и тестовую выборки.
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(dataframe[feature_names]),
        np.array(dataframe[label_name]),
        test_size=0.2,
        random_state=42,
        stratify=dataframe[label_name],
    )

    # Балансируем только обучающую выборку, не затрагивая тестовую.
    x_train, y_train = balance_train_dataset(
        x_train,
        y_train,
        feature_names,
        dataframe,
    )

    # Обучаем собственную реализацию дерева решений.
    tree_classifier = DecisionTreeClassifier(
        features=x_train,
        feature_names=feature_names,
        labels=y_train,
        pass_processing_type=arguments.pass_processing_type,
    )
    tree_classifier.id3()

    # Оцениваем качество классификации на тестовой выборке.
    y_predicted = tree_classifier.predict(x_test)
    print("Accuracy:", accuracy_score(y_test, y_predicted))

    # Печатаем дерево только по явному запросу пользователя.
    if arguments.show_tree == "yes":
        tree_classifier.print_tree()


# Запускаем сценарий только при прямом вызове файла.
if __name__ == "__main__":
    main()
