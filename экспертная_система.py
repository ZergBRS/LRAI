from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Rule:
    conditions: tuple[tuple[str, str], ...]
    conclusion: tuple[str, str]

    def __str__(self) -> str:
        left = " И ".join(f"{name}={value}" for name, value in self.conditions)
        name, value = self.conclusion
        return f"ЕСЛИ {left} ТО {name}={value}"


CONDITION_SPLIT = re.compile(r"\s+И\s+")



def parse_rule(line: str) -> Rule:
    match = re.fullmatch(r"\s*ЕСЛИ\s+(.+?)\s+ТО\s+(.+?)\s*", line)
    if not match:
        raise ValueError("Ожидается формат: ЕСЛИ ... ТО Объект=Значение")

    def parse_pair(text: str) -> tuple[str, str]:
        if text.count("=") != 1:
            raise ValueError(f"Некорректная пара Объект=Значение: {text}")
        name, value = (part.strip() for part in text.split("=", 1))
        if not name or not value:
            raise ValueError(f"Объект и значение не должны быть пустыми: {text}")
        return name, value

    conditions = tuple(parse_pair(part) for part in CONDITION_SPLIT.split(match.group(1)))
    if not conditions:
        raise ValueError("У правила должно быть хотя бы одно условие")
    return Rule(conditions, parse_pair(match.group(2)))


def load_rules(path: Path) -> list[Rule]:
    rules = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        content = line.strip()
        if not content or content.startswith("#"):
            continue
        try:
            rules.append(parse_rule(content))
        except ValueError as error:
            raise ValueError(f"Ошибка в файле правил, строка {line_number}: {error}") from error
    return rules


def save_rules(path: Path, rules: list[Rule]) -> None:
    path.write_text("\n".join(map(str, rules)) + "\n", encoding="utf-8")


def forward_chain(
    rules: list[Rule], facts: dict[str, str], ask_fact,
    goal: str = "Заключение",
) -> tuple[dict[str, str], list[str]]:
    """Прямой вывод; ask_fact(object_name) возвращает значение или пустую строку."""
    working_memory = dict(facts)
    trace: list[str] = []
    fired: set[int] = set()

    while True:
        fired_this_pass = False
        for index, rule in enumerate(rules):
            if index in fired:
                continue
            if all(working_memory.get(name) == value for name, value in rule.conditions):
                name, value = rule.conclusion
                fired.add(index)
                fired_this_pass = True
                if name not in working_memory:
                    working_memory[name] = value
                    trace.append(f"Сработало правило {index + 1}: {rule}; добавлен факт {name}={value}")
                elif working_memory[name] != value:
                    trace.append(
                        f"Правило {index + 1} выведено из рассмотрения: {name} уже имеет значение "
                        f"{working_memory[name]}"
                    )
        if fired_this_pass:
            continue
        if goal in working_memory:
            break

        missing = next(
            (name for rule in rules for name, value in rule.conditions
             if name not in working_memory and name != goal),
            None,
        )
        if missing is None:
            trace.append("Нет подходящих правил и неизвестных условий для запроса.")
            break
        supplied = ask_fact(missing)
        if supplied is None or not str(supplied).strip():
            trace.append(f"Данные для факта «{missing}» не получены. Вывод завершён.")
            break
        working_memory[missing] = str(supplied).strip()
        trace.append(f"Пользователь добавил факт {missing}={working_memory[missing]}")

    return working_memory, trace


def print_facts(facts: dict[str, str]) -> None:
    print("\nРабочая база данных:")
    if not facts:
        print("  (пусто)")
    for name, value in facts.items():
        print(f"  {name}={value}")


def edit_rules(path: Path, rules: list[Rule]) -> None:
    while True:
        print("\nПравила:")
        for index, rule in enumerate(rules, 1):
            print(f"{index}. {rule}")
        print("1 — добавить, 2 — удалить, 3 — редактировать, 0 — назад")
        choice = input("Выбор: ").strip()
        try:
            if choice == "1":
                rules.append(parse_rule(input("Новое правило: ")))
            elif choice == "2":
                index = int(input("Номер правила: ")) - 1
                if not 0 <= index < len(rules):
                    raise ValueError("Нет правила с таким номером")
                rules.pop(index)
            elif choice == "3":
                index = int(input("Номер правила: ")) - 1
                if not 0 <= index < len(rules):
                    raise ValueError("Нет правила с таким номером")
                rules[index] = parse_rule(input("Исправленное правило: "))
            elif choice == "0":
                return
            else:
                print("Неизвестная команда.")
                continue
            save_rules(path, rules)
            print("База правил сохранена.")
        except (ValueError, IndexError) as error:
            print(f"Не удалось выполнить операцию: {error}")


def main() -> None:
    rules_path = Path(__file__).with_name("база_правил.txt")
    try:
        rules = load_rules(rules_path)
    except (OSError, ValueError) as error:
        print(f"Не удалось загрузить базу правил: {error}")
        return

    while True:
        print("\nПерсональный контроль питания — демонстрационная экспертная система")
        print("1 — запустить вывод, 2 — показать правила, 3 — изменить правила, 0 — выход")
        choice = input("Выбор: ").strip()
        if choice == "0":
            break
        if choice == "2":
            for index, rule in enumerate(rules, 1):
                print(f"{index}. {rule}")
            continue
        if choice == "3":
            edit_rules(rules_path, rules)
            continue
        if choice != "1":
            print("Неизвестная команда.")
            continue

        facts = {
            "Время": input("Время суток (утро/день/вечер): ").strip().lower(),
            "Последний_приём": input("Последний приём пищи (давно/недавно): ").strip().lower(),
            "Хочется_пить": input("Хочется пить? (да/нет): ").strip().lower(),
        }
        facts = {name: value for name, value in facts.items() if value}
        print_facts(facts)

        def ask_fact(name: str) -> str:
            return input(f"Введите значение для нового факта «{name}» (Enter — завершить): ")

        result, trace = forward_chain(rules, facts, ask_fact)
        print("\nХод прямого вывода:")
        for step in trace:
            print(f"- {step}")
        print_facts(result)
        if "Заключение" in result:
            print(f"\nБытовая рекомендация: {result['Заключение']}")
        else:
            print("\nЗаключение не получено.")
        if "Совет_о_воде" in result:
            print(f"Подсказка о воде: {result['Совет_о_воде']}")


if __name__ == "__main__":
    main()
