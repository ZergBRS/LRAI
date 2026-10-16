from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import tempfile


@dataclass(frozen=True)
class Rule:
    conditions: tuple[tuple[str, str], ...]
    conclusion: tuple[str, str]

    def __str__(self) -> str:
        left = " И ".join(f"{name}={value}" for name, value in self.conditions)
        name, value = self.conclusion
        return f"ЕСЛИ {left} ТО {name}={value}"


CONDITION_SPLIT = re.compile(r"\s+И\s+", re.IGNORECASE)


def parse_pair(text: str) -> tuple[str, str]:
    if text.count("=") != 1 or "\n" in text or "\r" in text:
        raise ValueError("Ожидается один факт: Объект=Значение")
    name, value = (part.strip() for part in text.split("=", 1))
    if not name or not value or re.search(r"\s+(И|ТО)\s+", text, re.IGNORECASE):
        raise ValueError("Объект и значение должны быть непустыми и одиночными")
    return name, value


def parse_rule(line: str) -> Rule:
    if "\n" in line or "\r" in line:
        raise ValueError("Одно правило должно занимать одну строку")
    match = re.fullmatch(r"\s*ЕСЛИ\s+(.+?)\s+ТО\s+(.+?)\s*", line, re.IGNORECASE)
    if not match:
        raise ValueError("Ожидается формат: ЕСЛИ ... ТО Объект=Значение")
    conditions = tuple(
        parse_pair(part) for part in CONDITION_SPLIT.split(match.group(1))
    )
    seen = {}
    for name, value in conditions:
        if name in seen and seen[name] != value:
            raise ValueError(f"Противоречивые условия для объекта {name}")
        seen[name] = value
    return Rule(conditions, parse_pair(match.group(2)))


def load_rules(path: Path) -> list[Rule]:
    rules = []
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    for line_number, line in enumerate(lines, 1):
        content = line.strip()
        if not content or content.startswith("#"):
            continue
        try:
            rules.append(parse_rule(content))
        except ValueError as error:
            raise ValueError(
                f"Ошибка в файле правил, строка {line_number}: {error}"
            ) from error
    return rules


def save_rules(path: Path, rules: list[Rule]) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write("\n".join(map(str, rules)) + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def edit_rules(path: Path, rules: list[Rule]) -> None:
    while True:
        print("\nПравила:")
        for index, rule in enumerate(rules, 1):
            print(f"{index}. {rule}")
        print("1 — добавить, 2 — удалить, 3 — редактировать, 0 — назад")
        choice = input("Выбор: ").strip()
        try:
            updated = rules.copy()
            if choice == "1":
                updated.append(parse_rule(input("Новое правило: ")))
            elif choice == "2":
                index = int(input("Номер правила: ")) - 1
                if not 0 <= index < len(rules):
                    raise ValueError("Нет правила с таким номером")
                updated.pop(index)
            elif choice == "3":
                index = int(input("Номер правила: ")) - 1
                if not 0 <= index < len(rules):
                    raise ValueError("Нет правила с таким номером")
                updated[index] = parse_rule(input("Исправленное правило: "))
            elif choice == "0":
                return
            else:
                print("Неизвестная команда.")
                continue
            save_rules(path, updated)
            rules[:] = updated
            print("База правил сохранена.")
        except (ValueError, IndexError, OSError) as error:
            print(f"Не удалось выполнить операцию: {error}")


Fact = tuple[str, str]


def parse_situation(text: str) -> tuple[Fact, ...]:
    """Целевая ситуация может включать несколько обязательных фактов через И."""
    return parse_rule(f"ЕСЛИ {text} ТО Проверка=готово").conditions


@dataclass
class Result:
    status: str
    facts: dict[str, str]
    trace: list[str]
    missing: tuple[Fact, ...] = ()
    used_rules: tuple[int, ...] = ()


def backward_chain(rules, initial, goals, ask_fact=None) -> Result:
    """Поиск от целей к фактам с перебором альтернатив и возвратом по И."""
    known = dict(initial)
    trace = []
    asked = set()
    produced = {rule.conclusion[0] for rule in rules}

    def prove_all(conditions, memory, active):
        if not conditions:
            yield memory, frozenset(), ()
            return
        first, *rest = conditions
        for state, unknown, steps in prove(first, memory, active):
            for end, later_unknown, later_steps in prove_all(rest, state, active):
                yield end, unknown | later_unknown, steps + later_steps

    def prove(goal, memory, active):
        name, value = goal
        label = f"{name}={value}"
        trace.append(f"Проверяется подцель: {label}")
        if any(key in memory and memory[key] != val for key, val in known.items()):
            return
        memory = dict(memory, **known)
        if name in memory:
            if memory[name] == value:
                if name in known:
                    trace.append(f"Известный факт подтверждает: {label}")
                else:
                    trace.append(f"Совпадение в проверяемой ветви: {label}")
                yield memory, frozenset(), ()
            else:
                trace.append(
                    f"Противоречие: нужно {label}, известно {name}={memory[name]}"
                )
            return
        if goal in active:
            trace.append(f"Цикл зависимости: {label}; эта ветвь остановлена")
            return
        candidates = [(i, r) for i, r in enumerate(rules, 1) if r.conclusion == goal]
        if not candidates:
            if name in produced:
                trace.append(f"Нет правила, выводящего {label}")
                return
            if ask_fact is not None and name not in asked:
                asked.add(name)
                answer = ask_fact(name)
                if answer is not None and str(answer).strip():
                    known[name] = parse_pair(f"{name}={str(answer).strip()}")[1]
                    trace.append(f"Получен факт: {name}={known[name]}")
            if name in known:
                if known[name] == value:
                    yield dict(memory, **known), frozenset(), ()
                else:
                    trace.append(
                        f"Противоречие: нужно {label}, получено {name}={known[name]}"
                    )
                return
            trace.append(f"Нет сведений о {name}; {label} пока не подтверждён")
            # Допущение используется только при поиске возможной ветви.
            # В рабочую базу результата неизвестные факты не записываются.
            yield dict(memory, **{name: value}), frozenset({goal}), ()
            return
        for index, rule in candidates:
            trace.append(f"Проверка правила {index}: {rule}")
            branches = prove_all(rule.conditions, memory, active | {goal})
            for state, unknown, steps in branches:
                state = dict(state, **{name: value})
                yield state, unknown, steps + (index,)

    possible = None
    for memory, unknown, steps in prove_all(tuple(goals), known, frozenset()):
        if not unknown:
            trace.append("Целевая ситуация доказана.")
            return Result("proved", dict(memory, **known), trace,
                          used_rules=tuple(dict.fromkeys(steps)))
        if possible is None:
            possible = unknown
    if possible is not None:
        trace.append("Недостаточно сведений: цель не доказана и не опровергнута.")
        return Result("unknown", known.copy(), trace, tuple(sorted(possible)))
    trace.append("Цель опровергнута в заданной модели: подходящих доказательств нет.")
    return Result("refuted", known.copy(), trace)


def load_facts(path: Path) -> dict[str, str]:
    facts = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            name, value = parse_pair(line)
            if name in facts and facts[name] != value:
                raise ValueError(f"Противоречивые значения объекта {name}")
            facts[name] = value
        except ValueError as error:
            raise ValueError(f"Начальные факты, строка {line_number}: {error}") from error
    return facts


def print_facts(facts):
    print("\nРабочая база данных:")
    for name, value in facts.items():
        print(f"  {name}={value}")
    if not facts:
        print("  (пусто)")


def choose_value(values, prompt):
    for i, value in enumerate(values, 1):
        print(f"{i} — {value}")
    answer = input(prompt).strip()
    normalize = lambda s: s.casefold().replace("ё", "е").replace("_", " ")
    same = next((v for v in values if normalize(v) == normalize(answer)), None)
    if same is not None:
        return same
    if answer.isdecimal() and 1 <= int(answer) <= len(values):
        return values[int(answer) - 1]
    return answer


def input_goal(rules, default):
    print("\nЦель по умолчанию — доказать все условия:")
    for name, value in parse_situation(default):
        print(f"  {name}={value}")
    while True:
        try:
            text = input("Цель Объект=Значение [И ...]; ? — варианты; Enter — по умолчанию: ").strip()
            if text == "?":
                objects = list(dict.fromkeys(rule.conclusion[0] for rule in rules))
                objects.sort(key=lambda name: name != "Причина")
                if not objects:
                    print("Правил нет. Введите свою цель.")
                    continue
                name = choose_value(objects, "Объект: ")
                values = list(dict.fromkeys(r.conclusion[1] for r in rules if r.conclusion[0] == name))
                value = choose_value(values, "Значение: ")
                text = f"{name}={value}"
            return parse_situation(text or default)
        except ValueError as error:
            print(f"Некорректная цель: {error}")


def ask_observation(rules, name):
    print(f"\nДля доказательства нужны сведения: {name.replace('_', ' ')}")
    values = list(dict.fromkeys(v for r in rules for n, v in r.conditions if n == name))
    if values and set(values) <= {"да", "нет"}:
        values = ["да", "нет"]
    elif values and "другое" not in values:
        values.append("другое")
    while True:
        try:
            value = choose_value(values, "Значение (Enter — сведений нет): ")
            return parse_pair(f"{name}={value}")[1] if value else None
        except ValueError as error:
            print(f"Некорректное значение: {error}")


def recommendations(rules, result):
    if result.status != "proved":
        return []
    targets = dict.fromkeys(r.conclusion for r in rules if r.conclusion[0] == "Рекомендация")
    return [value for name, value in targets
            if backward_chain(rules, result.facts, ((name, value),)).status == "proved"]


def diagnose_all(rules, initial, ask_fact=None, show_goal=None):
    """Независимые гипотезы используют общие наблюдения, но разные выводы."""
    observations = dict(initial)
    asked = set()

    def ask(name):
        if name not in asked and ask_fact is not None:
            asked.add(name)
            value = ask_fact(name)
            if value is not None and str(value).strip():
                observations[name] = parse_pair(f"{name}={str(value).strip()}")[1]
                return observations[name]
        return observations.get(name)

    targets = dict.fromkeys(r.conclusion for r in rules if r.conclusion[0] == "Причина")
    reports = []
    for goal in targets:
        if show_goal is not None:
            show_goal(goal)
        reports.append((goal, backward_chain(rules, observations, (goal,), ask)))
    return reports, observations


def main():
    folder = Path(__file__).resolve().parent
    try:
        rules = load_rules(folder / "база_правил.txt")
        facts = load_facts(folder / "начальные_факты.txt")
        default = (folder / "целевая_ситуация.txt").read_text(encoding="utf-8-sig").strip()
        parse_situation(default)
    except (OSError, ValueError) as error:
        print(f"Ошибка загрузки: {error}")
        return
    while True:
        print("\nЛР № 2 — почему приложение не запускается, обратная цепочка")
        print("1 — проверить цель, 2 — показать правила, 3 — изменить правила,")
        print("4 — начальная ситуация, 5 — проверить все причины, 0 — выход")
        choice = input("Выбор: ").strip()
        if choice == "0":
            return
        if choice == "2":
            for i, rule in enumerate(rules, 1):
                print(f"{i}. {rule}")
        elif choice == "3":
            edit_rules(folder / "база_правил.txt", rules)
        elif choice == "5":
            reports, facts = diagnose_all(
                rules, facts, lambda name: ask_observation(rules, name),
                show_goal=lambda goal: print(f"\nПроверка гипотезы: {goal[1]}")
            )
            print("\nСравнение гипотез:")
            labels = {"proved": "ПОДТВЕРЖДЕНА", "refuted": "ИСКЛЮЧЕНА", "unknown": "НЕТ ДАННЫХ"}
            for goal, result in reports:
                print(f"{goal[1]}: {labels[result.status]}")
                if result.status == "refuted":
                    reason = next((s for s in result.trace if s.startswith(
                        ("Противоречие:", "Нет правила", "Цикл зависимости:")
                    )), "Все варианты доказательства исключены.")
                    print("  " + reason)
                if result.missing:
                    print("  Не хватает для одной ветви:", "; ".join(f"{n}={v}" for n, v in result.missing))
                for value in recommendations(rules, result):
                    print("  Следующий шаг:", value.replace("_", " "))
            if not any(result.status == "proved" for _, result in reports):
                print("Подтверждённых причин среди гипотез базы нет.")
            print("Подробное доказательство выбранной гипотезы — команда 1.")
        elif choice == "4":
            print_facts(facts)
            print("Объект=Значение — добавить/заменить; -Объект — удалить; Enter — назад.")
            while True:
                text = input("Факт: ").strip()
                if not text:
                    break
                try:
                    if text.startswith("-"):
                        facts.pop(text[1:].strip(), None)
                    else:
                        name, value = parse_pair(text)
                        facts[name] = value
                    print_facts(facts)
                except ValueError as error:
                    print(f"Некорректный факт: {error}")
        elif choice == "1":
            goals = input_goal(rules, default)
            print_facts(facts)

            result = backward_chain(rules, facts, goals, lambda name: ask_observation(rules, name))
            print("\nХод обратного вывода:")
            for step in result.trace:
                print(f"- {step}")
            print_facts(result.facts)
            print("\nРезультат:", {"proved": "ЦЕЛЬ ДОКАЗАНА", "refuted": "ЦЕЛЬ ОПРОВЕРГНУТА В ЗАДАННОЙ МОДЕЛИ",
                                     "unknown": "НЕДОСТАТОЧНО СВЕДЕНИЙ"}[result.status])
            for name, value in goals:
                print(f"  Проверяемая цель: {name}={value}")
            if result.used_rules:
                print("Правила доказательства:", ", ".join(map(str, result.used_rules)))
            for value in recommendations(rules, result):
                print("Рекомендация:", value.replace("_", " "))
            if result.missing:
                print("Для одной возможной ветви не хватает:",
                      "; ".join(f"{n}={v}" for n, v in result.missing))
                return
        else:
            print("Неизвестная команда.")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nРабота программы завершена.")
