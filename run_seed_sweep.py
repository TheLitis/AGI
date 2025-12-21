"""Helper to run run_all.py over multiple seeds for a given mode."""

import argparse
from typing import List

from run_all import main as run_single


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Запуск пайплайна proto-creature AGI для нескольких сидов подряд.\n"
            "Обёртка над run_all.main()."
        )
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Список сидов, через пробел. Пример: --seeds 0 1 2 3 4.\n"
            "Если не указано, по умолчанию используется [0, 1, 2, 3, 4]."
        ),
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=0,
        help=(
            "Стартовый сид для генерации диапазона, если не указан --seeds.\n"
            "Работает вместе с --n-seeds."
        ),
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help=(
            "Сколько сидов прогнать, начиная с --start-seed, "
            "если явно не указан список --seeds. По умолчанию 5."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=[
            "all",
            "stage1",
            "stage2",
            "stage3",
            "stage3b",
            "stage3c",
            "stage4",
            "lifelong",
            "lifelong_train",
        ],
        help=(
            "Режим, который передаётся в run_all.main(..., mode=...):\n"
            "  all    — полный пайплайн Stage1→4\n"
            "  stage1 — только сбор рандомного опыта + world model\n"
            "  stage2 — только policy (без self)\n"
            "  stage3 — только обучение SelfModel\n"
            "  stage3b — только self-reflection по traits\n"
            "  stage3c — обновление SelfModel после self-reflection\n"
            "  stage4 — дообучение policy с self-моделью и планировщиком"
        ),
    )
    parser.add_argument(
        "--use-skills",
        action="store_true",
        help="Enable hierarchical skills and high-level policy.",
    )
    parser.add_argument(
        "--skill-mode",
        type=str,
        default="handcrafted",
        choices=["handcrafted", "latent", "mixed"],
        help="Skill selection backend: handcrafted (default), latent, or mixed.",
    )
    parser.add_argument(
        "--n-latent-skills",
        type=int,
        default=0,
        help="Number of latent skills (used when skill-mode=latent/mixed).",
    )
    return parser.parse_args()


def build_seed_list(args: argparse.Namespace) -> List[int]:
    """
    Логика выбора сидов:

    - если явно указан --seeds 0 1 2 → используем ровно их;
    - иначе: берём диапазон [start_seed, start_seed + n_seeds).
    """
    if args.seeds is not None and len(args.seeds) > 0:
        return args.seeds
    return list(range(args.start_seed, args.start_seed + args.n_seeds))


def run_sweep():
    args = parse_args()
    seeds = build_seed_list(args)

    print("===============================================")
    print(" Proto-creature AGI seed sweep runner")
    print("===============================================")
    print(f"Mode:  {args.mode}")
    print(f"Seeds: {seeds}")
    print("===============================================\n")

    for idx, seed in enumerate(seeds):
        print("\n-----------------------------------------------")
        print(f"[{idx + 1}/{len(seeds)}] START seed={seed}, mode={args.mode}")
        print("-----------------------------------------------\n")

        try:
            # вызывем run_all.main(seed=..., mode=...)
            run_single(
                seed=seed,
                mode=args.mode,
                use_skills=args.use_skills,
                skill_mode=args.skill_mode,
                n_latent_skills=args.n_latent_skills,
            )
        except Exception as e:
            # не роняем весь спип, если один сид упал
            print(f"\n[ERROR] Seed {seed} crashed with exception: {e}")
            print("Продолжаю со следующим сидом.\n")
            continue

        print("\n-----------------------------------------------")
        print(f"[{idx + 1}/{len(seeds)}] FINISHED seed={seed}, mode={args.mode}")
        print("-----------------------------------------------\n")

    print("===============================================")
    print(" Seed sweep finished.")
    print("===============================================")


if __name__ == "__main__":
    run_sweep()
