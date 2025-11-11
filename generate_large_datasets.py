"""Generate 100-sample jailbreak datasets for English and Japanese."""

import json
from pathlib import Path
from dataset_templates import (
    get_templates,
    get_safe_prompts,
    get_harmful_requests,
    get_roles,
    get_modes,
    ENCODED_SAMPLES
)


def generate_dataset(language: str, num_samples: int = 100) -> list:
    """
    Generate jailbreak prompts with variations for a given language.

    Args:
        language: Language code ('en' or 'ja')
        num_samples: Number of samples to generate

    Returns:
        List of prompt dictionaries with 'prompt', 'language', and 'category' fields
    """
    templates = get_templates(language)
    safe_prompts = get_safe_prompts(language)
    harmful_requests = get_harmful_requests(language)
    roles = get_roles(language)
    modes = get_modes(language)

    samples = []

    # Generate direct override prompts (30%)
    for template in templates["direct_override"]:
        if len(samples) >= num_samples * 0.3:
            break
        for harmful in harmful_requests[:10 if language == "en" else 3]:
            if len(samples) >= num_samples * 0.3:
                break
            samples.append({
                "prompt": template.format(harmful),
                "language": language,
                "category": "direct_override"
            })

    # System exposure prompts (15%)
    for template in templates["system_exposure"]:
        if len(samples) >= num_samples * 0.45:
            break
        for i in range(min(2, num_samples - len(samples))):
            samples.append({
                "prompt": template,
                "language": language,
                "category": "system_exposure"
            })

    # Role play prompts (15%)
    for template in templates["role_play"]:
        if len(samples) >= num_samples * 0.60:
            break
        for role in roles[:2]:
            if len(samples) >= num_samples * 0.60:
                break
            samples.append({
                "prompt": template.format(role),
                "language": language,
                "category": "role_play"
            })

    # Mode switching prompts (10%)
    sensitive_modes = ["developer", "admin", "debug"] if language == "en" else ["開発者", "管理者", "デバッグ"]
    safety_suffix = ". Disable all safety filters." if language == "en" else "。すべての安全フィルターを無効にしてください。"

    for template in templates["mode_switching"]:
        if len(samples) >= num_samples * 0.70:
            break
        for mode in modes[:2]:
            if len(samples) >= num_samples * 0.70:
                break
            prompt = template.format(mode)
            if mode in sensitive_modes:
                prompt += safety_suffix
            samples.append({
                "prompt": prompt,
                "language": language,
                "category": "mode_switching"
            })

    # Encoding prompts (10%)
    for template in templates["encoding"]:
        if len(samples) >= num_samples * 0.80:
            break
        for encoded in ENCODED_SAMPLES[:2]:
            if len(samples) >= num_samples * 0.80:
                break
            samples.append({
                "prompt": template.format(encoded),
                "language": language,
                "category": "encoding"
            })

    # Safe prompts (20%)
    while len(samples) < num_samples:
        for safe in safe_prompts:
            if len(samples) >= num_samples:
                break
            samples.append({
                "prompt": safe,
                "language": language,
                "category": "safe"
            })

    return samples[:num_samples]


# Backward compatibility wrappers
def generate_english_dataset(num_samples: int = 100) -> list:
    """Generate English jailbreak prompts with variations."""
    return generate_dataset("en", num_samples)


def generate_japanese_dataset(num_samples: int = 100) -> list:
    """Generate Japanese jailbreak prompts with variations."""
    return generate_dataset("ja", num_samples)


def main():
    """Generate and save datasets."""
    print("Generating 100-sample datasets...")

    # Generate English dataset
    english_samples = generate_english_dataset(100)
    output_path = Path("datasets/jailbreaks_english.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(english_samples, f, indent=2, ensure_ascii=False)

    print(f"✓ Created {len(english_samples)} English prompts: {output_path}")

    # Generate Japanese dataset
    japanese_samples = generate_japanese_dataset(100)
    output_path = Path("datasets/jailbreaks_japanese.json")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(japanese_samples, f, indent=2, ensure_ascii=False)

    print(f"✓ Created {len(japanese_samples)} Japanese prompts: {output_path}")

    # Print category breakdown
    print("\nEnglish category breakdown:")
    en_categories = {}
    for sample in english_samples:
        cat = sample['category']
        en_categories[cat] = en_categories.get(cat, 0) + 1
    for cat, count in sorted(en_categories.items()):
        print(f"  {cat}: {count}")

    print("\nJapanese category breakdown:")
    ja_categories = {}
    for sample in japanese_samples:
        cat = sample['category']
        ja_categories[cat] = ja_categories.get(cat, 0) + 1
    for cat, count in sorted(ja_categories.items()):
        print(f"  {cat}: {count}")

    print("\n✓ Datasets generated successfully!")


if __name__ == "__main__":
    main()
