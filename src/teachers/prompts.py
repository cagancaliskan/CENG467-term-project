"""Teacher prompt templates.

Two variants per the project plan: 'concise' (production default) and 'detailed'
(more guidance, used in the prompt-design ablation). Both prompts are
documented as a course requirement.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSpec:
    name: str
    system: str
    user_template: str  # must contain {article}
    max_output_tokens: int


CONCISE = PromptSpec(
    name="concise",
    system=(
        "Sen, Türkçe haber metinlerini özetleyen profesyonel bir editörsün. "
        "Tek paragraflık, kısa ve doğru özetler üretirsin. Yalnızca metinde geçen "
        "bilgileri kullan; haber metninde yoksa hiçbir şey uydurma."
    ),
    user_template=(
        "Aşağıdaki haber metnini Türkçe olarak özetle. Çıktı 2-4 cümlelik tek bir paragraf "
        "olsun ve yalnızca özet içersin. Başlık, madde işareti veya açıklama ekleme.\n\n"
        "Haber:\n{article}\n\nÖzet:"
    ),
    max_output_tokens=160,
)


DETAILED = PromptSpec(
    name="detailed",
    system=(
        "Sen, Türkçe haber metinlerini özetleyen profesyonel bir editörsün. "
        "Görevin haberi sadık bir şekilde, kim-ne-nerede-ne zaman-neden bilgilerini "
        "koruyarak özetlemektir. Üretici dil kullan ama metinde olmayan hiçbir bilgi ekleme."
    ),
    user_template=(
        "Aşağıdaki Türkçe haber metnini, gazetecilik standartlarına uygun olarak özetle.\n"
        "Kurallar:\n"
        "1) Çıktı tek paragraf, 3-5 cümle olsun.\n"
        "2) İlk cümle haberin ana olayını (ne oldu, kim, nerede) içersin.\n"
        "3) Sayısal değerler (tarih, oran, miktar) doğrudan metinden alınsın.\n"
        "4) Sübjektif yorum, tahmin veya metinde geçmeyen detay ekleme.\n"
        "5) Başlık, etiket veya madde işareti kullanma; sadece düz metin özet ver.\n\n"
        "Haber:\n{article}\n\nÖzet:"
    ),
    max_output_tokens=220,
)


PROMPTS: dict[str, PromptSpec] = {p.name: p for p in (CONCISE, DETAILED)}


def get_prompt(name: str) -> PromptSpec:
    if name not in PROMPTS:
        raise KeyError(f"Unknown prompt variant: {name!r}. Choose from {list(PROMPTS)}")
    return PROMPTS[name]
