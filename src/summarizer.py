import argparse
import json
import os
from pathlib import Path
from typing import List
from urllib import response

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field, ValidationError


class SummaryResult(BaseModel):
    resume: str = Field(
        description="Résumé court, clair et fidèle au texte fourni."
    )
    points_cles: List[str] = Field(
        description="Liste des idées principales du texte."
    )
    actions: List[str] = Field(
        description="Liste des actions concrètes à faire après lecture du texte."
    )


SYSTEM_PROMPT = """
Tu es un assistant spécialisé dans le résumé professionnel.

Ta mission :
- Lire le texte fourni.
- Produire un résumé clair.
- Extraire les points clés.
- Proposer des actions concrètes.

Règles :
- Ne pas inventer d'informations absentes du texte.
- Répondre en français.
- Faire un résumé de 3 à 5 phrases maximum.
- Extraire entre 3 et 7 points clés.
- Proposer entre 1 et 5 actions concrètes.
- Les actions doivent commencer par un verbe d'action.
"""


def read_text_file(file_path: str) -> str:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    content = path.read_text(encoding="utf-8").strip()

    if not content:
        raise ValueError("Le fichier est vide.")

    return content


def summarize_text(text: str) -> SummaryResult:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    use_fake_llm = os.getenv("USE_FAKE_LLM", "false").lower() == "true"

    if use_fake_llm:
        return SummaryResult(
            resume="Résumé de test généré sans appel API.",
            points_cles=[
                "Le système lit un texte depuis un fichier.",
                "Le résultat respecte une structure JSON.",
                "Le projet peut être testé sans consommer d'appel API.",
            ],
            actions=[
                "Vérifier la configuration de la clé Gemini.",
                "Désactiver USE_FAKE_LLM quand l'API Gemini est prête.",
            ],
        )

    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY est introuvable. Vérifie ton fichier .env."
        )

    client = genai.Client(api_key=api_key)

    prompt = f"""
{SYSTEM_PROMPT}

Voici le texte à résumer :

{text}
"""

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": SummaryResult,
        },
    )

    return response.parsed


def save_summary(summary: SummaryResult, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = summary.model_dump()

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assistant de résumé intelligent avec Gemini."
    )

    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Chemin vers le fichier texte à résumer.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output/summary.json",
        help="Chemin du fichier JSON de sortie.",
    )

    args = parser.parse_args()

    try:
        text = read_text_file(args.file)
        summary = summarize_text(text)

        print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))

        save_summary(summary, args.output)

        print(f"\nRésumé sauvegardé dans : {args.output}")

    except FileNotFoundError as error:
        print(f"Erreur fichier : {error}")

    except ValueError as error:
        print(f"Erreur de contenu : {error}")

    except EnvironmentError as error:
        print(f"Erreur environnement : {error}")

    except ValidationError as error:
        print("Erreur : Gemini a retourné une réponse qui ne respecte pas le schéma.")
        print(error)

    except Exception as error:
        print(f"Erreur inattendue : {error}")


if __name__ == "__main__":
    main()