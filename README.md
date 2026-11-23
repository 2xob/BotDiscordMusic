# Bot Discord – séparation de pistes (Demucs)

Envoie un fichier audio dans un salon, choisis ce que tu veux enlever
(Drums, Bass, Voice, Drums + Bass) et le bot te renvoie le MP3 avec le BPM.
100 % local : aucune API externe.

## Discord Developer Portal
1. Crée l'application + le bot, copie le token.
2. Onglet **Bot** → active **Message Content Intent** (obligatoire pour lire les pièces jointes).
3. Invite le bot avec les permissions : View Channels, Send Messages, Attach Files,
   Read Message History.

## Déploiement Railway
1. Pousse ces fichiers (`bot.py`, `Dockerfile`, `requirements.txt`) sur un repo GitHub.
2. Railway → New Project → Deploy from GitHub repo (le Dockerfile est détecté tout seul).
3. Onglet **Variables** : ajoute `DISCORD_TOKEN` (et les autres de `.env.example` si besoin).
4. Prévois au moins 2 à 4 Go de RAM pour le service.

Le premier build est long (PyTorch + modèle Demucs), c'est normal.

## Variables
| Variable | Défaut | Rôle |
|---|---|---|
| `DISCORD_TOKEN` | – | Token du bot (obligatoire) |
| `CHANNEL_ID` | 0 | Limiter à un salon |
| `UPLOAD_LIMIT_MB` | 10 | Limite d'upload du serveur (le bitrate MP3 s'adapte) |
| `MAX_DURATION_SECONDS` | 600 | Durée max d'un morceau |
| `MAX_QUEUE` | 15 | Taille max de la file |
| `BPM_MIN` / `BPM_MAX` | 75 / 175 | Plage de correction du BPM |

## Ajouter un bouton
Dans `bot.py`, ajoute une ligne au dictionnaire `OPTIONS`, par exemple :
`"drums_voice": ("Drums + Voice", {"drums", "vocals"}, "🎶")`.
