import asyncio
import itertools
import logging
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import discord
import librosa
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("stembot")

# ---------------------------------------------------------------- config
TOKEN = os.environ["DISCORD_TOKEN"]
CHANNEL_ID = int(os.getenv("CHANNEL_ID", "0"))            # 0 = tous les salons
UPLOAD_LIMIT_MB = float(os.getenv("UPLOAD_LIMIT_MB", "10"))  # 10 par défaut, 50 si serveur boosté niv.2
MAX_INPUT_MB = float(os.getenv("MAX_INPUT_MB", "50"))
MAX_DURATION = int(os.getenv("MAX_DURATION_SECONDS", "600"))
MAX_QUEUE = int(os.getenv("MAX_QUEUE", "15"))
BPM_MIN = float(os.getenv("BPM_MIN", "75"))
BPM_MAX = float(os.getenv("BPM_MAX", "175"))
DEMUCS_MODEL = os.getenv("DEMUCS_MODEL", "htdemucs")
DEMUCS_SEGMENT = os.getenv("DEMUCS_SEGMENT", "7")

ALL_STEMS = ("drums", "bass", "other", "vocals")

# clé -> (libellé du bouton, pistes à enlever, emoji)
# Pour ajouter un bouton, ajoute simplement une ligne ici.
OPTIONS = {
    "drums": ("Drums", {"drums"}, "🥁"),
    "bass": ("Bass", {"bass"}, "🎸"),
    "voice": ("Voice", {"vocals"}, "🎤"),
    "drums_bass": ("Drums + Bass", {"drums", "bass"}, "🔊"),
}

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma"}

# ---------------------------------------------------------------- état
queue: asyncio.Queue = asyncio.Queue()
pending: list = []  # jobs acceptés non terminés (le premier = en cours)
job_ids = itertools.count(1)


@dataclass
class Job:
    id: int
    user_id: int
    channel: discord.abc.Messageable
    filename: str
    workdir: Path
    src: Path
    duration: float
    remove: str = ""
    status_msg: discord.Message | None = None


# ---------------------------------------------------------------- utilitaires
async def run(*cmd: str):
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out, err = await proc.communicate()
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


async def probe_duration(path: Path) -> float:
    _, out, _ = await run(
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(path),
    )
    try:
        return float(out.strip())
    except ValueError:
        return 0.0


def detect_bpm(path: Path) -> int | None:
    y, sr = librosa.load(str(path), sr=22050, mono=True, duration=150)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    if bpm <= 0:
        return None
    # corrige les erreurs de facteur 2 (70 au lieu de 140, etc.)
    while bpm < BPM_MIN:
        bpm *= 2
    while bpm > BPM_MAX:
        bpm /= 2
    return round(bpm)


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"[^\w\-. ()]", "_", stem)[:80] or "audio"


def is_audio(att: discord.Attachment) -> bool:
    if att.content_type and att.content_type.startswith("audio"):
        return True
    return Path(att.filename).suffix.lower() in AUDIO_EXTS


def position_text(job: Job) -> str:
    try:
        idx = pending.index(job)
    except ValueError:
        return "⏳ En file d'attente…"
    if idx == 0:
        return "⚙️ Traitement en cours…"
    return f"⏳ En file d'attente — position **{idx}**"


async def edit_status(job: Job, text: str):
    if job.status_msg is None:
        return
    try:
        await job.status_msg.edit(content=text, view=None)
    except discord.HTTPException:
        pass


async def refresh_positions():
    for job in list(pending)[1:]:
        await edit_status(job, position_text(job))


# ---------------------------------------------------------------- traitement
async def process(job: Job):
    label, removed, _ = OPTIONS[job.remove]
    await edit_status(job, "⚙️ Séparation des pistes en cours… (1 à 5 min)")

    # 1. conversion en wav 44.1 kHz stéréo
    wav = job.workdir / "input.wav"
    code, _, err = await run(
        "ffmpeg", "-y", "-i", str(job.src), "-vn", "-ac", "2", "-ar", "44100", str(wav)
    )
    if code != 0:
        raise RuntimeError(f"Conversion impossible : {err[-300:]}")

    # 2. Demucs (CPU)
    out_root = job.workdir / "sep"
    code, _, err = await run(
        sys.executable, "-m", "demucs",
        "-n", DEMUCS_MODEL, "-d", "cpu", "-j", "1",
        "--segment", DEMUCS_SEGMENT, "--overlap", "0.1",
        "-o", str(out_root), str(wav),
    )
    stems_dir = out_root / DEMUCS_MODEL / "input"
    if code != 0 or not stems_dir.exists():
        raise RuntimeError(f"Demucs a échoué : {err[-300:]}")

    # 3. mixage des pistes conservées + encodage MP3 adapté à la limite Discord
    await edit_status(job, "🎛️ Mixage et encodage…")
    keep = [stems_dir / f"{s}.wav" for s in ALL_STEMS if s not in removed]

    max_bytes = UPLOAD_LIMIT_MB * 1024 * 1024 * 0.96
    kbps = int(min(192, max_bytes * 8 / job.duration / 1000))
    if kbps < 64:
        raise RuntimeError(
            "Morceau trop long pour la limite d'upload du serveur "
            f"({UPLOAD_LIMIT_MB:g} Mo)."
        )

    out_name = f"{safe_stem(job.filename)}_sans_{job.remove}.mp3"
    out_path = job.workdir / out_name
    cmd = ["ffmpeg", "-y"]
    for f in keep:
        cmd += ["-i", str(f)]
    cmd += [
        "-filter_complex",
        f"amix=inputs={len(keep)}:duration=longest:normalize=0,alimiter=limit=0.95:level=0",
        "-c:a", "libmp3lame", "-b:a", f"{kbps}k", str(out_path),
    ]
    code, _, err = await run(*cmd)
    if code != 0 or not out_path.exists():
        raise RuntimeError(f"Encodage échoué : {err[-300:]}")

    # 4. BPM (sur la piste drums, plus fiable ; repli sur le morceau complet)
    bpm = None
    for candidate in (stems_dir / "drums.wav", wav):
        try:
            bpm = await asyncio.to_thread(detect_bpm, candidate)
            if bpm:
                break
        except Exception:
            log.exception("Détection BPM échouée sur %s", candidate)

    # 5. envoi
    text = (
        f"<@{job.user_id}> ✅ **{job.filename}**\n"
        f"🎚️ Retiré : **{label}**\n"
        f"🥁 BPM : **{bpm if bpm else 'non détecté'}**"
    )
    await job.channel.send(text, file=discord.File(out_path, filename=out_name))
    await edit_status(job, "✅ Terminé")


async def worker():
    while True:
        job = await queue.get()
        try:
            await process(job)
        except discord.HTTPException as e:
            log.exception("Envoi Discord échoué")
            await edit_status(job, f"❌ Envoi impossible (fichier trop gros ?) : {e.status}")
        except Exception as e:
            log.exception("Job %s échoué", job.id)
            await edit_status(job, f"❌ Erreur : {e}")
        finally:
            shutil.rmtree(job.workdir, ignore_errors=True)
            if job in pending:
                pending.remove(job)
            queue.task_done()
            await refresh_positions()


# ---------------------------------------------------------------- interface
class ChoiceView(discord.ui.View):
    def __init__(self, job: Job):
        super().__init__(timeout=300)
        self.job = job
        self.chosen = False
        self.message: discord.Message | None = None
        for key, (label, _, emoji) in OPTIONS.items():
            btn = discord.ui.Button(label=label, emoji=emoji, style=discord.ButtonStyle.primary)
            btn.callback = self._make_callback(key)
            self.add_item(btn)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.job.user_id:
            await interaction.response.send_message(
                "Ce fichier n'est pas à toi.", ephemeral=True
            )
            return False
        return True

    def _make_callback(self, key: str):
        async def callback(interaction: discord.Interaction):
            if self.chosen:
                await interaction.response.defer()
                return
            if len(pending) >= MAX_QUEUE:
                await interaction.response.send_message(
                    "La file d'attente est pleine, réessaie dans quelques minutes.",
                    ephemeral=True,
                )
                return
            self.chosen = True
            self.job.remove = key
            self.job.status_msg = interaction.message
            pending.append(self.job)
            queue.put_nowait(self.job)
            self.stop()
            await interaction.response.edit_message(content=position_text(self.job), view=None)

        return callback

    async def on_timeout(self):
        if self.chosen:
            return
        shutil.rmtree(self.job.workdir, ignore_errors=True)
        if self.message:
            try:
                await self.message.edit(
                    content="⌛ Temps écoulé, renvoie le fichier pour recommencer.", view=None
                )
            except discord.HTTPException:
                pass


intents = discord.Intents.default()
intents.message_content = True  # à activer aussi dans le Developer Portal


class StemBot(discord.Client):
    async def setup_hook(self):
        self.worker_task = asyncio.create_task(worker())


client = StemBot(intents=intents)


@client.event
async def on_ready():
    log.info("Connecté en tant que %s", client.user)


@client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    if CHANNEL_ID and message.channel.id != CHANNEL_ID:
        return
    att = next((a for a in message.attachments if is_audio(a)), None)
    if att is None:
        return
    if att.size > MAX_INPUT_MB * 1024 * 1024:
        await message.reply(f"❌ Fichier trop gros (max {MAX_INPUT_MB:g} Mo).", mention_author=False)
        return

    workdir = Path(tempfile.mkdtemp(prefix="stem_"))
    try:
        src = workdir / ("source" + Path(att.filename).suffix.lower())
        await att.save(src)
        duration = await probe_duration(src)
    except Exception:
        log.exception("Téléchargement échoué")
        shutil.rmtree(workdir, ignore_errors=True)
        await message.reply("❌ Impossible de lire ce fichier.", mention_author=False)
        return

    if duration <= 0:
        shutil.rmtree(workdir, ignore_errors=True)
        await message.reply("❌ Fichier audio illisible.", mention_author=False)
        return
    if duration > MAX_DURATION:
        shutil.rmtree(workdir, ignore_errors=True)
        await message.reply(
            f"❌ Morceau trop long (max {MAX_DURATION // 60} min).", mention_author=False
        )
        return

    job = Job(
        id=next(job_ids),
        user_id=message.author.id,
        channel=message.channel,
        filename=att.filename,
        workdir=workdir,
        src=src,
        duration=duration,
    )
    view = ChoiceView(job)
    view.message = await message.reply(
        f"🎚️ **{att.filename}** — Que veux-tu enlever ?", view=view, mention_author=False
    )


if __name__ == "__main__":
    client.run(TOKEN, log_handler=None)
