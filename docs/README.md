# Sbobinator — Documentazione Architetturale

> **Scopo**: Questo documento è la fonte di verità per il progetto Sbobinator.
> Guida gli agenti AI e gli sviluppatori nella comprensione, navigazione e
> modifica del codice. Se qualcosa non è chiaro, consulta questo file prima
> di cercare nel codice.

---

## Indice

1. [Panoramica del Progetto](#1-panoramica-del-progetto)
2. [Struttura dei File](#2-struttura-dei-file)
3. [Architettura ad Alto Livello](#3-architettura-ad-alto-livello)
4. [Modulo API (`backend/api`)](#4-modulo-api-backendapi)
5. [Modulo Storage (`backend/modules/storage`)](#5-modulo-storage-backendmodulesstorage)
6. [Modulo Preprocessing (`backend/modules/preprocessing`)](#6-modulo-preprocessing-backendmodulespreprocessing)
7. [Modulo Transcription (`backend/modules/transcription`)](#7-modulo-transcription-backendmodulestranscription)
8. [Modulo Output (`backend/modules/output`)](#8-modulo-output-backendmodulesoutput)
9. [Pipeline di Trascrizione (`backend/transcription_pipeline.py`)](#9-pipeline-di-trascrizione-backendtranscription_pypelinepy)
10. [Configurazione (`utils/config.py`)](#10-configurazione-utilsconfigpy)
11. [Entry Point (`main.py`)](#11-entry-point-mainpy)
12. [Flusso dei Dati Completo](#12-flusso-dei-dati-completo)
13. [Variabili d'Ambiente](#13-variabili-dambiente)
14. [Pattern e Convenzioni](#14-pattern-e-convenzioni)
15. [Riferimenti Rapidi](#15-riferimenti-rapidi)

---

## 1. Panoramica del Progetto

**Sbobinator** è un servizio REST per la trascrizione automatica di file audio
basato su [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

### Cosa fa

1. **Accetta** file audio via API REST (upload multipart)
2. **Salva** il file in una cartella temporanea
3. **Preprocessa** l'audio (decoding, mono, resampling, noise reduction, trimming)
4. **Trascrive** con Whisper (batched o custom pipeline)
5. **Formatta** l'output (SRT, VTT, TXT, CSV, TSV, JSON)
6. **Salva** la trascrizione in un file temporaneo
7. **Espone** l'output al chiamante via API
8. **Cancella** i file temporanei dopo ACK del chiamante

### Tecnologie Principali

| Tecnologie | Ruolo |
|---|---|
| **FastAPI** | Framework web, API REST |
| **Uvicorn** | ASGI server |
| **faster-whisper** | Motore di trascrizione |
| **boto3** | Client S3/MinIO |
| **numpy / scipy** | Elaborazione numerica audio |
| **ffmpeg-python** | Decoding formati audio |
| **soundfile** | Decoding formati nativi |
| **noisereduce** | Riduzione del rumore |
| **pydantic-settings** | Configurazione |

### Requisiti

- Python ≥ 3.14 (vedi `.python-version`)
- GPU CUDA (per inference su GPU)
- FFmpeg (per decoding mp3, m4a, webm)

---

## 2. Struttura dei File

```
sbobinator/
├── main.py                          # Entry point — crea e avvia l'app FastAPI
├── pyproject.toml                   # Dipendenze e metadata del progetto
├── uv.lock                          # Lock file per uv
├── .env.local                       # Configurazione环境变量 (non committare)
├── .gitignore
│
├── backend/
│   ├── api/
│   │   ├── __init__.py              # Esporta `router`
│   │   └── routes.py                # Endpoint REST (upload, convert, status, output)
│   │
│   ├── modules/
│   │   ├── __init__.py
│   │   │
│   │   ├── storage/
│   │   │   ├── __init__.py          # Esporta `AudioFetcher`
│   │   │   └── fetcher.py           # Interazione S3/MinIO (fetch + upload)
│   │   │
│   │   ├── preprocessing/
│   │   │   ├── __init__.py          # Esporta `AudioProcessor`, `ProcessedAudio`
│   │   │   └── audio_processor.py   # Decoding, mono, resample, noise reduction
│   │   │
│   │   ├── transcription/
│   │   │   ├── __init__.py          # Esporta `TranscriptionEngine`, ecc.
│   │   │   ├── whisper_engine.py    # Factory pipeline (batched vs custom)
│   │   │   ├── batched_pipeline.py  # BatchedInferencePipeline
│   │   │   └── custom_pipeline.py   # Silence-based chunking + parallel
│   │   │
│   │   ├── output/
│   │   │   ├── __init__.py          # Esporta `OutputFormatter`
│   │   │   └── formatter.py         # Genera SRT, VTT, TXT, CSV, TSV, JSON
│   │   │
│   │   └── notifications/
│   │       └── __init__.py          # Placeholder per callback frontend
│   │
│   └── transcription_pipeline.py    # Facade principale — orchestratore
│
└── utils/
    └── config.py                    # Configurazione centralizzata (Settings)
```

### Regole di Navigazione

| Cosa vuoi fare? | Vai a... |
|---|---|
| Capire come funziona l'API | `backend/api/routes.py` |
| Modificare i parametri di Whisper | `utils/config.py` → `WhisperInferenceConfig`, `WhisperConstraintsConfig`, ecc. |
| Cambiare il formato di output | `backend/modules/output/formatter.py` |
| Modificare il preprocessing audio | `backend/modules/preprocessing/audio_processor.py` |
| Cambiare storage (S3/MinIO) | `backend/modules/storage/fetcher.py` |
| Capire l'orchestrazione completa | `backend/transcription_pipeline.py` |
| Modificare la configurazione | `.env.local` e `utils/config.py` |

---

## 3. Architettura ad Alto Livello

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI (main.py)                        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              CORS Middleware + Logging                     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────▼─────────────────────────────────┐│
│  │                    API Routes                                ││
│  │  POST /api/audio/upload      → Upload file, avvia task      ││
│  │  POST /api/audio/convert/{id}/format → Set output format    ││
│  │  GET  /api/audio/convert/{id} → Poll status                 ││
│  │  GET  /api/audio/convert/{id}/output → Download output      ││
│  │  POST /api/audio/convert/{id}/ack → Ack + delete temp files ││
│  │  DELETE /api/audio/convert/{id}  → Cancella task            ││
│  │  GET  /api/audio/convert         → Lista tutti i task       ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│           TranscriptionPipeline (Facade Singleton)               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ Audio    │  │ Audio    │  │ Whisper  │  │ Output       │   │
│  │Processor │→ │Processor │→ │ Engine   │→ │ Formatter    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Principi Architetturali

1. **Facade Pattern**: `TranscriptionPipeline` è il punto unico di ingresso
   per qualsiasi operazione di trascrizione.
2. **Singleton**: `TranscriptionPipeline` e `Settings` sono singleton.
3. **Lazy Initialization**: I moduli (AudioFetcher, AudioProcessor, ecc.)
   sono inizializzati al primo utilizzo, non all'avvio.
4. **Async Background Tasks**: La trascrizione avviene in background
   (FastAPI `BackgroundTasks`), l'API risponde immediatamente con un `task_id`.
5. **Temp-File Based**: I file temporanei (input e output) risiedono nel filesystem locale.
6. **ACK-Based Cleanup**: I file vengono cancellati solo dopo ACK esplicito del chiamante.
7. **Configurabilità Totale**: Ogni parametro è configurabile via `.env.local`.

---

## 4. Modulo API (`backend/api`)

### File: `backend/api/routes.py`

**Route esposte:**

| Metodo | Path | Descrizione |
|---|---|---|
| `POST` | `/api/audio/convert` | Upload file + avvia trascrizione (async) |
| `POST` | `/api/audio/convert/{task_id}/format` | Imposta formato output |
| `GET` | `/api/audio/convert/{task_id}` | Poll status del task |
| `GET` | `/api/audio/convert/{task_id}/output` | Download output trascrizione |
| `POST` | `/api/audio/convert/{task_id}/ack` | Ack + cancella file temporanei |
| `DELETE` | `/api/audio/convert/{task_id}` | Cancella un task in corso |
| `GET` | `/api/audio/convert` | Lista tutti i task |

### Modelli di Richiesta/Risposta

```python
# POST /api/audio/convert — risposta
class ConvertTaskResponse(BaseModel):
    task_id: str
    status: str                  # "processing"
    filename: str
    content_type: str
    size_bytes: int

# POST /api/audio/convert/{task_id}/format — corpo della richiesta
class ConvertFormatRequest(BaseModel):
    output_format: str = "txt"   # Formato output (json, srt, vtt, txt, csv, tsv)

# POST /api/audio/convert/{task_id}/ack — risposta
class AckResponse(BaseModel):
    acknowledged: bool
    task_id: str

# GET /api/audio/convert/{task_id} — risposta
class TaskResponse(BaseModel):
    task_id: str
    status: str                  # "processing" | "completed" | "failed" | "cancelled"
    progress: float              # 0.0 → 1.0
    started_at: str | None       # ISO timestamp
    completed_at: str | None     # ISO timestamp
    filename: str | None         # Nome file originale
    content_type: str | None     # MIME type
    duration: float | None       # Durata audio in secondi
    sample_rate: int | None      # Sample rate processato
    original_format: str | None  # Formato originale (mp3, wav, ecc.)
    original_sample_rate: int | None
    language: str | None         # Lingua rilevata
    segment_count: int | None    # Numero di segmenti
    processing_time_ms: float | None
    error: str | None            # Messaggio di errore (se fallito)
    acknowledged: bool           # True se output già consumato
```

### Come Funziona `POST /api/audio/convert`

```
1. Riceve file audio (multipart/form-data)
2. Chama _pipeline.upload_audio(file_bytes, content_type, filename)
   → Crea un tempfile per l'audio sul disco
   → Crea un task con ID univoco
   → Avvia process_task() in background thread
   → Restituisce task_id immediatamente
3. Restituisce ConvertTaskResponse con status="processing"
```

### Come Funziona `POST /api/audio/convert/{task_id}/ack`

```
1. Controlla che il task esista e sia COMPLETED
2. Controlla che non sia già stato ACKed
3. Cancella tutti i tempfile associati al task
4. Restituisce AckResponse con acknowledged=true
5. Richieste successive a /output restituiranno 404
```

### File: `backend/api/__init__.py`

Esporta solo `router`. Usato da `main.py` per includere le route:

```python
from backend.api.routes import router
```

---

## 5. Modulo Storage (`backend/modules/storage`)

> **Nota**: Il modulo storage (S3/MinIO) è mantenuto per backward compatibility.
> Il flusso principale di upload/trascrizione ora utilizza file temporanei
> sul filesystem locale invece di S3.

### File: `backend/modules/storage/fetcher.py`

**Classi Principali:**

#### `S3StorageBackend`
Interfaccia diretta con S3/MinIO tramite `boto3`.

```python
backend = S3StorageBackend(config=settings().storage)
result = backend.fetch(key="audio/uploads/file.mp3", bucket="sbobinator")
backend.upload(key="transcripts/file.txt", data=b"hello", content_type="text/plain")
```

**Metodi:**
- `fetch(key, bucket)` → `AudioFetchResult` (download)
- `upload(key, data, content_type, bucket)` → None (upload)

**Eccezioni:**
- `FetchError` — Errore di fetch/upload (oggetto non trovato, permessi, ecc.)

#### `AudioFetcher`
Wrapper di alto livello che incapsula `S3StorageBackend`.

```python
fetcher = AudioFetcher()
result = fetcher.fetch(key="audio/uploads/file.mp3")
fetcher.upload(key="transcripts/output.txt", data=b"content", content_type="text/plain")
```

**Attributi:**
- `backend: S3StorageBackend` — Backend S3 sottostante
- `default_bucket: str` — Bucket di default da config

### File: `backend/modules/storage/__init__.py`

```python
from .fetcher import AudioFetcher
__all__ = ["AudioFetcher"]
```

---

## 6. Modulo Preprocessing (`backend/modules/preprocessing`)

### File: `backend/modules/preprocessing/audio_processor.py`

**Scopo**: Trasforma raw audio bytes in un array numpy float32, mono, 16 kHz,
pronto per Whisper.

### Pipeline di Preprocessing (in ordine)

```
1. _guess_format(content_type, filename)     → Determina il formato (mp3, wav, ecc.)
2. load_audio(data, fmt)                     → Decoding (soundfile o ffmpeg)
3. convert_to_mono(data)                     → Downmix a mono
4. resample(data, src_sr, dst_sr)            → Resampling a 16 kHz
5. remove_noise(data, sr)                    → Noise reduction (opzionale)
6. trim_audio(data, sr, max_duration)        → Trimming (opzionale)
7. remove_silence(data, sr)                  → Rimozione silence iniziale/finalle
8. normalize_peak(data)                      → Peak normalization (0.97)
```

### Classi Principali

#### `AudioProcessor`

```python
processor = AudioProcessor()
result = processor.process(
    raw_bytes=audio_data,
    content_type="audio/mpeg",
    filename="podcast.mp3"
)
# result: ProcessedAudio(data=np.ndarray, sample_rate=16000, duration=123.45, ...)
```

#### `ProcessedAudio` (dataclass frozen)

```python
@dataclass(frozen=True)
class ProcessedAudio:
    data: np.ndarray                  # float32, mono, [-1.0, 1.0]
    sample_rate: int                  # 16000
    duration: float                   # secondi
    original_format: str              # "mp3", "wav", ecc.
    original_sample_rate: int         # Sample rate originale
```

### Eccezioni

- `AudioProcessingError` — Qualsiasi errore nella pipeline

### Funzioni Helper

| Funzione | Scopo |
|---|---|
| `_guess_format(content_type, filename)` | Determina formato da MIME o estensione |
| `_to_mono(data)` | Downmix multi-channel → mono |
| `_estimate_noise_profile(data, sr, frame_length)` | Stima profilo rumore (primi N secondi) |
| `load_audio(data, fmt)` | Decoding (soundfile per wav/flac/ogg, ffmpeg per mp3/m4a) |
| `convert_to_mono(data)` | Conversione a mono |
| `resample(data, src_sr, dst_sr)` | Resampling via scipy |
| `remove_noise(data, sr, frame_length)` | Noise reduction via noisereduce |
| `trim_audio(data, sr, max_duration)` | Trimming a durata massima |
| `remove_silence(data, sr)` | Rimozione silence iniziale/finalle (-45 dBFS) |
| `normalize_peak(data)` | Peak normalization a 0.97 |

### Formati Supportati

| Formato | Metodo | Note |
|---|---|---|
| wav | soundfile | Nativo |
| flac | soundfile | Nativo |
| ogg | soundfile | Nativo |
| mp3 | ffmpeg | Fallback |
| m4a | ffmpeg | Fallback |
| mp4 | ffmpeg | Fallback |
| webm | ffmpeg | Fallback |

---

## 7. Modulo Transcription (`backend/modules/transcription`)

### File: `backend/modules/transcription/whisper_engine.py`

**Scopo**: Factory che seleziona la pipeline di trascrizione in base alla config.

```python
engine = TranscriptionEngine()
result = engine.transcribe(
    audio_data=processed.data,
    sr=processed.sample_rate,
    audio_duration=processed.duration
)
```

**Selezione Pipeline:**
- `settings().whisper_inference.bip == True` → `BatchedPipeline`
- `settings().whisper_inference.bip == False` → `CustomPipeline`

### File: `backend/modules/transcription/batched_pipeline.py`

**BatchedPipeline** — Usa `BatchedInferencePipeline` di faster-whisper.

- Passa l'audio completo alla pipeline
- La pipeline gestisce chunking interno via Silero VAD
- Batching per decoding parallelo su GPU
- Configurabile via `batch_size` e `chunk_length`

```python
pipeline = BatchedPipeline()
result = pipeline.transcribe(audio_data, sr=16000)
pipeline.close()
```

### File: `backend/modules/transcription/custom_pipeline.py`

**CustomPipeline** — Silence-based chunking + parallel processing.

**Pipeline:**
1. `_find_speech_regions(data, sr, min_silence_ms, speech_pad_ms)`
   → Trova regioni di speech separate da silence
2. `_create_chunks(data, sr, regions)`
   → Crea `AudioChunk` con offset e durata
3. `_transcribe_chunk(chunk, **kwargs)`
   → Trascrive un chunk con `WhisperModel.transcribe()` standard
4. `ThreadPoolExecutor` per parallelizzazione
5. Merge e ordinamento dei segmenti

```python
pipeline = CustomPipeline()
result = pipeline.transcribe(audio_data, sr=16000)
```

### Classi Condivise

#### `TranscriptionSegment` (dataclass frozen)

```python
@dataclass(frozen=True)
class TranscriptionSegment:
    start: float      # Tempo di inizio (secondi)
    end: float        # Tempo di fine (secondi)
    text: str         # Testo trascritto
    language: str | None  # Lingua (se rilevata)
```

#### `TranscriptionResult` (dataclass frozen)

```python
@dataclass(frozen=True)
class TranscriptionResult:
    segments: list[TranscriptionSegment]  # Lista di segmenti
    text: str                              # Testo completo
    duration: float                        # Durata audio
    language: str                          # Lingua principale
```

#### `AudioChunk` (dataclass frozen) — solo CustomPipeline

```python
@dataclass(frozen=True)
class AudioChunk:
    chunk_id: int        # ID incrementale
    data: np.ndarray     # Samples del chunk
    start_offset: float  # Offset dall'inizio dell'audio
    end_offset: float    # Fine dall'inizio dell'audio
    duration: float      # Durata del chunk
```

### Eccezioni

- `TranscriptionError` — Qualsiasi errore nella trascrizione

---

## 8. Modulo Output (`backend/modules/output`)

### File: `backend/modules/output/formatter.py`

**Scopo**: Genera file di output in vari formati dai segmenti di trascrizione.

### Formati Supportati

| Formato | Estensione | MIME Type | Descrizione |
|---|---|---|---|
| `json` | `.json` | `application/json` | JSON con segments + text |
| `srt` | `.srt` | `text/srt` | SRT subtitle file |
| `vtt` | `.vtt` | `text/vtt` | WebVTT subtitle file |
| `txt` | `.txt` | `text/plain` | Testo puro (tutti i segmenti) |
| `csv` | `.csv` | `text/csv` | CSV con header (start, end, text) |
| `tsv` | `.tsv` | `text/tab-separated-values` | TSV con header (start, end, text) |

### Classi e Funzioni

#### `OutputFormatter`

```python
formatter = OutputFormatter()

# Formattazione
content = formatter.format(
    segments=result.segments,
    fmt="srt",
    metadata={"language": "it", "duration": 120.0}  # opzionale, per JSON
)

# Utility
ext = formatter.get_extension("srt")        # ".srt"
mime = formatter.get_mime_type("json")      # "application/json"
key = formatter.build_output_key("audio/file.mp3", "srt")  # "audio/file.srt"
```

### Formattatori Interni

| Funzione | Formato |
|---|---|
| `_format_srt(segments)` | SRT (HH:MM:SS,mmm) |
| `_format_vtt(segments)` | WebVTT (HH:MM:SS.mmm) |
| `_format_txt(segments)` | Testo puro |
| `_format_csv(segments, include_header)` | CSV |
| `_format_tsv(segments, include_header)` | TSV |
| `_format_json(segments, metadata)` | JSON |

### Timestamp Helpers

| Funzione | Formato |
|---|---|
| `_format_timestamp_srt(seconds)` | `HH:MM:SS,mmm` |
| `_format_timestamp_vtt(seconds)` | `HH:MM:SS.mmm` |

---

## 9. Pipeline di Trascrizione (`backend/transcription_pipeline.py`)

### File: `backend/transcription_pipeline.py`

**Scopo**: Facade principale che orchestra l'intera pipeline.

### Pattern Singleton

```python
pipeline = TranscriptionPipeline.get_instance()  # Thread-safe
```

### Task Lifecycle

```
PROCESSING → COMPLETED
        ↘ FAILED
        ↘ CANCELLED
```

> **Nota**: Non esiste più lo stato PENDING. Il task parte subito in
> stato PROCESSING all'upload.

### Classi

#### `TaskStatus` (Enum)

```python
class TaskStatus(str, Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

#### `TaskInfo` (dataclass frozen) — Snapshot immutabile

```python
@dataclass(frozen=True)
class TaskInfo:
    task_id: str
    status: str
    progress: float
    started_at: datetime | None
    completed_at: datetime | None
    filename: str | None           # Nome file originale
    content_type: str | None       # MIME type
    duration: float | None
    sample_rate: int | None
    original_format: str | None
    original_sample_rate: int | None
    language: str | None
    segment_count: int | None
    processing_time_ms: float | None
    error: str | None
    acknowledged: bool             # True se output già consumato
```

#### `_ConversionTask` (classe interna mutable)

Stato mutabile del task, usato internamente.

**Attributi aggiuntivi:**
- `temp_files: list[Path]` — Lista dei file temporanei (audio + output)
- `acknowledged: bool` — True se l'output è stato ACKed

#### `UploadTaskResult` (dataclass frozen)

```python
@dataclass(frozen=True)
class UploadTaskResult:
    task_id: str
    status: str
    filename: str
    content_type: str
    size_bytes: int
```

### Metodi Pubblici

| Metodo | Descrizione |
|---|---|
| `upload_audio(file_bytes, content_type, filename)` | Salva audio in tempfile, avvia task, restituisce task_id |
| `convert_task(task_id, output_format)` | Imposta formato output per un task |
| `process_task(task_id)` | Esegue la pipeline completa (chiamato in background) |
| `get_task_status(task_id)` → `TaskInfo` | Ottiene stato del task |
| `get_task_output(task_id)` → `str` | Ottiene il contenuto dell'output (da tempfile) |
| `acknowledge_task(task_id)` | ACK output, cancella tempfile |
| `cancel_task(task_id)` → `TaskInfo` | Cancella un task e i suoi tempfile |
| `list_tasks()` → `list[TaskInfo]` | Lista tutti i task |

### Pipeline Interna (`process_task`)

```
1. Leggi audio dal tempfile             → progress: 0.2
2. Preprocess audio                     → progress: 0.4
3. Transcribe con Whisper               → progress: 0.7
4. Build output (raw o formatted)       → progress: 0.9
5. Salva output in secondo tempfile
6. Finalize task (status=COMPLETED)     → progress: 1.0
```

### Gestione File Temporanei

```
upload_audio() → Crea tempfile_audio
process_task() → Crea tempfile_output
get_task_output() → Legge tempfile_output (NON cancella)
acknowledge_task() → Cancella tempfile_audio + tempfile_output
```

### Error Handling

La pipeline cattura e gestisce:
- `AudioProcessingError` → status=FAILED, error="Processing error: ..."
- `TranscriptionError` → status=FAILED, error="Transcription error: ..."
- `ValueError` → status=FAILED, error="Format error: ..."
- `Exception` (generico) → status=FAILED, error="Unexpected error: ..."

---

## 10. Configurazione (`utils/config.py`)

### File: `utils/config.py`

**Scopo**: Configurazione centralizzata tramite pydantic-settings.
Tutti i valori sono caricati da `.env.local`.

### Convenzione Nomi

```
Flat fields:     SBO_<FIELD_NAME>
Nested fields:   SBO_<FIELD_NAME>__<SUB_FIELD>

Esempio:
  SBO_WHISPER_INFERENCE__BEAM_SIZE → settings.whisper_inference.beam_size
  SBO_AUDIO__SAMPLE_RATE           → settings.audio.sample_rate
```

### Gerarchia delle Classi di Configurazione

```
Settings (main)
├── host, port, debug, cors
├── whisper_model_size, whisper_model_path, whisper_language, whisper_task
├── whisper_device, whisper_device_index, whisper_compute_type
├── whisper_inference: WhisperInferenceConfig
│   ├── bip, beam_size, patience, temperature
│   ├── num_workers, batch_size
├── whisper_constraints: WhisperConstraintsConfig
│   ├── repetition_penalty, length_ratio
│   ├── spm_silence_threshold, wer_silence_threshold
│   ├── hotwords, ban_token_ids, allow_punctuation_tokens
├── whisper_vad: WhisperVadConfig
│   ├── enabled, method, threshold
│   ├── min_silence_duration_ms, speech_pad_ms
├── whisper_timestamp: WhisperTimestampConfig
│   ├── max_initial_timestamp, one_segment
│   ├── segment_duration
├── audio: AudioConfig
│   ├── sample_rate, chunk_duration, max_duration
│   ├── noise_reduction, noise_reduction_sample_rate
│   ├── noise_reduction_frame_length
├── temp: TempConfig
│   ├── dir, cleanup_on_ack
├── output: OutputConfig
│   ├── format, use_formatter, include_metadata
│   ├── save_transcripts, transcripts_dir
├── logging: LoggingConfig
│   ├── level, format
└── storage: S3StorageConfig
    ├── endpoint_url, access_key_id, secret_access_key
    ├── bucket, region, secure, max_retries, timeout
    ├── max_audio_size
```

### Accessi alla Config

```python
from utils.config import settings

cfg = settings()  # Singleton

# Accesso ai valori
cfg.host              # "0.0.0.0"
cfg.port              # 8000
cfg.whisper_model_size  # "large-v3"
cfg.whisper_device    # "cuda"
cfg.audio.sample_rate # 16000
cfg.storage.bucket    # "sbobinator"

# Property utili
cfg.vad_params        # dict per VAD
cfg.whisper_kwargs    # dict per Whisper.load_model()
cfg.run_kwargs        # dict per model.transcribe()
cfg.storage_endpoint  # URL completo con scheme
```

### Property di Convenienza

| Property | Descrizione |
|---|---|
| `cfg.vad_params` | Dict VAD compatibile con faster-whisper |
| `cfg.whisper_kwargs` | Kwargs per `WhisperModel()` |
| `cfg.run_kwargs` | Kwargs per `model.transcribe()` |
| `cfg.storage_endpoint` | URL storage con scheme |
| `cfg.summary()` | Stringa formattata con riepilogo config |

---

## 11. Entry Point (`main.py`)

### File: `main.py`

**Scopo**: Crea e configura l'applicazione FastAPI.

### Funzione `create_app()`

```python
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

def create_app() -> FastAPI:
    cfg = settings()

    app = FastAPI(
        title="Sbobinator",
        description="AI-powered audio transcription service",
        version="0.1.0",
        debug=cfg.debug,
        lifespan=lifespan,  # Gestisce startup e shutdown
    )

    # CORS middleware (se origins configurate)
    if cfg.cors.origins:
        app.add_middleware(CORSMiddleware, ...)

    # Logging
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    # Include routes
    app.include_router(router)

    return app


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Gestisce startup e shutdown dell'applicazione."""
    logger.info(settings().summary())
    yield
```

### Avvio

```bash
# Opzione 1: python main.py
python main.py

# Opzione 2: uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Opzione 3: tramite pyproject.toml (se configurato)
# uv run python main.py
```

---

## 12. Flusso dei Dati Completo

### Scenario: Upload → Convert → Output → Ack

```
┌──────────┐     POST /api/audio/upload      ┌─────────────┐
│  Client  │ ──────────────────────────────→ │  FastAPI    │
│          │                                 │  (routes.py)│
└──────────┘                                 └──────┬──────┘
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TranscriptionPipeline                        │
│                                                                 │
│  1. upload_audio(file_bytes, content_type, filename)            │
│     → Crea tempfile audio sul disco                             │
│     → Crea task con status=PROCESSING                           │
│     → Avvia process_task() in background thread                 │
│     → Restituisce task_id immediatamente                        │
│                                                                 │
│  2. process_task(task_id) [background thread]                   │
│     ┌─────────────────────────────────────────────────────┐     │
│     │ a. Leggi audio dal tempfile                           │     │
│     │    AudioProcessor.process(raw_bytes, content_type)   │     │
│     │    → ProcessedAudio(data, sr, duration, ...)         │     │
│     │                                                      │     │
│     │ b. Transcribe                                        │     │
│     │    TranscriptionEngine.transcribe(audio_data, sr)    │     │
│     │    → TranscriptionResult(segments, text, ...)        │     │
│     │                                                      │     │
│     │ c. Build output                                      │     │
│     │    Se use_formatter=True:                            │     │
│     │      OutputFormatter.format(segments, fmt)           │     │
│     │    Altrimenti:                                       │     │
│     │      result.text (raw)                               │     │
│     │                                                      │     │
│     │ d. Salva output in tempfile sul disco                │     │
│     └─────────────────────────────────────────────────────┘     │
│                                                                 │
│  3. get_task_status(task_id) → TaskInfo                         │
│                                                                 │
│  4. get_task_output(task_id) → str                              │
│     → Legge output dal tempfile (NON cancella)                 │
│                                                                 │
│  5. acknowledge_task(task_id)                                   │
│     → Cancella tutti i tempfile del task                       │
│     → Richieste successive a /output → 404                     │
└─────────────────────────────────────────────────────────────────┘
```

### Diagramma del Ciclo di Vita

```
POST /upload
    │
    ├──▶ [tempfile audio creato]
    ├──▶ [task_id restituito, status=processing]
    ├──▶ [background: process_task inizia]
    │
    │    GET /convert/{id}  → status=processing
    │    GET /convert/{id}  → status=completed
    │
    ├──▶ [tempfile output creato]
    │
    GET /convert/{id}/output
    │       │
    │       ├──▶ [output restituito, file NON cancellati]
    │       │
    │       POST /convert/{id}/ack
    │               │
    │               ├──▶ [tutti i tempfile cancellati]
    │               │
    │               └──▶ [output restituito → 404]
```

### Gestione File Temporanei

| Fase | File Audio | File Output |
|---|---|---|
| Upload | Creato | — |
| Processing | Letto | — |
| Completato | Letto | Creato |
| Output scaricato | Letto | Letto |
| ACK | **Cancellato** | **Cancellato** |

---

## 13. Variabili d'Ambiente

### File: `.env.local`

**Nota**: Questo file è in `.gitignore` e non deve essere commitatato.
Copiare da `.env.example` se disponibile.

### Riepilogo Variabili

```bash
# FastAPI Service
SBO_HOST=0.0.0.0
SBO_PORT=8000
SBO_DEBUG=False
SBO_CORS__ORIGINS=["http://localhost", "http://localhost:3000"]

# Whisper Model
SBO_WHISPER_MODEL_SIZE=large-v3
SBO_WHISPER_MODEL_PATH=
SBO_WHISPER_LANGUAGE=it
SBO_WHISPER_TASK=transcribe

# Whisper GPU
SBO_WHISPER_DEVICE=cuda
SBO_WHISPER_DEVICE_INDEX=0
SBO_WHISPER_COMPUTE_TYPE=int8

# Whisper Inference
SBO_WHISPER_INFERENCE__BEAM_SIZE=5
SBO_WHISPER_INFERENCE__PATIENCE=1.0
SBO_WHISPER_INFERENCE__TEMPERATURE=0.0
SBO_WHISPER_INFERENCE__BIP=True          # BatchedInferencePipeline
SBO_WHISPER_INFERENCE__BATCH_SIZE=8       # Solo se BIP=True
SBO_WHISPER_INFERENCE__NUM_WORKERS=3      # Solo se BIP=False

# Whisper Constraints
SBO_WHISPER_CONSTRAINTS__REPETITION_PENALTY=1.0
SBO_WHISPER_CONSTRAINTS__LENGTH_RATIO=2.5
SBO_WHISPER_CONSTRAINTS__HOTWORDS=[]
SBO_WHISPER_CONSTRAINTS__BAN_TOKEN_IDS=[]

# Whisper VAD
SBO_WHISPER_VAD__ENABLED=True
SBO_WHISPER_VAD__METHOD=silero
SBO_WHISPER_VAD__THRESHOLD=0.5
SBO_WHISPER_VAD__MIN_SILENCE_DURATION_MS=500
SBO_WHISPER_VAD__SPEECH_PAD_MS=250

# Whisper Timestamp
SBO_WHISPER_TIMESTAMP__MAX_INITIAL_TIMESTAMP=1
SBO_WHISPER_TIMESTAMP__ONE_SEGMENT=False
SBO_WHISPER_TIMESTAMP__SEGMENT_DURATION=30

# Audio Preprocessing
SBO_AUDIO__SAMPLE_RATE=16000
SBO_AUDIO__CHUNK_DURATION=30
SBO_AUDIO__MAX_DURATION=300
SBO_AUDIO__NOISE_REDUCTION=False
SBO_AUDIO__NOISE_REDUCTION_FRAME_LENGTH=0.5

# Output
SBO_OUTPUT__FORMAT=json
SBO_OUTPUT__USE_FORMATTER=False      # False = raw text, True = formatted
SBO_OUTPUT__SAVE_TRANSCRIPTS=True
SBO_OUTPUT__TRANSCRIPTS_DIR=./transcripts

# Temporary Files
SBO_TEMP__DIR=/tmp/sbobinator
SBO_TEMP__CLEANUP_ON_ACK=True

# Logging
SBO_LOGGING__LEVEL=INFO
SBO_LOGGING__FORMAT=json

# S3/MinIO Storage (kept for backward compatibility)
SBO_STORAGE__ENDPOINT_URL=http://localhost:9000
SBO_STORAGE__ACCESS_KEY_ID=minioadmin
SBO_STORAGE__SECRET_ACCESS_KEY=minioadmin
SBO_STORAGE__BUCKET=sbobinator
SBO_STORAGE__REGION=us-east-1
SBO_STORAGE__SECURE=False
SBO_STORAGE__MAX_AUDIO_SIZE=2147483648
```

---

## 14. Pattern e Convenzioni

### Naming

- **Classi**: PascalCase (`AudioProcessor`, `TranscriptionEngine`)
- **Funzioni**: snake_case (`load_audio`, `remove_noise`)
- **Variabili**: snake_case (`task_id`, `output_format`)
- **Costanti**: UPPER_SNAKE_CASE (`SILENCE_THRESHOLD_DBFS`)
- **Private**: underscore prefix (`_guess_format`, `_tasks_lock`)

### Eccezioni

Ogni modulo definisce le proprie eccezioni specifiche:

| Modulo | Eccezione |
|---|---|
| `storage` | `FetchError` |
| `preprocessing` | `AudioProcessingError` |
| `transcription` | `TranscriptionError` |

### Type Hints

Tutti i file usano `from __future__ import annotations` per type hints
compatibili con Python 3.10+.

### Docstrings

Tutti i file, classi e funzioni hanno docstrings con:
- Descrizione
- Parametri (con tipi e descrizioni)
- Ritorni
- Eccezioni

### Logging

- `logger = logging.getLogger(__name__)` in ogni file
- Livelli: `logger.info()`, `logger.warning()`, `logger.error()`
- Format: `%(asctime)s %(levelname)s %(name)s: %(message)s`

### Thread Safety

- `TranscriptionPipeline._tasks_lock` per accesso al task store
- `TranscriptionPipeline._instance_lock` per singleton
- `threading.Event` per cancellazione task

### Lazy Initialization

I moduli sono inizializzati al primo utilizzo, non all'avvio:

```python
@property
def _fetcher(self) -> AudioFetcher:
    if self._audio_fetcher is None:
        self._audio_fetcher = AudioFetcher()
    return self._audio_fetcher
```

---

## 15. Riferimenti Rapidi

### Dove Trovare...

| Cosa | File |
|---|---|
| Endpoint REST | `backend/api/routes.py` |
| Configurazione | `utils/config.py` |
| Upload audio | `backend/api/routes.py::upload_audio()` |
| ACK output | `backend/api/routes.py::acknowledge_output()` |
| Pipeline completa | `backend/transcription_pipeline.py::process_task()` |
| Acknowledge | `backend/transcription_pipeline.py::acknowledge_task()` |
| Preprocessing audio | `backend/modules/preprocessing/audio_processor.py` |
| Trascrizione batched | `backend/modules/transcription/batched_pipeline.py` |
| Trascrizione custom | `backend/modules/transcription/custom_pipeline.py` |
| Formattazione output | `backend/modules/output/formatter.py` |
| Storage S3/MinIO | `backend/modules/storage/fetcher.py` |
| Entry point | `main.py` |
| Dipendenze | `pyproject.toml` |
| Configurazione env | `.env.local` |

### Moduli Esterni Chiave

| Import | Da |
|---|---|
| `FastAPI` | `fastapi` |
| `uvicorn` | `uvicorn` |
| `boto3` | `boto3` (storage S3, opzionale) |
| `WhisperModel`, `BatchedInferencePipeline` | `faster_whisper` |
| `np` | `numpy` |
| `sf` | `soundfile` |
| `ffmpeg` | `ffmpeg` |
| `nr` | `noisereduce` |
| `scipy.signal.resample` | `scipy.signal` |
| `Settings` | `pydantic_settings` |

### Comandi Utili

```bash
# Avviare il server
python main.py

# Eseguire i test (se presenti)
pytest

# Formattare il codice
ruff format .

# Linting
ruff check .

# Installare dipendenze
uv sync

# Aggiornare dipendenze
uv sync --upgrade
```

---

## Appendice A: Diagramma delle Dipendenze tra Moduli

```
main.py
  └── backend/api/routes.py
        └── backend/transcription_pipeline.py
              ├── backend/modules/preprocessing/audio_processor.py
              ├── backend/modules/transcription/whisper_engine.py
              │     ├── backend/modules/transcription/batched_pipeline.py
              │     └── backend/modules/transcription/custom_pipeline.py
              └── backend/modules/output/formatter.py
                    └── backend/modules/transcription/batched_pipeline.py
                          (TranscriptionSegment, TranscriptionResult)
              └── backend/modules/storage/fetcher.py  (opzionale, backward compat)

utils/config.py
  └── (dipende da pydantic, pydantic_settings)
```

## Appendice B: Stati del Task

```
POST /upload
    │
    ▼
┌────────────┐
│ PROCESSING │───────────────────────────────────────┐
└──────┬─────┘                                       │
       │                                             │
       │ (successo)                                  │ (errore)
       ▼                                             ▼
┌─────────────┐                            ┌─────────────┐
│ COMPLETED   │                            │   FAILED    │
└──────┬──────┘                            └─────────────┘
       │
       │ POST /convert/{id}/ack
       ▼
┌─────────────┐
│ ACKED*      │
└─────────────┘

* ACK è uno stato logico (flag `acknowledged`), non un task state enum.
```

## Appendice C: Formati di Output

### JSON
```json
{
  "segments": [
    {"start": 0.00, "end": 3.50, "text": "Ciao mondo"},
    {"start": 3.50, "end": 7.20, "text": "Come stai?"}
  ],
  "text": "Ciao mondo Come stai?",
  "segment_count": 2,
  "metadata": {...}
}
```

### SRT
```
1
00:00:00,000 --> 00:00:03,500
Ciao mondo

2
00:00:03,500 --> 00:00:07,200
Come stai?
```

### VTT
```
WEBVTT

00:00:00.000 --> 00:00:03.500
Ciao mondo

00:00:03.500 --> 00:00:07.200
Come stai?
```

### CSV
```
start,end,text
0.00,3.50,"Ciao mondo"
3.50,7.20,"Come stai?"
```

---

*Ultimo aggiornamento: 2025-07-28*
*Versione documento: 2.0*
