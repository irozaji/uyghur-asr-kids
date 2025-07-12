# Uyghur ASR for Kids (Ages 4-12) - Hugging Face Starter Project

## 📁 Directory Structure

```
uyghur-asr-kids/
├── README.md
├── requirements.txt
├── huggingface.yml
├── data/
│   ├── clips/
│   └── metadata.csv
├── training/
│   ├── fine_tune_whisper.py
│   └── preprocess.py
├── eval/
│   └── evaluate_wer.py
├── app/
│   ├── gradio_demo.py
│   └── static/
├── mobile_recorder/
│   ├── App.js
│   ├── firebase.js
│   └── components/
```

---

## ✅ 1. `README.md` (simplified content)

```markdown
# Uyghur ASR for Kids (Ages 4-12)
A community-driven, child-safe speech recognition model for Uyghur learners ages 4–12. Built using Whisper and crowd-collected voice data.
```

---

## ✅ 2. `requirements.txt`
```txt
transformers
datasets
torchaudio
accelerate
jiwer
gradio
```

---

## ✅ 3. `huggingface.yml`
```yaml
model:
  name: whisper-small-uyghur-kids
  license: apache-2.0
  tags:
    - whisper
    - uyghur
    - kids
    - speech
    - ASR
```

---

## ✅ 4. `training/fine_tune_whisper.py`
```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Trainer, TrainingArguments
from datasets import load_dataset

model = WhisperForConditionalGeneration.from_pretrained("ixxan/whisper-small-uyghur-common-voice")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

data = load_dataset("csv", data_files="../data/metadata.csv")

# Preprocessing function
# (similar to previous example)

args = TrainingArguments(
    output_dir="whisper-uyghur-kids",
    per_device_train_batch_size=4,
    num_train_epochs=5,
    learning_rate=1e-5,
    logging_steps=10,
    save_steps=100,
    fp16=True,
)

trainer = Trainer(model=model, args=args, train_dataset=data)
trainer.train()
```

---

## ✅ 5. `eval/evaluate_wer.py`
```python
from jiwer import wer
reference = "ياخشىمۇسىز"
hypothesis = "ياخشىسىز"
print("WER:", wer(reference, hypothesis))
```

---

## ✅ 6. `app/gradio_demo.py`
```python
import gradio as gr
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="ixxan/whisper-small-uyghur-common-voice")
def transcribe(audio):
    return asr(audio)["text"]

gr.Interface(fn=transcribe, inputs=gr.Audio(), outputs="text").launch()
```

---

## ✅ 7. `mobile_recorder/`
Built with React Native + Firebase:
- `App.js`: Microphone recorder + text prompt
- `firebase.js`: Connects to Firestore and Storage
- `components/Recorder.js`: Record & upload

---

This is your full **starter kit**. Let me know when you’re ready to:
- Publish the repo to GitHub
- Generate the mobile build configs (Firebase project setup + Expo export)
- Expand with scoring engine, evaluation dashboard, or teacher UI
