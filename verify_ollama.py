# 1. Impor library Llama
from llama_cpp import Llama

# 2. Tentukan path ke file model GGUF Anda
# Ganti dengan path yang Anda dapatkan dari LM Studio!
# model_path = "C:/Users/vince/.lmstudio/models/lmstudio-community/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q8_0.gguf"
model_path = "C:/Users/vince/.lmstudio/models/lmstudio-community/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q6_K.gguf"

# Contoh path di Mac/Linux: "/Users/vincent/.cache/lm-studio/models/..."

# 3. Muat model langsung dari path
#    - n_ctx: Ukuran konteks (berapa banyak token yang bisa diingat model). 2048 atau 4096 adalah nilai umum.
#    - verbose=False: Agar tidak menampilkan banyak log teknis saat loading.
print("Memuat model...")
llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    verbose=False
)
print("Model berhasil dimuat!")

# 4. Buat prompt dan kirim ke model
prompt = '''You are a game NPC dialogue generator. You will be given a character's personality, their original lie, the true fact the player used to expose the lie, and the desired outcome. Generate the NPC's next line of dialogue.

Character: Arrogant Antique Dealer
Original Lie: "This dagger is the authentic 'Serpent's Fang,' forged in the Majapahit era!"
Player's Rebuttal (based on True Fact): "The real Serpent's Fang has a dragon engraving, not a serpent."
Desired Outcome: Flustered and defensive
NPC Dialogue:'''

output = llm(
    prompt,
    max_tokens=100,      # Berapa banyak token maksimal untuk dihasilkan
    # stop=["\n", " "],  # Berhenti jika bertemu baris baru atau spasi (opsional)
)

# 5. Cetak hasilnya
# Strukturnya sedikit berbeda dari library openai
print("\nOutput dari model:")
print(output)

print(output["choices"][0])
print(output["choices"][0]['text'])