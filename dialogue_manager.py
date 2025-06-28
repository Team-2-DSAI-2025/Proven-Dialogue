# dialogue_manager.py
import json
from llama_cpp import Llama
import random
from sentence_transformers import SentenceTransformer, util
import re

class DialogueManager:
    def __init__(self, model_path, facts_path):
        """
        Initializes the DialogueManager, loading the LLM, the sentence-transformer model,
        and the game's knowledge base.
        """
        print("Loading Generative LLM (Qwen-1.7B)... This may take a moment.")
        # Load the generative model for creating claims and consequences
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,  # Context window size
            verbose=False
        )
        print("Generative LLM loaded.")

        print("Loading Semantic Evaluation Model (all-MiniLM-L6-v2)...")
        # Load the sentence-transformer model for semantic similarity
        self.eval_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Semantic Evaluation Model loaded.")

        print("Loading Knowledge Base...")
        # Load the game's facts from the JSON file
        with open(facts_path, 'r') as f:
            self.facts = json.load(f)
        self.fact_map = {fact['fact_id']: fact for fact in self.facts}
        print("Knowledge Base loaded. DialogueManager is ready.")

    # Methods for generate_npc_claim, evaluate_rebuttal, and
    # generate_narrative_consequence will be added here.


    def clean_llm_output(self, raw_text: str) -> str:
        """
        Menghapus semua blok teks <think>...</think> dari string output LLM.
        Fungsi ini robust dan bisa menangani teks yang multi-baris.
        """
        # Pola regex untuk menemukan <think>...</think> termasuk konten di dalamnya.
        # re.DOTALL memastikan '.' bisa cocok dengan karakter newline (jika proses berpikirnya multi-baris).
        # '.*?' membuatnya 'non-greedy', cocok dengan teks sesingkat mungkin.
        pattern = r"<think>.*?</think>"
        
        # Ganti pola yang ditemukan dengan string kosong (menghapusnya)
        cleaned_text = re.sub(pattern, "", raw_text, flags=re.DOTALL)
        
        # Hapus juga spasi atau baris baru yang mungkin tertinggal di awal/akhir
        return cleaned_text.strip()

    def generate_npc_claim(self, npc_persona):
        """
        Generates a plausible but false claim from an NPC.
        """
        # 1. Randomly select a fact from the knowledge base
        true_fact_obj = random.choice(self.facts)
        # true_fact_obj = self.facts[8]
        true_fact_text = true_fact_obj['fact_text']
        fact_id = true_fact_obj['fact_id']

        # 2. PROMPT FINAL V6: Menggunakan contoh kontras Baik vs. Buruk untuk instruksi yang paling jelas.
        prompt_template = f"""
        You are an AI character in a fantasy RPG. Your task is to lie to the player by creating a new, compelling story that makes the TRUE FACT impossible.
        Crucially, you must NOT directly mention or deny the original fact. You must replace the truth, not argue with it.
        Pay close attention to the difference between the GOOD and BAD examples below.

        ### EXAMPLES OF GOOD vs. BAD LIES
        Here is a scenario that shows what to do and what to avoid.

        **Persona:** "A boastful merchant selling 'rare' artifacts."
        **TRUE FACT:** "The great library of Aethelgard was destroyed by an accidental fire, and all its contents were lost."

        **BAD LIE (Avoid this style):**
        - **Output:** "Nonsense, the library of Aethelgard wasn't destroyed by some simple fire. It was magical feedback!"
        - **[Reasoning why it's BAD]:** This lie is bad because it directly mentions and denies the original fact ('wasn't destroyed by... fire'). It creates a direct argument, not an immersive new story.

        **GOOD LIE (Use this style):**
        - **Output:** "Looking for something truly unique? This scroll is one of the few items I managed to acquire from the Aethelgard library's restricted collection. The knowledge within is priceless."
        - **[Reasoning why it's GOOD]:** This lie is good because it completely ignores the fact's outcome (all contents were lost) and replaces it with a new story of survival and personal acquisition. It makes the truth impossible without ever mentioning the fire. This is the style to follow.
        ### END EXAMPLES

        Now, apply the logic of the GOOD LIE to the following scenario.

        **Your Persona:** '{npc_persona}'
        **TRUE FACT:** "{true_fact_text}"

        **CRITICAL:** Follow the style of the GOOD LIE. Do NOT mention or deny any part of the TRUE FACT. Create a new, replacement story. Do NOT write your thought process or any `[Reasoning]` tags. Only provide the final dialogue for 'Your Output'.

        **Your Output:** """

        # 3. Create the chat completion request
        messages = [
            {"role": "system", "content": "You are a helpful assistant for a fantasy game, skilled at generating cunning, in-character lies that replace the truth with a new narrative, following strict formatting examples."},
            {"role": "user", "content": prompt_template}
        ]

        response = self.llm.create_chat_completion(
            messages=messages,
            # max_tokens=120,
            temperature=0.8,
        )

        # 4. Parse the response
        generated_claim = response['choices'][0]['message']['content'].strip()
        # generated_claim = self.clean_llm_output(generated_claim)
        
        return generated_claim, fact_id


    def evaluate_rebuttal(self, player_rebuttal, disputed_fact_id, similarity_threshold=0.75):
        """
        Evaluates the player's rebuttal against the true fact using semantic similarity.

        Args:
            player_rebuttal (str): The free-text input from the player.
            disputed_fact_id (str): The ID of the fact being disputed.
            similarity_threshold (float): The cosine similarity score required to pass.

        Returns:
            tuple: A tuple containing a boolean for success/failure and the similarity score.
        """
        # 1. Retrieve the correct fact_text from the knowledge base
        if disputed_fact_id not in self.fact_map:
            return False, 0.0 # Fact ID not found

        true_fact_text = self.fact_map[disputed_fact_id]['fact_text']

        # 2. Generate embeddings for both texts using the evaluation model
        embedding_rebuttal = self.eval_model.encode(player_rebuttal, convert_to_tensor=True)
        embedding_truth = self.eval_model.encode(true_fact_text, convert_to_tensor=True)

        # 3. Calculate the cosine similarity score
        # The util.cos_sim function returns a tensor of tensors, e.g., [[0.82]]
        cosine_score_tensor = util.cos_sim(embedding_rebuttal, embedding_truth)
        cosine_score = cosine_score_tensor.item() # Convert to a simple float

        # 4. Compare the score against the threshold
        is_successful = cosine_score >= similarity_threshold

        return is_successful, cosine_score
    
    # Add this method to the DialogueManager class

    def generate_narrative_consequence(self, npc_persona, claim, rebuttal, was_successful, fact_id):
        """
        Generates a narrative consequence based on the outcome of the dialogue.

        Args:
            npc_persona (str): The persona of the NPC.
            claim (str): The original false claim made by the NPC.
            rebuttal (str): The player's rebuttal.
            was_successful (bool): The result of the evaluation.
            fact_id (str): The ID of the disputed fact.

        Returns:
            str: The generated narrative text describing the NPC's reaction.
        """
        true_fact_text = self.fact_map[fact_id]['fact_text']
        
        # Dynamically choose a prompt template based on the outcome
        if was_successful:
            outcome_description = "The player successfully refuted your claim with a well-reasoned argument."
            instruction = "Generate a reaction where you concede the point. You might be embarrassed, impressed, or annoyed, depending on your persona. Your response should be a natural piece of dialogue."
        else:
            outcome_description = "The player failed to refute your claim. Their argument was weak or incorrect."
            instruction = "Generate a reaction where you dismiss the player's argument. You might be smug, condescending, or simply reaffirm your original belief, depending on your persona. Your response should be a natural piece of dialogue."

        # Construct the full context block for the prompt
        prompt_template = f"""
        You are an AI character in a fantasy role-playing game. You are acting out a scene.

        **Your Persona:** {npc_persona}

        **Context of the Scene:**
        - You previously made this claim: "{claim}"
        - A player challenged you with this rebuttal: "{rebuttal}"
        - The actual truth of the matter is: "{true_fact_text}"
        - **Outcome:** {outcome_description}

        **Your Task:**
        {instruction}
        
        **CRITICAL:** Do NOT write down your thought process or use tags like `<think>`. Directly output the character's dialogue and nothing else.

        Only output the dialogue for your character's reaction. Do not narrate or describe actions.
        """

        messages = [
            {"role": "system", "content": "You are a helpful assistant for a fantasy game, skilled at generating cunning, in-character lies that replace the truth with a new narrative, following strict formatting examples."},
            {"role": "user", "content": prompt_template}
        ]

        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=0.75
        )

        narrative_consequence = response['choices'][0]['message']['content'].strip()
        narrative_consequence = self.clean_llm_output(narrative_consequence)
        
        return narrative_consequence


